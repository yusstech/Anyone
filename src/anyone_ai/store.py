"""Local SQLite evidence store with FTS5 retrieval. One file per persona."""
from __future__ import annotations

import datetime as _dt
import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .redact import redact_names as _redact_names, redact_pii as _redact_pii
from .text import Chunk, chunk_text, clean_text, extract, query_terms

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  title TEXT NOT NULL,
  date TEXT,
  sha256 TEXT NOT NULL UNIQUE,
  added_at TEXT NOT NULL,
  excluded INTEGER NOT NULL DEFAULT 0,
  text TEXT NOT NULL,
  authorship TEXT NOT NULL DEFAULT 'by_me',
  origin TEXT NOT NULL DEFAULT 'file',
  ownership_proof TEXT NOT NULL DEFAULT 'local_file',
  url TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  start INTEGER NOT NULL,
  end INTEGER NOT NULL,
  text TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize='porter unicode61');
"""

# Columns added after v0.1.0.dev0; older indexes are migrated in place.
_SOURCE_MIGRATIONS = {
    "authorship": "TEXT NOT NULL DEFAULT 'by_me'",
    "origin": "TEXT NOT NULL DEFAULT 'file'",
    "ownership_proof": "TEXT NOT NULL DEFAULT 'local_file'",
    "url": "TEXT",
}

# by_me: the owner wrote it; verbatim quotes allowed.
# ai_assisted: drafted with AI or others; usable as context, never shown as the owner's verbatim words.
AUTHORSHIP = ("by_me", "ai_assisted")


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    source_id: str
    title: str
    date: str | None
    path: str
    text: str
    score: float = 0.0
    authorship: str = "by_me"
    url: str | None = None

    @property
    def quotable(self) -> bool:
        return self.authorship == "by_me"


@dataclass(frozen=True)
class SourceRow:
    id: str
    path: str
    title: str
    date: str | None
    added_at: str
    excluded: bool
    chunks: int
    sha256: str
    authorship: str = "by_me"
    origin: str = "file"
    ownership_proof: str = "local_file"
    url: str | None = None


@dataclass
class IngestResult:
    added: list[SourceRow]
    skipped_duplicates: list[str]
    errors: list[tuple[str, str]]
    replaced: list[str] = field(default_factory=list)  # old source IDs superseded by a re-ingested file


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Callers that share a Store across threads (MCP, web) must serialize access themselves.
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(sources)")}
        with self.conn:
            for col, decl in _SOURCE_MIGRATIONS.items():
                if col not in have:
                    self.conn.execute(f"ALTER TABLE sources ADD COLUMN {col} {decl}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- ingest

    def ingest_paths(self, paths: list[Path], redact_names: list[str] | None = None,
                     date_override: str | None = None, authorship: str = "by_me",
                     redact_pii: bool = True) -> IngestResult:
        result = IngestResult([], [], [])
        for path in _expand(paths):
            path = path.resolve()
            try:
                ex = extract(path)
                row = self.ingest_text(str(path), ex.title, date_override or ex.date, ex.text, result=result,
                                       redact_names=redact_names, redact_pii=redact_pii, authorship=authorship)
            except (OSError, ValueError) as exc:
                result.errors.append((str(path), str(exc)))
                continue
            if row is None:
                result.skipped_duplicates.append(str(path))
            else:
                result.added.append(row)
        return result

    def ingest_text(self, key: str, title: str, date: str | None, text: str, *, result: IngestResult | None = None,
                    redact_names: list[str] | None = None, redact_pii: bool = True,
                    authorship: str = "by_me", origin: str = "file",
                    ownership_proof: str = "local_file", url: str | None = None) -> SourceRow | None:
        """Add one document. ``key`` identifies it across re-imports (file path or canonical URL).

        Returns None if identical content is already indexed. If ``key`` exists with different
        content (edited, or new redaction rules), the new version replaces the old one.
        Redaction (names, and emails/phones/addresses when ``redact_pii``) happens before
        hashing, storage and indexing.
        """
        if authorship not in AUTHORSHIP:
            raise ValueError(f"authorship must be one of {AUTHORSHIP}")
        text = clean_text(text)
        if redact_pii:
            text = _redact_pii(text)
        text = _redact_names(text, redact_names or [])
        if not text:
            raise ValueError("no extractable text")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if self.conn.execute("SELECT 1 FROM sources WHERE sha256 = ?", (digest,)).fetchone():
            return None
        for (old_id,) in self.conn.execute("SELECT id FROM sources WHERE path = ?", (key,)).fetchall():
            self.remove_source(old_id)
            if result is not None:
                result.replaced.append(old_id)
        source_id = f"src-{digest[:10]}"
        added_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        chunks: list[Chunk] = chunk_text(text)
        with self.conn:
            self.conn.execute(
                "INSERT INTO sources (id, path, title, date, sha256, added_at, text, authorship, origin, "
                "ownership_proof, url) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (source_id, key, title, date, digest, added_at, text, authorship, origin, ownership_proof, url),
            )
            for c in chunks:
                chunk_id = f"{source_id}#{c.ordinal}"
                self.conn.execute(
                    "INSERT INTO chunks (id, source_id, ordinal, start, end, text) VALUES (?,?,?,?,?,?)",
                    (chunk_id, source_id, c.ordinal, c.start, c.end, c.text),
                )
                self.conn.execute("INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)", (chunk_id, c.text))
        return self.get_source(source_id)

    def keys_with_prefix(self, prefix: str) -> dict[str, str]:
        """Map key -> source id for all sources whose key starts with ``prefix`` (e.g. a site URL)."""
        rows = self.conn.execute("SELECT path, id FROM sources WHERE substr(path, 1, ?) = ?", (len(prefix), prefix))
        return {r[0]: r[1] for r in rows}

    # --------------------------------------------------------------- sources

    def list_sources(self) -> list[SourceRow]:
        rows = self.conn.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM chunks c WHERE c.source_id = s.id) AS n "
            "FROM sources s ORDER BY COALESCE(s.date, ''), s.added_at"
        ).fetchall()
        return [_source_row(r) for r in rows]

    def get_source(self, source_id: str) -> SourceRow | None:
        r = self.conn.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM chunks c WHERE c.source_id = s.id) AS n FROM sources s WHERE s.id = ?",
            (source_id,),
        ).fetchone()
        return _source_row(r) if r else None

    def set_authorship(self, source_id: str, authorship: str) -> bool:
        if authorship not in AUTHORSHIP:
            raise ValueError(f"authorship must be one of {AUTHORSHIP}")
        with self.conn:
            cur = self.conn.execute("UPDATE sources SET authorship = ? WHERE id = ?", (authorship, source_id))
        return cur.rowcount == 1

    def set_excluded(self, source_id: str, excluded: bool) -> bool:
        with self.conn:
            cur = self.conn.execute("UPDATE sources SET excluded = ? WHERE id = ?", (int(excluded), source_id))
        return cur.rowcount == 1

    def remove_source(self, source_id: str) -> bool:
        with self.conn:
            self.conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE source_id = ?)", (source_id,)
            )
            cur = self.conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        return cur.rowcount == 1

    def stats(self) -> dict:
        s = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(excluded), 0) FROM sources").fetchone()
        c = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return {"sources": s[0], "excluded": s[1], "chunks": c[0]}

    def source_hashes(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT sha256 FROM sources ORDER BY sha256")]

    # ------------------------------------------------------------- retrieval

    def search(self, query: str, k: int = 6) -> list[Evidence]:
        terms = query_terms(query)
        if not terms:
            return []
        # Every term is double-quoted, so user text can never inject FTS5 syntax.
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        rows = self.conn.execute(
            """
            SELECT c.id AS chunk_id, c.text, s.id AS source_id, s.title, s.date, s.path,
                   s.authorship, s.url, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            JOIN sources s ON s.id = c.source_id
            WHERE chunks_fts MATCH ? AND s.excluded = 0
            ORDER BY score, c.id
            LIMIT ?
            """,
            (match, k),
        ).fetchall()
        return [
            Evidence(chunk_id=r["chunk_id"], source_id=r["source_id"], title=r["title"], date=r["date"],
                     path=r["path"], text=r["text"], score=round(-r["score"], 4),
                     authorship=r["authorship"], url=r["url"])
            for r in rows
        ]



def _expand(paths: list[Path]) -> list[Path]:
    from .text import SUPPORTED_SUFFIXES

    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*")
                              if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
                              and not any(part.startswith(".") for part in f.relative_to(p).parts)))
        else:
            out.append(p)
    return out


def _source_row(r: sqlite3.Row) -> SourceRow:
    return SourceRow(id=r["id"], path=r["path"], title=r["title"], date=r["date"], added_at=r["added_at"],
                     excluded=bool(r["excluded"]), chunks=r["n"], sha256=r["sha256"], authorship=r["authorship"],
                     origin=r["origin"], ownership_proof=r["ownership_proof"], url=r["url"])
