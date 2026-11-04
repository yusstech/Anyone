"""Local SQLite evidence store with FTS5 retrieval. One file per persona."""
from __future__ import annotations

import datetime as _dt
import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

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
  text TEXT NOT NULL
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


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    source_id: str
    title: str
    date: str | None
    path: str
    text: str
    score: float = 0.0


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

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- ingest

    def ingest_paths(self, paths: list[Path], redact_names: list[str] | None = None,
                     date_override: str | None = None) -> IngestResult:
        result = IngestResult([], [], [])
        for path in _expand(paths):
            path = path.resolve()
            try:
                row = self._ingest_file(path, redact_names or [], date_override, result)
            except (OSError, ValueError) as exc:
                result.errors.append((str(path), str(exc)))
                continue
            if row is None:
                result.skipped_duplicates.append(str(path))
            else:
                result.added.append(row)
        return result

    def _ingest_file(self, path: Path, redact_names: list[str], date_override: str | None,
                     result: IngestResult) -> SourceRow | None:
        ex = extract(path)
        text = redact(clean_text(ex.text), redact_names)
        if not text:
            raise ValueError("no extractable text")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if self.conn.execute("SELECT 1 FROM sources WHERE sha256 = ?", (digest,)).fetchone():
            return None
        # Same file, new content (edited, or new redaction rules): the new version replaces the old,
        # so stale or unredacted text never lingers in the index.
        for (old_id,) in self.conn.execute("SELECT id FROM sources WHERE path = ?", (str(path),)).fetchall():
            self.remove_source(old_id)
            result.replaced.append(old_id)
        source_id = f"src-{digest[:10]}"
        added_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        chunks: list[Chunk] = chunk_text(text)
        with self.conn:
            self.conn.execute(
                "INSERT INTO sources (id, path, title, date, sha256, added_at, text) VALUES (?,?,?,?,?,?,?)",
                (source_id, str(path), ex.title, date_override or ex.date, digest, added_at, text),
            )
            for c in chunks:
                chunk_id = f"{source_id}#{c.ordinal}"
                self.conn.execute(
                    "INSERT INTO chunks (id, source_id, ordinal, start, end, text) VALUES (?,?,?,?,?,?)",
                    (chunk_id, source_id, c.ordinal, c.start, c.end, c.text),
                )
                self.conn.execute("INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)", (chunk_id, c.text))
        return self.get_source(source_id)

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
                   bm25(chunks_fts) AS score
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
                     path=r["path"], text=r["text"], score=round(-r["score"], 4))
            for r in rows
        ]


def redact(text: str, names: list[str]) -> str:
    for name in sorted({n for n in names if n.strip()}, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(name.strip())}(?!\w)", "[redacted]", text, flags=re.I)
    return text


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
                     excluded=bool(r["excluded"]), chunks=r["n"], sha256=r["sha256"])
