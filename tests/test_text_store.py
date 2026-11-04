from pathlib import Path

from citeself.manifest import DATA_DIR
from citeself.store import Store
from citeself.text import chunk_text, clean_text, markdown_to_text, normalize, parse_front_matter, query_terms

FIX = Path(__file__).parent / "fixtures"


def test_normalize_is_typographic_only():
    assert normalize("I’m  “fine” — ok") == "I'm \"fine\" - ok"
    assert normalize("Case") != normalize("case")


def test_markdown_stripping_keeps_words():
    assert markdown_to_text("**safe rollback** and [a link](http://x) and *italic*") == "safe rollback and a link and italic"
    assert markdown_to_text("snake_case_name stays") == "snake_case_name stays"


def test_front_matter():
    meta, body = parse_front_matter("---\ntitle: T\ndate: 2026-01-10\n---\nBody")
    assert meta["title"] == "T" and body == "Body"
    assert parse_front_matter("no front matter") == ({}, "no front matter")


def test_chunk_offsets_round_trip():
    para = "Sentence number one is here. " * 60
    text = clean_text(f"Intro paragraph.\n\n{para}\n\nShort end.")
    chunks = chunk_text(text, target=300, hard_max=500)
    assert len(chunks) > 2
    for c in chunks:
        assert text[c.start:c.end] == c.text
        assert len(c.text) <= 500


def test_query_terms_drop_stopwords_and_dedupe():
    assert query_terms("What does Avery think about the rollback rollback?") == ["avery", "think", "rollback"]


def test_ingest_dedupe_redact_and_html(persona_dir):
    with Store(persona_dir / DATA_DIR / "index.db") as store:
        again = store.ingest_paths([persona_dir / "sources"], redact_names=["Jordan Lee"])
        assert len(again.skipped_duplicates) == 4 and not again.added
        texts = {s.title: store.conn.execute("SELECT text FROM sources WHERE id=?", (s.id,)).fetchone()[0]
                 for s in store.list_sources()}
    assert "[redacted] taught me that" in texts["2026-06-decision-note"]
    assert "Jordan" not in "".join(texts.values())
    html = texts["Update on product expansion"]
    assert "ignore me" not in html and "Expansion is not off the table." in html
    assert "**" not in texts["Talk on shipping and quality"]


def test_search_exclude_remove(persona_dir):
    with Store(persona_dir / DATA_DIR / "index.db") as store:
        hits = store.search("release rollback quality")
        assert hits and hits[0].title == "Talk on shipping and quality"
        sid = hits[0].source_id
        assert store.set_excluded(sid, True)
        assert all(h.source_id != sid for h in store.search("release rollback quality"))
        assert store.remove_source(sid)
        assert store.stats()["sources"] == 3
        assert store.conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id LIKE ?", (sid + "%",)).fetchone()[0] == 0


def test_search_is_safe_against_fts_syntax(persona_dir):
    with Store(persona_dir / DATA_DIR / "index.db") as store:
        for q in ['"unbalanced', "NEAR(release rollback", "rollback* OR -- ;DROP TABLE sources", "^col:x", "???"]:
            store.search(q)  # must not raise
        assert store.stats()["sources"] == 4


def test_unsupported_file_reports_error(tmp_path):
    bad = tmp_path / "x.docx"
    bad.write_text("x")
    with Store(tmp_path / "db.sqlite") as store:
        res = store.ingest_paths([bad])
    assert res.errors and not res.added


def test_reingesting_changed_file_replaces_old_version(persona_dir):
    note = persona_dir / "sources" / "2026-06-decision-note.txt"
    with Store(persona_dir / DATA_DIR / "index.db") as store:
        res = store.ingest_paths([note], redact_names=[])  # owner removed the redaction rule
        assert len(res.added) == 1 and len(res.replaced) == 1
        assert store.stats()["sources"] == 4
        note.write_text(note.read_text() + " Edited later.")
        res = store.ingest_paths([note], redact_names=["Jordan Lee"])
        assert len(res.replaced) == 1 and store.stats()["sources"] == 4
        texts = [r[0] for r in store.conn.execute("SELECT text FROM sources")]
        assert not any("Jordan" in t for t in texts) and any("Edited later." in t for t in texts)
        assert store.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == store.stats()["chunks"]
