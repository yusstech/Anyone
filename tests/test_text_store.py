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
    bad = tmp_path / "x.pages"
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


def test_html_keeps_main_drops_chrome_and_reads_date(tmp_path):
    from citeself.text import extract

    page = tmp_path / "post.html"
    page.write_text(
        '<html><head><title>My Post — Site</title><meta property="article:published_time" content="2026-08-27T10:00:00Z"/>'
        '</head><body><header><nav>Home Writing Contact</nav></header>'
        '<main><article><h1>My Post</h1><p>Real words here.<br>Second line.</p><form><button>Subscribe</button></form>'
        '</article></main><footer>(c) 2026 Footer text</footer></body></html>')
    ex = extract(page)
    assert "Real words here." in ex.text and "Second line." in ex.text
    assert "Home Writing" not in ex.text and "Footer text" not in ex.text and "Subscribe" not in ex.text
    assert ex.date == "2026-08-27" and ex.title == "My Post — Site"


def test_html_without_main_falls_back_to_body(tmp_path):
    from citeself.text import extract

    page = tmp_path / "plain.html"
    page.write_text("<html><body><nav>menu</nav><p>Only paragraph.</p></body></html>")
    ex = extract(page)
    assert "Only paragraph." in ex.text and "menu" not in ex.text


def _make_docx(path, paragraphs, title="Voice guide", created="2026-05-01T09:00:00Z", extra=b""):
    import zipfile

    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{t}</w:t></w:r></w:p>" for t in paragraphs)
    doc = f'<?xml version="1.0"?>{extra.decode()}<w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>'
    core = ('<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/">'
            f"<dc:title>{title}</dc:title><dcterms:created>{created}</dcterms:created></cp:coreProperties>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", doc)
        z.writestr("docProps/core.xml", core)


def test_docx_extraction(tmp_path):
    from citeself.text import extract

    f = tmp_path / "guide.docx"
    _make_docx(f, ["Write plainly.", "Lead with the customer problem."])
    ex = extract(f)
    assert ex.text == "Write plainly.\n\nLead with the customer problem."
    assert ex.title == "Voice guide" and ex.date == "2026-05-01"


def test_docx_rejects_dtd_and_garbage(tmp_path):
    import pytest

    from citeself.text import extract

    bad = tmp_path / "bomb.docx"
    _make_docx(bad, ["x"], extra=b'<!DOCTYPE lolz [<!ENTITY lol "lol">]>')
    with pytest.raises(ValueError, match="DTD"):
        extract(bad)
    junk = tmp_path / "junk.docx"
    junk.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="valid .docx"):
        extract(junk)
