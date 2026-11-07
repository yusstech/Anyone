import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from anyone_ai.cli import main
from anyone_ai.importers.site import canonical, import_site, parse_feed, parse_sitemap
from anyone_ai.manifest import DATA_DIR
from anyone_ai.store import Store

WORDS = " ".join(["Shipping small experiments beats long plans because evidence compounds."] * 12)


def page(title, body, date=None):
    meta = f'<meta property="article:published_time" content="{date}"/>' if date else ""
    return (f"<html><head><title>{title}</title>{meta}</head><body><nav>Home Writing About</nav>"
            f"<main><h1>{title}</h1><p>{body}</p></main><footer>Footer</footer></body></html>")


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "site"
    (root / "writing").mkdir(parents=True)
    (root / "secret").mkdir()
    # Allow: / first, then a more specific Disallow -> RFC 9309 longest match must block /secret
    (root / "robots.txt").write_text("User-agent: *\nAllow: /\nDisallow: /secret\n")
    (root / "index.html").write_text(
        '<html><head><title>Me</title><link rel="alternate" type="application/rss+xml" href="/feed.xml"></head>'
        f'<body><main><p>{WORDS}</p><a href="/writing/post-two.html">two</a><a href="https://evil.example/x">x</a>'
        '<a href="/privacy.html">privacy</a><a href="/short.html">short</a></main></body></html>')
    (root / "writing" / "post-one.html").write_text(page("Post one", "SHOULD NOT BE FETCHED: feed has full text"))
    (root / "writing" / "post-two.html").write_text(page("Post two", WORDS + " Second post.", "2026-02-03"))
    (root / "secret" / "draft.html").write_text(page("Secret", WORDS))
    (root / "privacy.html").write_text(page("Privacy", WORDS))
    (root / "short.html").write_text(page("Short", "Too short."))
    (root / "feed.xml").write_text(
        '<?xml version="1.0"?><rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>'
        "<item><title>Post one</title><link>/writing/post-one.html</link><pubDate>Tue, 06 Jan 2026 10:00:00 GMT</pubDate>"
        f"<content:encoded><![CDATA[<p>{WORDS} First post full text.</p>]]></content:encoded></item>"
        "</channel></rss>")
    (root / "sitemap.xml").write_text(
        '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>PLACEHOLDER/writing/post-two.html</loc></url><url><loc>PLACEHOLDER/secret/draft.html</loc></url>"
        "</urlset>")
    handler = functools.partial(_QuietHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    base = f"http://127.0.0.1:{httpd.server_port}"
    sm = root / "sitemap.xml"
    sm.write_text(sm.read_text().replace("PLACEHOLDER", base))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield base
    httpd.shutdown()
    httpd.server_close()


class _QuietHandler(SimpleHTTPRequestHandler):
    requested: list[str] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        _QuietHandler.requested.append(self.path)
        super().do_GET()


def test_import_site_discovery_and_rules(site):
    _QuietHandler.requested.clear()
    res = import_site(site + "/index.html", delay=0)
    urls = {p.url: p for p in res.pages}
    one, two = canonical(site + "/writing/post-one.html"), canonical(site + "/writing/post-two.html")
    assert one in urls and two in urls
    assert urls[one].via == "feed" and "First post full text." in urls[one].text and urls[one].date == "2026-01-06"
    assert urls[two].date == "2026-02-03" and "Home Writing" not in urls[two].text and "Footer" not in urls[two].text
    skipped = dict(res.skipped)
    assert skipped[canonical(site + "/secret/draft.html")] == "disallowed by robots.txt"
    assert "not content" in skipped[canonical(site + "/privacy.html")]
    assert "too short" in skipped[canonical(site + "/short.html")]
    assert "/secret/draft.html" not in _QuietHandler.requested          # never even requested
    assert "/writing/post-one.html" not in _QuietHandler.requested      # full text came from the feed
    assert not any("evil" in u for u in urls)                           # off-site links ignored
    assert any(v.startswith("feed") for v in res.discovered_via) and any(v.startswith("sitemap") for v in res.discovered_via)


def test_exclude_and_max_pages(site):
    res = import_site(site + "/index.html", delay=0, exclude=[r"post-two"], max_pages=1)
    reasons = [r for _, r in res.skipped]
    assert "excluded by --exclude" in reasons and len(res.pages) <= 1


def test_cli_add_url_then_resync(site, tmp_path, capsys):
    me = tmp_path / "me"
    assert main(["-C", str(me), "init", "--name", "Test Person", "--i-am-this-person"]) == 0
    assert main(["-C", str(me), "add", site + "/index.html", "--dry-run", "--delay", "0"]) == 0
    assert "Dry run" in capsys.readouterr().out
    with Store(me / DATA_DIR / "index.db") as s:
        assert s.stats()["sources"] == 0
    assert main(["-C", str(me), "add", site + "/index.html", "--authorship", "by_me", "--delay", "0"]) == 0
    with Store(me / DATA_DIR / "index.db") as s:
        rows = s.list_sources()
        assert len(rows) == 3  # home, post one, post two
        assert all(r.origin == "site" and r.ownership_proof == "unverified_site" and r.url for r in rows)
    capsys.readouterr()
    assert main(["-C", str(me), "add", site + "/index.html", "--authorship", "by_me", "--delay", "0"]) == 0  # re-run = sync
    assert "0 added, 0 updated, 3 unchanged" in capsys.readouterr().out
    sid = rows[0].id
    assert main(["-C", str(me), "authorship", sid, "ai_assisted"]) == 0
    with Store(me / DATA_DIR / "index.db") as s:
        assert s.get_source(sid).authorship == "ai_assisted"


def test_feed_and_sitemap_parsers_refuse_entities():
    bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><rss><channel><item><link>/a</link></item></channel></rss>'
    assert parse_feed(bomb, "http://x") == []
    assert parse_sitemap(bomb.replace("rss", "urlset")) == ([], [])
    atom = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>T</title><link href="/p"/>'
            "<published>2025-04-01T00:00:00Z</published><content>Body</content></entry></feed>")
    item = parse_feed(atom, "http://x")[0]
    assert item["url"] == "http://x/p" and item["date"] == "2025-04-01" and item["full"]


def test_migrates_old_index(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
      CREATE TABLE sources (id TEXT PRIMARY KEY, path TEXT NOT NULL, title TEXT NOT NULL, date TEXT,
        sha256 TEXT NOT NULL UNIQUE, added_at TEXT NOT NULL, excluded INTEGER NOT NULL DEFAULT 0, text TEXT NOT NULL);
      INSERT INTO sources VALUES ('src-1','/a','A',NULL,'h','2026-01-01',0,'text');
    """)
    con.commit()
    con.close()
    with Store(db) as s:
        row = s.get_source("src-1")
    assert row.authorship == "by_me" and row.origin == "file"
