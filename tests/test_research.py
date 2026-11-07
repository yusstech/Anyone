"""Research pipeline: discovery filtering, X oEmbed parsing, classification and secondhand quotes. Offline."""
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace as NS

import pytest
import yaml

from anyone_ai.cli import main
from anyone_ai.engine import Persona
from anyone_ai.manifest import Manifest, Research, Subject, index_path, load_manifest, write_new_manifest
from anyone_ai.providers import AnthropicProvider, GenerationResult, ProviderError
from anyone_ai.research import pipeline
from anyone_ai.research.collect import Collected, CollectError, check_public_url, parse_oembed
from anyone_ai.research.discover import Candidate, Discovery, discover
from anyone_ai.store import Evidence, Store
from anyone_ai.validate import validate_segments

from .conftest import FakeProvider

OEMBED = {"url": "https://x.com/elonmusk/status/1585341984679469056", "author_name": "Elon Musk",
          "author_url": "https://x.com/elonmusk",
          "html": '<blockquote class="twitter-tweet" data-dnt="true"><p lang="en" dir="ltr">Entering Twitter HQ &#8211; '
                  'let that sink in! <a href="https://t.co/D68z4K2wq7">pic.twitter.com/D68z4K2wq7</a></p>&mdash; Elon Musk '
                  '(@elonmusk) <a href="https://x.com/elonmusk/status/1585341984679469056?ref_src=twsrc%5Etfw">October 26, '
                  "2022</a></blockquote>\n"}


def test_parse_oembed_takes_text_author_and_date():
    text, handle, date, author = parse_oembed(OEMBED)
    assert text == "Entering Twitter HQ – let that sink in!"
    assert (handle, date, author) == ("elonmusk", "2022-10-26", "Elon Musk")
    with pytest.raises(CollectError):
        parse_oembed({"html": None})


@pytest.mark.parametrize("url", ["http://127.0.0.1/x", "http://169.254.169.254/latest/meta-data", "file:///etc/passwd",
                                 "http://localhost:8000/", "https://user:pw@example.com/", "https://www.spokeo.com/Jane-Doe",
                                 "http://[::1]/", "http://10.0.0.5/admin", "http://printer.local/"])
def test_unsafe_urls_refused(url):
    with pytest.raises(CollectError):
        check_public_url(url)


def test_public_urls_allowed():
    check_public_url("https://paulgraham.com/ds.html")
    check_public_url("http://127.0.0.1:9/x", allow_private=True)  # tests only


# ------------------------------------------------------------------ discovery

def _resp(content, stop="end_turn"):
    return NS(content=content, stop_reason=stop, model="claude-opus-5-5",
              usage=NS(model_dump=lambda: {"input_tokens": 100, "output_tokens": 10}))


def _search(*urls):
    return [NS(type="server_tool_use", name="web_search"),
            NS(type="web_search_tool_result", content=[NS(type="web_search_result", url=u) for u in urls])]


class FakeMessages:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def fake_provider(responses):
    msgs = FakeMessages(responses)
    client = NS(messages=msgs, beta=NS(messages=msgs))
    return AnthropicProvider(Manifest(subject=Subject(kind="public_figure", name="Dana Reyes")).model, client=client), msgs


def manifest(**research):
    return Manifest(subject=Subject(kind="public_figure", name="Dana Reyes"), research=Research(**research))


def test_discover_keeps_only_urls_seen_in_search_results():
    answer = {"candidates": [
        {"url": "https://dana.blog/essay", "type": "own_writing", "title": "Essay", "reason": "her blog"},
        {"url": "https://news.example/interview", "type": "own_writing", "title": "Interview", "reason": ""},
        {"url": "https://invented.example/fake", "type": "reporting", "title": "Made up", "reason": ""},
        {"url": "https://x.com/danareyes/status/123", "type": "reporting", "title": "post", "reason": ""},
        {"url": "https://www.spokeo.com/Dana-Reyes", "type": "reporting", "title": "broker", "reason": ""},
        {"url": "javascript:alert(1)", "type": "reporting", "title": "x", "reason": ""}],
        "x_handles": ["@danareyes", "not a handle!"], "identity_notes": "Matched by employer."}
    provider, msgs = fake_provider([
        _resp(_search("https://dana.blog/essay"), stop="pause_turn"),  # server paused mid-research
        _resp(_search("https://news.example/interview", "https://x.com/danareyes/status/123",
                      "https://www.spokeo.com/Dana-Reyes") + [NS(type="text", text=json.dumps(answer), citations=None)]),
    ])
    d = discover(provider, manifest(links=["dana.blog"], hint="CEO of Acme"), max_searches=3)

    assert [(c.url, c.type, c.from_user) for c in d.candidates] == [
        ("https://dana.blog/", "own_writing", True),
        ("https://dana.blog/essay", "own_writing", False),
        ("https://news.example/interview", "own_writing", False),
        ("https://x.com/danareyes/status/123", "x_post", False),  # type corrected from the URL
    ]
    assert d.dropped_unseen == ["https://invented.example/fake"]
    assert d.proposed_x_handles == ["danareyes"] and d.searches == 2 and d.identity_notes == "Matched by employer."

    first, second = msgs.calls
    tool = first["tools"][0]
    assert tool["type"] == "web_search_20260209" and tool["max_uses"] == 3 and "spokeo.com" in tool["blocked_domains"]
    assert first["output_config"]["format"]["type"] == "json_schema"
    assert "CEO of Acme" in first["messages"][0]["content"]
    # pause_turn: the paused assistant turn is sent back unchanged, with no extra user message
    assert second["messages"][1]["role"] == "assistant" and second["messages"][1]["content"][0].type == "server_tool_use"
    assert len(second["messages"]) == 2


def test_discover_private_individual_prompt_and_refusal():
    m = Manifest(subject=Subject(kind="private_individual", name="Sam Okoro"))
    provider, msgs = fake_provider([_resp([], stop="refusal")])
    with pytest.raises(ProviderError):
        discover(provider, m)
    assert "PROFESSIONAL material only" in msgs.calls[0]["system"]


# ----------------------------------------------------------------- pipeline

PAGE = "<html><head><title>{t}</title></head><body><main><p>{b}</p></main></body></html>"
OWN = ("Pricing is a promise about value. I would rather charge more and over-deliver than discount and resent "
       "every support ticket, because cheap customers churn and expensive ones stay when you earn it.")
NEWS = ('In an interview this week, Dana Reyes defended the price rise. "We charge for outcomes, not for seats," '
        'Reyes said. Analysts were unconvinced and called the move risky for a company of its size. '
        'A rival CEO, Mark Hale, said: "Nobody should pay for outcomes they cannot measure." The debate continues.')


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture
def web(tmp_path):
    site = tmp_path / "www"
    site.mkdir()
    (site / "robots.txt").write_text("User-agent: *\nAllow: /\n")
    (site / "pricing.html").write_text(PAGE.format(t="On pricing", b=OWN))
    (site / "news.html").write_text(PAGE.format(t="Acme raises prices", b=NEWS))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(site)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_port
    httpd.shutdown()
    httpd.server_close()


def _fake_post(url, opener=None):
    handle = url.split("/")[3]
    return Collected(url=url, title=f"@{handle} on X", date="2026-10-01", outlet=f"@{handle} on X", x_handle=handle,
                     text=f"Post by {handle}: outcomes beat seats when you price software.")


def test_research_classifies_and_stores(tmp_path, web, monkeypatch):
    own, news = f"http://127.0.0.1:{web}", f"http://localhost:{web}"
    root = tmp_path / "dana"
    root.mkdir()
    write_new_manifest(root, "Dana Reyes", "public_figure",
                       Research(links=[f"{own}/pricing.html", "https://x.com/danareyes"], hint="CEO of Acme"))
    m = load_manifest(root)
    monkeypatch.setattr(pipeline, "fetch_x_post", _fake_post)
    found = Discovery(candidates=[
        Candidate(f"{own}/pricing.html", "own_writing", from_user=True),
        Candidate("https://x.com/danareyes", "own_writing", from_user=True),
        Candidate(f"{news}/news.html", "own_writing", "Acme raises prices"),
        Candidate("https://x.com/danareyes/status/1", "x_post"),
        Candidate("https://x.com/someoneelse/status/2", "x_post"),
        Candidate("https://x.com/dana_alt/status/3", "x_post"),
        Candidate(f"{news}/missing.html", "reporting"),
    ], proposed_x_handles=["dana_alt"], identity_notes="notes")
    asked = []

    result = pipeline.research(root, m, provider=None, discover_fn=lambda *a, **k: found, allow_private=True, delay=0,
                               confirm_handles=lambda h, notes: asked.append((h, notes)) or ["dana_alt"])

    assert asked == [(["dana_alt"], "notes")]
    by_url = {i.url: i for i in result.added}
    assert by_url[f"{own}/pricing.html"].authorship == "by_me"            # host the user linked
    assert by_url[f"{news}/news.html"].authorship == "secondhand"         # anything else is reporting
    assert by_url[f"{news}/news.html"].outlet == "localhost" and "anyone authorship" in by_url[f"{news}/news.html"].why
    assert by_url["https://x.com/danareyes/status/1"].authorship == "by_me"   # confirmed via the user's x.com link
    assert by_url["https://x.com/dana_alt/status/3"].authorship == "by_me"    # confirmed by the user just now
    skipped = dict(result.skipped)
    assert "not a confirmed account" in skipped["https://x.com/someoneelse/status/2"]
    assert "X profile page" in skipped["https://x.com/danareyes"] and "HTTP 404" in skipped[f"{news}/missing.html"]
    assert load_manifest(root).research.x_handles == ["dana_alt"]           # confirmation persisted

    # Re-running finds nothing new: identical text is not stored twice.
    again = pipeline.research(root, load_manifest(root), provider=None, discover_fn=lambda *a, **k: found,
                              allow_private=True, delay=0)
    assert not again.added and len(again.unchanged) == 4

    # Answers: the outlet's own wording can't become a quote; Reyes's quoted remark can, labeled.
    def script(req):
        label = next(k for k, e in req.evidence.items() if e.authorship == "secondhand")
        return GenerationResult(model="fake", segments=[
            {"kind": "quote", "text": "We charge for outcomes, not for seats", "evidence_ids": [label]},
            {"kind": "quote", "text": "Analysts were unconvinced and called the move risky", "evidence_ids": [label]},
            {"kind": "quote", "text": "Nobody should pay for outcomes they cannot measure", "evidence_ids": [label]}])
    p = Persona(root, provider=FakeProvider(script))
    try:
        a = p.ask("What does Dana think about pricing by outcomes or seats?")
    finally:
        p.close()
    quote, journalist, rival = a.segments
    assert quote.kind == "quote" and quote.verified and quote.reported_by == "localhost"
    assert journalist.kind == "extrapolation" and "not as a quoted remark" in journalist.note
    # The rival's remark is a few words from "Reyes", but its attribution clause names Mark Hale.
    assert rival.kind == "extrapolation"
    assert 'authorship="secondhand" outlet="localhost"' in p.provider.requests[0].messages[-1]["content"]


def test_cli_new_and_deepen(tmp_path, web, monkeypatch, capsys):
    own = f"http://127.0.0.1:{web}"
    calls = []

    def fake_discover(provider, m, topic=None, max_searches=6):
        calls.append((m.subject.name, topic, max_searches))
        return Discovery(candidates=[Candidate(f"{own}/pricing.html", "own_writing", from_user=True)], searches=1)

    monkeypatch.setattr(pipeline, "discover", fake_discover)
    real_research = pipeline.research
    monkeypatch.setattr("anyone_ai.research.pipeline.research",
                        functools.partial(real_research, allow_private=True, delay=0))
    out = tmp_path / "dana"
    assert main(["new", "Dana Reyes", "--link", f"{own}/pricing.html", "--hint", "CEO of Acme",
                 "--out", str(out), "--max-searches", "2", "--yes"]) == 0
    written = yaml.safe_load((out / "persona.yaml").read_text())
    assert written["subject"] == {"kind": "public_figure", "name": "Dana Reyes", "bio": ""}
    assert written["research"]["hint"] == "CEO of Acme"
    printed = capsys.readouterr().out
    assert "Their own words" in printed and "On pricing" in printed
    with Store(index_path(out)) as s:
        assert [r.authorship for r in s.list_sources()] == ["by_me"]

    assert main(["-C", str(out), "deepen", "pricing", "--yes"]) == 0
    assert calls == [("Dana Reyes", None, 2), ("Dana Reyes", "pricing", 4)]
    assert main(["new", "Dana Reyes", "--out", str(out), "--yes"]) == 2          # never overwrites
    with pytest.raises(SystemExit):                                                # self mode uses init/add
        main(["new", "Me", "--kind", "self", "--yes"])
    assert main(["new", "X", "--x-handle", "bad handle!", "--out", str(tmp_path / "x"), "--yes"]) == 2
    assert not (tmp_path / "x").exists()


def test_archive_on_linked_site_is_followed_and_x_posts_come_from_real_results(tmp_path, web, monkeypatch):
    own, other = f"http://127.0.0.1:{web}", f"http://localhost:{web}"
    site = tmp_path / "www"
    links = "".join(f'<li><a href="/essay{i}.html">Essay number {i}</a></li>' for i in range(16))
    (site / "articles.html").write_text(f"<html><head><title>Essays</title></head><body><ul>{links}</ul></body></html>")
    for i in range(16):
        (site / f"essay{i}.html").write_text(PAGE.format(t=f"Essay {i}", b=f"Essay {i}. " + OWN))
    root = tmp_path / "dana"
    root.mkdir()
    write_new_manifest(root, "Dana Reyes", "public_figure", Research(links=[f"{own}/pricing.html"], x_handles=["danareyes"]))
    monkeypatch.setattr(pipeline, "fetch_x_post", _fake_post)
    found = Discovery(candidates=[Candidate(f"{own}/articles.html", "own_writing"),
                                  Candidate(f"{other}/articles.html", "reporting")],
                      seen_urls={"https://x.com/danareyes/status/9", "https://x.com/stranger/status/8"})

    result = pipeline.research(root, load_manifest(root), provider=None, discover_fn=lambda *a, **k: found,
                               allow_private=True, max_site_pages=10, delay=0)

    essays = [i for i in result.added if "/essay" in i.url]
    assert len(essays) == 10 and all(i.authorship == "by_me" for i in essays)        # budget respected
    assert not any(i.url.endswith("articles.html") for i in result.added)            # the index itself isn't a source
    assert any("page budget reached" in why for _, why in result.skipped)
    assert any("index page on a site you didn't link" in why for _, why in result.skipped)
    added = {i.url for i in result.added}
    assert "https://x.com/danareyes/status/9" in added and "https://x.com/stranger/status/8" not in added
