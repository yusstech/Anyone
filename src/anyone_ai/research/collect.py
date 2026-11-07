"""Collection: Anyone fetches source text itself. A model's summary is never used as quotable text.

- X posts: X's public oEmbed endpoint returns the exact post text, author and date, without auth.
- Web pages: one robots.txt-respecting, size-capped, same-host fetch per page (the site importer's Fetcher).
"""
from __future__ import annotations

import datetime as _dt
import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser

from ..importers.site import (ASSET, MAX_BYTES, SKIP_PATH, TIMEOUT, USER_AGENT, Fetcher, ImportError_, canonical,
                              html_to_text, page_links)

OEMBED = "https://publish.x.com/oembed"
X_STATUS = re.compile(r"^https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/([A-Za-z0-9_]{1,15})/status(?:es)?/(\d{1,25})(?:[/?#]|$)", re.I)
X_PROFILE = re.compile(r"^https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/([A-Za-z0-9_]{1,15})/?(?:[?#].*)?$", re.I)
_X_RESERVED = {"home", "search", "explore", "i", "intent", "share", "settings", "notifications", "messages", "hashtag"}

# People-search and data-broker sites: never fetched, never offered to the search model.
BLOCKED_HOSTS = (
    "whitepages.com", "spokeo.com", "beenverified.com", "truepeoplesearch.com", "fastpeoplesearch.com",
    "peoplefinders.com", "intelius.com", "radaris.com", "mylife.com", "instantcheckmate.com", "truthfinder.com",
    "pipl.com", "thatsthem.com", "zabasearch.com", "familytreenow.com", "clustrmaps.com", "nuwber.com",
)


class CollectError(RuntimeError):
    """A source could not be collected; the message is safe to show."""


@dataclass
class Collected:
    url: str          # canonical URL, used as the source key
    title: str
    date: str | None
    text: str
    outlet: str       # site host, or "@handle on X"
    x_handle: str | None = None  # author of an X post (from oEmbed, not from the URL)
    index_links: list[str] = field(default_factory=list)  # set when the page is a list of links (an archive)


def host_of(url: str) -> str:
    h = urllib.parse.urlparse(url).hostname or ""
    h = h.lower()
    return h[4:] if h.startswith("www.") else h


def is_blocked_host(url: str) -> bool:
    h = host_of(url)
    return any(h == b or h.endswith("." + b) for b in BLOCKED_HOSTS)


def check_public_url(url: str, allow_private: bool = False) -> None:
    """Refuse non-web schemes and obviously internal hosts (SSRF guard for model- or user-supplied URLs).

    DNS names are not resolved here, so a public name pointing at a private address is not caught;
    a hosted deployment must add egress filtering."""
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise CollectError(f"not an http(s) URL: {url}")
    if p.username or p.password:
        raise CollectError("URLs with credentials are not fetched")
    if is_blocked_host(url):
        raise CollectError(f"{host_of(url)} is a people-search site; Anyone never collects from those")
    if allow_private:
        return
    host = p.hostname.lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan")):
        raise CollectError(f"refusing internal host {host}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return
    if not ip.is_global:
        raise CollectError(f"refusing non-public address {host}")


# ------------------------------------------------------------------- X posts

class _PostHTML(HTMLParser):
    """Text of the first <p> in X's oEmbed blockquote (the post body), and the date link text after it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.links: list[str] = []
        self._in_p = False
        self._done_p = False
        self._in_a = False
        self._a_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "p" and not self._done_p:
            self._in_p = True
        elif tag == "br" and self._in_p:
            self.text.append("\n")
        elif tag == "a":
            self._in_a, self._a_text = True, []

    def handle_endtag(self, tag):
        if tag == "p" and self._in_p:
            self._in_p, self._done_p = False, True
        elif tag == "a" and self._in_a:
            self._in_a = False
            if self._done_p:
                self.links.append("".join(self._a_text).strip())

    def handle_data(self, data):
        if self._in_p:
            self.text.append(data)
        if self._in_a:
            self._a_text.append(data)


_MEDIA = re.compile(r"\s*\bpic\.(?:twitter|x)\.com/\w+")


def parse_oembed(data: dict) -> tuple[str, str | None, str | None, str]:
    """Return (text, handle, date, author_name) from an oEmbed response."""
    if not isinstance(data, dict) or not isinstance(data.get("html"), str):
        raise CollectError("unexpected oEmbed response")
    parser = _PostHTML()
    parser.feed(data["html"])
    parser.close()
    text = _MEDIA.sub("", "".join(parser.text)).strip()
    m = X_PROFILE.match(str(data.get("author_url") or ""))
    handle = m.group(1) if m else None
    date = None
    for link in reversed(parser.links):
        try:
            date = _dt.datetime.strptime(link, "%B %d, %Y").date().isoformat()
            break
        except ValueError:
            continue
    return text, handle, date, str(data.get("author_name") or "")


def fetch_x_post(url: str, opener=None) -> Collected:
    m = X_STATUS.match(url)
    if not m:
        raise CollectError(f"not an X post URL: {url}")
    post_url = f"https://x.com/{m.group(1)}/status/{m.group(2)}"
    q = urllib.parse.urlencode({"url": post_url, "omit_script": "1", "dnt": "true"})
    req = urllib.request.Request(f"{OEMBED}?{q}", headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with (opener or urllib.request.build_opener()).open(req, timeout=TIMEOUT) as res:
            body = res.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        # 404: deleted, protected or never existed. Such posts must not be kept.
        raise CollectError(f"X post unavailable (HTTP {exc.code})") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CollectError(f"could not reach X oEmbed ({getattr(exc, 'reason', exc)})") from exc
    if len(body) > MAX_BYTES:
        raise CollectError("oEmbed response too large")
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectError("invalid oEmbed response") from exc
    text, handle, date, author = parse_oembed(data)
    if not text:
        raise CollectError("X post has no text")
    if not handle:
        raise CollectError("oEmbed did not name the author")
    return Collected(url=f"https://x.com/{handle}/status/{m.group(2)}", title=f"{author or handle} (@{handle}) on X",
                     date=date, text=text, outlet=f"@{handle} on X", x_handle=handle)


# --------------------------------------------------------------------- pages

class PageCollector:
    """Fetches single pages, one robots-aware Fetcher per host (so crawl-delay is honoured across pages)."""

    def __init__(self, delay: float = 1.0, opener=None, allow_private: bool = False):
        self.delay, self.opener, self.allow_private = delay, opener, allow_private
        self._fetchers: dict[str, Fetcher] = {}

    def fetch(self, url: str) -> Collected:
        check_public_url(url, self.allow_private)
        p = urllib.parse.urlparse(url)
        site = f"{p.scheme}://{p.netloc}"
        f = self._fetchers.get(site)
        if f is None:
            f = self._fetchers[site] = Fetcher(site, delay=self.delay, opener=self.opener)
            f.load_robots()
        try:
            body, ctype = f.get_text(canonical(url))
        except ImportError_ as exc:
            raise CollectError(str(exc)) from exc
        if ctype and "html" not in ctype.lower() and "text/plain" not in ctype.lower():
            raise CollectError(f"unsupported content type {ctype.split(';')[0]}")
        links: list[str] = []
        if "html" in (ctype or "html").lower():
            text, title, date = html_to_text(body)
            links = index_links(body, url, len(text.split()))
        else:
            text, title, date = body, "", None
        if links:
            return Collected(url=canonical(url), title=title or url, date=date, text=text, outlet=host_of(url),
                             index_links=links)
        if len(text.split()) < 20:
            raise CollectError("page has too little text (paywall, login wall or script-only page?)")
        return Collected(url=canonical(url), title=title or url, date=date, text=text, outlet=host_of(url))


def index_links(html: str, url: str, words: int) -> list[str]:
    """Same-site article links if the page looks like an archive or index (mostly links, little prose)."""
    host = host_of(url)
    seen: dict[str, None] = {}
    for link in page_links(html, url):
        c = canonical(link)
        path = urllib.parse.urlparse(c).path
        if host_of(c) == host and c != canonical(url) and path not in ("", "/") \
                and not ASSET.search(path) and not SKIP_PATH.search(path):
            seen[c] = None
    links = list(seen)
    return links if len(links) >= 15 and words / len(links) < 12 else []


def fetch_any(url: str, pages: PageCollector, opener=None) -> Collected:
    if X_STATUS.match(url):
        return fetch_x_post(url, opener=opener)
    return pages.fetch(url)
