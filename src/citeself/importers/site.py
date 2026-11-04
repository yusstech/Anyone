"""Import your own website or blog from a URL.

Discovery, in order of fidelity:
  1. RSS/Atom feeds (linked from the homepage, or common paths): full text when the feed carries it
  2. WordPress REST API (/wp-json/wp/v2/posts): full HTML, dates and links, no auth needed
  3. sitemap.xml (including sitemap indexes and robots.txt Sitemap: lines)
  4. Same-site links on the homepage, one level deep

Rules: robots.txt is honoured per RFC 9309 (via protego), requests stay on the
site's host, are rate-limited, size-capped and identify themselves. Pages that
look like site chrome (privacy, login, tag archives...) or are too short to
carry a view are skipped. Nothing is written outside the persona's index.
"""
from __future__ import annotations

import email.utils
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html import unescape
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

from protego import Protego

from .. import __version__
from ..text import _HTMLText

USER_AGENT = f"CiteSelf/{__version__} (+https://github.com/yusstech/PersonaLab; owner-initiated import)"
MAX_BYTES = 5 * 1024 * 1024
TIMEOUT = 20
MIN_WORDS = 60
FEED_PATHS = ("/feed", "/rss", "/feed.xml", "/rss.xml", "/atom.xml", "/index.xml", "/rss/", "/blog/feed")
SKIP_PATH = re.compile(
    r"/(privacy|privacy-policy|terms|cookies?|legal|contact|login|signin|sign-in|signup|sign-up|register|"
    r"subscribe|unsubscribe|account|cart|checkout|search|tag|tags|category|categories|author|page/\d+|"
    r"wp-admin|wp-login|feed|rss|amp)(\.html?|\.php)?(/|$)", re.I)
ASSET = re.compile(r"\.(png|jpe?g|gif|svg|webp|ico|css|js|mjs|json|xml|txt|pdf|zip|mp3|mp4|woff2?)$", re.I)
_CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}encoded"
_ATOM = "{http://www.w3.org/2005/Atom}"
_SM = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


class ImportError_(RuntimeError):
    """Import failed in a way the user should see."""


@dataclass
class Page:
    url: str
    title: str
    date: str | None
    text: str
    via: str  # feed | wordpress | sitemap | links


@dataclass
class SiteImport:
    site: str
    pages: list[Page] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (url, reason)
    discovered_via: list[str] = field(default_factory=list)


def canonical(url: str) -> str:
    url, _ = urldefrag(url.strip())
    p = urlparse(url)
    path = re.sub(r"/+$", "", p.path) or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", p.query, ""))


def _host(url: str) -> str:
    h = urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


class Fetcher:
    def __init__(self, site: str, delay: float = 1.0, opener=None):
        self.site = site
        self.host = _host(site)
        self.delay = delay
        self._last = 0.0
        self.opener = opener or urllib.request.build_opener()
        self.robots: Protego | None = None

    def load_robots(self) -> None:
        try:
            body, _ = self._get(urljoin(self.site, "/robots.txt"), check_robots=False)
            self.robots = Protego.parse(body.decode("utf-8", "replace"))
            cd = self.robots.crawl_delay(USER_AGENT)
            if cd:
                self.delay = max(self.delay, float(cd))
        except ImportError_:
            self.robots = Protego.parse("")  # no robots.txt: everything allowed

    def allowed(self, url: str) -> bool:
        return self.robots is None or self.robots.can_fetch(url, USER_AGENT)

    def same_site(self, url: str) -> bool:
        p = urlparse(url)
        return p.scheme in ("http", "https") and _host(url) == self.host

    def _get(self, url: str, check_robots: bool = True) -> tuple[bytes, str]:
        if not self.same_site(url):
            raise ImportError_(f"refusing to fetch off-site URL {url}")
        if check_robots and not self.allowed(url):
            raise ImportError_("disallowed by robots.txt")
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xml,*/*;q=0.5"})
        try:
            with self.opener.open(req, timeout=TIMEOUT) as res:
                self._last = time.monotonic()
                if not self.same_site(res.geturl()):
                    raise ImportError_(f"redirected off-site to {res.geturl()}")
                body = res.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    raise ImportError_("response too large")
                return body, res.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            self._last = time.monotonic()
            raise ImportError_(f"HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._last = time.monotonic()
            raise ImportError_(f"could not connect ({getattr(exc, 'reason', exc)})") from exc

    def get_text(self, url: str) -> tuple[str, str]:
        body, ctype = self._get(url)
        m = re.search(r"charset=([\w-]+)", ctype)
        return body.decode(m.group(1) if m else "utf-8", "replace"), ctype


# ------------------------------------------------------------------ parsing

def _xml(text: str) -> ET.Element | None:
    if "<!DOCTYPE" in text[:2048].upper() or "<!ENTITY" in text[:4096].upper():
        return None  # never needed for feeds/sitemaps; refuse entity expansion tricks
    try:
        return ET.fromstring(text.encode("utf-8"))
    except ET.ParseError:
        return None


def _date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", value):
        return value[:10]
    try:
        return email.utils.parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError, IndexError):
        return None


def html_to_text(html: str) -> tuple[str, str, str | None]:
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    return parser.text, " ".join(parser.title.split()), parser.date


def parse_feed(text: str, base: str) -> list[dict]:
    root = _xml(text)
    if root is None:
        return []
    items = []
    for it in root.iter("item"):  # RSS 2.0
        content = it.findtext(_CONTENT_NS) or ""
        items.append({"url": urljoin(base, (it.findtext("link") or "").strip()), "title": it.findtext("title") or "",
                      "date": _date(it.findtext("pubDate")), "html": content or it.findtext("description") or "",
                      "full": bool(content)})
    for e in root.iter(f"{_ATOM}entry"):  # Atom
        link = next((l.get("href") for l in e.findall(f"{_ATOM}link") if l.get("rel") in (None, "alternate")), "")
        content = e.findtext(f"{_ATOM}content") or ""
        items.append({"url": urljoin(base, link or ""), "title": e.findtext(f"{_ATOM}title") or "",
                      "date": _date(e.findtext(f"{_ATOM}published") or e.findtext(f"{_ATOM}updated")),
                      "html": content or e.findtext(f"{_ATOM}summary") or "", "full": bool(content)})
    return [i for i in items if i["url"]]


def feed_links(html: str, base: str) -> list[str]:
    out = []
    for tag in re.findall(r"<link\b[^>]*>", html, re.I):
        if re.search(r'type=["\']application/(rss|atom)\+xml', tag, re.I):
            m = re.search(r'href=["\']([^"\']+)', tag, re.I)
            if m:
                out.append(urljoin(base, unescape(m.group(1))))
    return out


def page_links(html: str, base: str) -> list[str]:
    return [urljoin(base, unescape(h)) for h in re.findall(r'<a\b[^>]*href=["\']([^"\'#]+)', html, re.I)]


def parse_sitemap(text: str) -> tuple[list[str], list[str]]:
    """Returns (page urls, nested sitemap urls)."""
    root = _xml(text)
    if root is None:
        return [], []
    pages = [(l.text or "").strip() for l in root.iter(f"{_SM}loc")] if root.tag == f"{_SM}urlset" else []
    nested = [(l.text or "").strip() for l in root.iter(f"{_SM}loc")] if root.tag == f"{_SM}sitemapindex" else []
    return [p for p in pages if p], [n for n in nested if n]


# ------------------------------------------------------------------- import

def import_site(url: str, max_pages: int = 200, delay: float = 1.0, exclude: list[str] | None = None,
                opener=None, progress=None) -> SiteImport:
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    p = urlparse(url)
    if not p.netloc:
        raise ImportError_(f"Not a valid URL: {url}")
    site = f"{p.scheme}://{p.netloc}"
    f = Fetcher(site, delay=delay, opener=opener)
    f.load_robots()
    result = SiteImport(site=site)
    excl = [re.compile(x) for x in (exclude or [])]

    try:
        home_html, _ = f.get_text(canonical(url))
    except ImportError_ as exc:
        raise ImportError_(f"Could not open {url}: {exc}") from exc

    candidates: dict[str, Page | None] = {}  # url -> prefilled page (from full-text feeds/WP) or None

    def consider(u: str, page: Page | None = None) -> None:
        u = canonical(u)
        if u in candidates and candidates[u] is not None:
            return
        if not f.same_site(u) or ASSET.search(urlparse(u).path):
            return
        candidates[u] = page if page is not None else candidates.get(u)

    # 1. feeds
    feeds = feed_links(home_html, site) or [urljoin(site, fp) for fp in FEED_PATHS]
    for feed_url in dict.fromkeys(feeds):
        try:
            text, ctype = f.get_text(feed_url)
        except ImportError_:
            continue
        items = parse_feed(text, site)
        if not items:
            continue
        result.discovered_via.append(f"feed {feed_url}")
        for it in items:
            page = None
            if it["full"]:
                body, _, _ = html_to_text(it["html"])
                page = Page(canonical(it["url"]), " ".join(unescape(it["title"]).split()), it["date"], body, "feed")
            consider(it["url"], page)
        break  # one working feed is enough; others usually duplicate it

    # 2. WordPress REST API (public, full content)
    for page_no in range(1, 21):
        try:
            text, ctype = f.get_text(urljoin(site, f"/wp-json/wp/v2/posts?per_page=100&page={page_no}"))
            posts = json.loads(text)
        except (ImportError_, ValueError):
            break
        if not isinstance(posts, list) or not posts:
            break
        if page_no == 1:
            result.discovered_via.append("wordpress api")
        for post in posts:
            try:
                body, _, _ = html_to_text(post["content"]["rendered"])
                title = " ".join(unescape(re.sub(r"<[^>]+>", "", post["title"]["rendered"])).split())
                consider(post["link"], Page(canonical(post["link"]), title, _date(post.get("date")), body, "wordpress"))
            except (KeyError, TypeError):
                continue

    # 3. sitemaps
    # protego returns a generator; materialize it, or an empty one would count as truthy
    sitemaps = (list(f.robots.sitemaps) if f.robots else []) or [urljoin(site, "/sitemap.xml")]
    seen_maps: set[str] = set()
    queue = list(sitemaps)
    while queue and len(seen_maps) < 20:
        sm = queue.pop(0)
        if sm in seen_maps:
            continue
        seen_maps.add(sm)
        try:
            text, _ = f.get_text(sm)
        except ImportError_:
            continue
        pages, nested = parse_sitemap(text)
        if pages or nested:
            result.discovered_via.append(f"sitemap {sm}")
        queue.extend(nested)
        for u in pages:
            consider(u)

    # 4. homepage links (one level)
    if not result.discovered_via:
        result.discovered_via.append("homepage links")
    consider(url)
    for u in page_links(home_html, site):
        consider(u)

    # filter, then fetch what isn't prefilled
    todo = []
    for u, page in candidates.items():
        path = urlparse(u).path
        if SKIP_PATH.search(path):
            result.skipped.append((u, "not content (privacy/login/tag/... page)"))
        elif any(x.search(u) for x in excl):
            result.skipped.append((u, "excluded by --exclude"))
        elif not f.allowed(u):
            result.skipped.append((u, "disallowed by robots.txt"))
        else:
            todo.append((u, page))
    if len(todo) > max_pages:
        for u, _ in todo[max_pages:]:
            result.skipped.append((u, f"over --max-pages {max_pages}"))
        todo = todo[:max_pages]

    for n, (u, page) in enumerate(todo, start=1):
        if progress:
            progress(n, len(todo), u)
        if page is None:
            try:
                html, ctype = f.get_text(u)
            except ImportError_ as exc:
                result.skipped.append((u, str(exc)))
                continue
            if "html" not in ctype.lower():
                result.skipped.append((u, f"not HTML ({ctype or 'unknown type'})"))
                continue
            text, title, date = html_to_text(html)
            page = Page(u, title or u, date, text, "page")
        if len(page.text.split()) < MIN_WORDS:
            result.skipped.append((u, f"too short (< {MIN_WORDS} words)"))
            continue
        result.pages.append(page)
    return result
