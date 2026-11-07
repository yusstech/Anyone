"""name + links → discover → collect → classify → store. The dossier review happens in the CLI.

Classification decides what may be quoted as the subject's own words (authorship ``by_me``), and it
never relies on the model's say-so alone:

- X post: ``by_me`` only if oEmbed names a *confirmed* handle as the author (the user's x.com link,
  ``--x-handle``, or a model-proposed handle the user confirmed). Posts by anyone else are skipped.
- Page on a host the user linked: ``by_me`` (the user vouched for that site). A linked site root is
  imported through its feed/sitemap.
- Any other page: ``secondhand``, labeled with its outlet. Quotes from it verify only as remarks the
  outlet itself quotes near the subject's name. The user can upgrade it with ``anyone authorship``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from ..importers.site import ImportError_, import_site
from ..manifest import Manifest, Research, index_path, save_research
from ..store import IngestResult, Store
from .collect import (X_PROFILE, X_STATUS, Collected, CollectError, PageCollector, check_public_url,
                      fetch_x_post, host_of)
from .discover import Candidate, Discovery, discover


@dataclass
class DossierItem:
    source_id: str
    url: str
    title: str
    date: str | None
    authorship: str
    outlet: str
    why: str


@dataclass
class ResearchResult:
    discovery: Discovery
    added: list[DossierItem] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (url, reason)
    confirmed_handles: list[str] = field(default_factory=list)


def link_handles(links: list[str]) -> list[str]:
    out = []
    for link in links:
        m = X_PROFILE.match(link)
        if m and m.group(1).lower() not in ("home", "search", "explore", "i", "intent", "share"):
            out.append(m.group(1))
    return out


def _is_x(url: str) -> bool:
    return host_of(url) in ("x.com", "twitter.com", "mobile.twitter.com")


def research(root: Path, manifest: Manifest, provider, *, topic: str | None = None, max_searches: int = 6,
             max_site_pages: int = 50, confirm_handles: Callable[[list[str], str], list[str]] = lambda h, notes: [],
             progress: Callable[[str], None] = lambda msg: None, opener=None, allow_private: bool = False,
             delay: float = 1.0,
             discover_fn=None) -> ResearchResult:
    r = manifest.research
    progress(f"Searching the web for {manifest.subject.name} (up to {max_searches} searches)...")
    found = (discover_fn or discover)(provider, manifest, topic=topic, max_searches=max_searches)
    result = ResearchResult(discovery=found)

    confirmed = {h.lower(): h for h in [*r.x_handles, *link_handles(r.links)]}
    proposals = [h for h in found.proposed_x_handles if h.lower() not in confirmed]
    newly = [h for h in confirm_handles(proposals, found.identity_notes) if h in proposals] if proposals else []
    if newly:
        r = Research(links=r.links, hint=r.hint, x_handles=[*r.x_handles, *newly])
        save_research(root, r)
        manifest.research = r
        confirmed.update({h.lower(): h for h in newly})
    result.confirmed_handles = list(confirmed.values())

    # X posts by confirmed accounts that real search results contained, even if the model didn't list them.
    listed = {c.url for c in found.candidates}
    for url in sorted(found.seen_urls):
        m = X_STATUS.match(url)
        if m and m.group(1).lower() in confirmed and url not in listed:
            found.candidates.append(Candidate(url, "x_post", reason="in search results, by a confirmed account"))

    vouched = {host_of(u) for u in r.links if not _is_x(u)}
    budget = [max_site_pages]  # pages followed from archives/feeds across this run
    pages = PageCollector(delay=delay, opener=opener, allow_private=allow_private)
    redact = dict(redact_names=manifest.privacy.redact_names, redact_pii=manifest.privacy.redact_pii)

    with Store(index_path(root)) as store:
        def keep(c: Collected, authorship: str, proof: str, why: str) -> None:
            res = IngestResult([], [], [])
            try:
                row = store.ingest_text(c.url, c.title, c.date, c.text, result=res, authorship=authorship,
                                        origin="research", ownership_proof=proof, url=c.url, outlet=c.outlet, **redact)
            except ValueError as exc:
                result.skipped.append((c.url, str(exc)))
                return
            if row is None:
                result.unchanged.append(c.url)
            else:
                result.added.append(DossierItem(row.id, c.url, c.title, c.date, authorship, c.outlet, why))

        for cand in found.candidates:
            url = cand.url
            if X_PROFILE.match(url):
                result.skipped.append((url, "X profile page (posts are collected individually)"))
                continue
            try:
                if X_STATUS.match(url):
                    progress(f"  X post {url}")
                    post = fetch_x_post(url, opener=opener)
                    if post.x_handle.lower() not in confirmed:
                        result.skipped.append((url, f"posted by @{post.x_handle}, not a confirmed account of "
                                                    f"{manifest.subject.name}"))
                        continue
                    keep(post, "by_me", "x_oembed_author", f"posted by confirmed account @{post.x_handle}")
                elif cand.from_user and urlparse(url).path in ("", "/"):
                    progress(f"  site {url} (feed/sitemap, robots.txt respected)")
                    check_public_url(url, allow_private)
                    site = import_site(url, max_pages=max(budget[0], 1), delay=delay, opener=opener)
                    budget[0] -= len(site.pages)
                    for pg in site.pages:
                        keep(Collected(pg.url, pg.title, pg.date, pg.text, host_of(pg.url)), "by_me",
                             "user_link_site", "on a site you linked")
                    result.skipped.extend(site.skipped)
                else:
                    progress(f"  page {url}")
                    page = pages.fetch(url)
                    if page.index_links:
                        if host_of(url) not in vouched:
                            result.skipped.append((url, "index page on a site you didn't link"))
                            continue
                        progress(f"  archive {url}: following up to {min(budget[0], len(page.index_links))} "
                                 f"of {len(page.index_links)} links")
                        for link in page.index_links:
                            if budget[0] <= 0:
                                result.skipped.append((url, f"page budget reached ({max_site_pages}); "
                                                            "raise it or run `anyone deepen`"))
                                break
                            budget[0] -= 1
                            try:
                                sub = pages.fetch(link)
                            except CollectError as exc:
                                result.skipped.append((link, str(exc)))
                                continue
                            if not sub.index_links:
                                keep(sub, "by_me", "user_link_domain", f"linked from {url}, on a site you linked")
                        continue
                    if host_of(url) in vouched:
                        keep(page, "by_me", "user_link_domain", "on a site you linked")
                    else:
                        hint = " (search model says these are their own words: if so, `anyone authorship <id> by_me`)" \
                            if cand.type == "own_writing" else ""
                        keep(page, "secondhand", "reported", f"reporting by {page.outlet}{hint}")
            except (CollectError, ImportError_) as exc:
                result.skipped.append((url, str(exc)))
    return result
