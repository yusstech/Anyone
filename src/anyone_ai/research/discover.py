"""Discovery: the user's model searches the web and proposes candidate source URLs.

The model only *finds* URLs. A candidate is kept only if its URL appeared in a real search
result (or was supplied by the user), so an invented URL can never enter the dossier, and
its text is fetched by Anyone itself (collect.py). The model's labels are hints that the
pipeline checks; they never decide on their own whether text counts as the subject's words.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..importers.site import canonical
from ..manifest import Manifest
from ..providers import AnthropicProvider, MalformedOutput, ProviderError
from .collect import BLOCKED_HOSTS, X_STATUS, is_blocked_host

WEB_SEARCH_TOOL = "web_search_20260209"
MAX_CONTINUATIONS = 4
CANDIDATE_TYPES = ("own_writing", "x_post", "reporting")

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "type": {"type": "string", "enum": list(CANDIDATE_TYPES)},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["url", "type", "title", "reason"],
                "additionalProperties": False,
            },
        },
        "x_handles": {"type": "array", "items": {"type": "string"}},
        "identity_notes": {"type": "string"},
    },
    "required": ["candidates", "x_handles", "identity_notes"],
    "additionalProperties": False,
}


@dataclass
class Candidate:
    url: str
    type: str
    title: str = ""
    reason: str = ""
    from_user: bool = False


@dataclass
class Discovery:
    candidates: list[Candidate] = field(default_factory=list)
    proposed_x_handles: list[str] = field(default_factory=list)
    identity_notes: str = ""
    searches: int = 0
    dropped_unseen: list[str] = field(default_factory=list)  # URLs the model gave that no search returned
    seen_urls: set[str] = field(default_factory=set)          # every URL real search results contained
    usage: dict = field(default_factory=dict)


def _system(m: Manifest) -> str:
    n, kind = m.subject.name, m.subject.kind
    lines = [
        f"You are the research step of Anyone, which builds a cited AI simulation of {n} from public sources. "
        "Search the web and return candidate URLs whose text shows how this person thinks, in their own words "
        "wherever possible.",
        "",
        "Run several distinct searches: their own site or newsletter, interviews and profiles, talks or podcast "
        "transcripts, and X posts. Every URL a search returns counts as a search result, including results your "
        "search code printed. Return only such URLs, copied exactly; never construct or guess a URL.",
        "Types:",
        f'- "own_writing": text {n} wrote or said themselves: their blog, newsletter, op-eds, essays, talk or '
        "podcast transcripts, their posts on their own site.",
        f'- "x_post": an individual post by {n} on X, as https://x.com/<handle>/status/<id>.',
        f'- "reporting": interviews, profiles and news coverage about {n} that quote them directly.',
        "For the person's own site, include their archive or essay index page if there is one: Anyone follows "
        "its links itself. Prefer primary sources and pages with substantial text. Skip videos without transcripts, paywalled "
        "pages, social media profiles, aggregator and SEO spam pages.",
        f"Make sure every source is about this specific {n} and not someone with the same name; use the "
        "hint and links below to tell them apart, and explain how you did so in identity_notes.",
        f"x_handles: X accounts you have good evidence belong to {n} (without @). The user confirms them.",
    ]
    if kind == "private_individual":
        lines += [
            "",
            f"{n} is a private individual who has not consented. Collect PROFESSIONAL material only: their "
            "published articles, talks, company or conference pages, professional posts. Never return pages "
            "about their family, health, religion, sexuality, home or location, personal social media, court or "
            "property records, or people-search sites.",
        ]
    lines += ["", "Return at most 25 candidates, best first."]
    return "\n".join(lines)


def _user(m: Manifest, topic: str | None) -> str:
    r = m.research
    parts = [f"Person: {m.subject.name}"]
    if r.hint:
        parts.append(f"Hint (who they are): {r.hint}")
    if m.subject.bio:
        parts.append(f"Bio: {m.subject.bio}")
    if r.links:
        parts.append("Reference links from the user (trusted to be about this person):\n" + "\n".join(f"- {u}" for u in r.links))
    if r.x_handles:
        parts.append("Confirmed X accounts: " + ", ".join("@" + h for h in r.x_handles))
    if topic:
        parts.append(f"Focus this search on what they have said about: {topic}")
    return "\n\n".join(parts)


def _search_urls(content) -> tuple[set[str], int]:
    """URLs that actually appeared in web_search results, and the number of searches run."""
    seen, searches = set(), 0
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "server_tool_use" and getattr(block, "name", None) == "web_search":
            searches += 1
        elif btype == "web_search_tool_result":
            results = getattr(block, "content", None)
            if isinstance(results, list):  # an error result is a single object, not a list
                for r in results:
                    url = getattr(r, "url", None)
                    if isinstance(url, str):
                        seen.add(canonical(url))
        elif btype == "text":
            for c in getattr(block, "citations", None) or []:
                url = getattr(c, "url", None)
                if isinstance(url, str):
                    seen.add(canonical(url))
    return seen, searches


def _final_json(content) -> dict:
    texts = [b.text for b in content if getattr(b, "type", None) == "text" and getattr(b, "text", None)]
    for text in reversed(texts):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise MalformedOutput("The search model did not return its candidate list.")


def run_search(provider: AnthropicProvider, m: Manifest, topic: str | None, max_searches: int):
    """One research turn with the server-side web search tool, resuming paused turns."""
    tool: dict = {"type": WEB_SEARCH_TOOL, "name": "web_search", "max_uses": max_searches,
                  "blocked_domains": list(BLOCKED_HOSTS)}
    messages: list[dict] = [{"role": "user", "content": _user(m, topic)}]
    content: list = []
    usage: dict = {}
    for _ in range(MAX_CONTINUATIONS + 1):
        response = provider._create(
            model=provider.settings.name, max_tokens=provider.max_tokens, system=_system(m), messages=messages,
            tools=[tool], output_config={"effort": provider.settings.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        )
        content.extend(response.content)
        for k, v in (getattr(response, "usage", None) and response.usage.model_dump() or {}).items():
            if isinstance(v, int):
                usage[k] = usage.get(k, 0) + v
        if response.stop_reason == "refusal":
            raise ProviderError("The model declined to research this person.")
        if response.stop_reason == "max_tokens":
            raise MalformedOutput("The search model ran out of output tokens.")
        if response.stop_reason != "pause_turn":
            return content, response.content, usage
        messages = [messages[0], {"role": "assistant", "content": response.content}]
    raise ProviderError("Research did not finish after several continuations; try fewer searches or a narrower topic.")


def discover(provider: AnthropicProvider, m: Manifest, topic: str | None = None, max_searches: int = 6,
             max_candidates: int = 25) -> Discovery:
    all_content, final_content, usage = run_search(provider, m, topic, max_searches)
    seen, searches = _search_urls(all_content)
    data = _final_json(final_content)

    out = Discovery(searches=searches, usage=usage, identity_notes=str(data.get("identity_notes") or "")[:2000],
                    seen_urls=seen)
    keep: dict[str, Candidate] = {}
    for link in m.research.links:
        keep[canonical(link)] = Candidate(canonical(link), "x_post" if X_STATUS.match(link) else "own_writing",
                                          reason="link supplied by you", from_user=True)
    for c in data.get("candidates") or []:
        if not isinstance(c, dict) or not isinstance(c.get("url"), str):
            continue
        url = canonical(c["url"])
        if not url.lower().startswith(("http://", "https://")) or is_blocked_host(url) or url in keep:
            continue
        if url not in seen:
            out.dropped_unseen.append(url)
            continue
        ctype = c.get("type") if c.get("type") in CANDIDATE_TYPES else "reporting"
        if ctype == "x_post" and not X_STATUS.match(url):
            ctype = "reporting"
        if ctype != "x_post" and X_STATUS.match(url):
            ctype = "x_post"
        keep[url] = Candidate(url, ctype, str(c.get("title") or "")[:300], str(c.get("reason") or "")[:500])
        if len(keep) >= max_candidates + len(m.research.links):
            break
    out.candidates = list(keep.values())
    handles = []
    for h in data.get("x_handles") or []:
        h = str(h).strip().lstrip("@")
        if h and len(h) <= 15 and h.replace("_", "").isalnum() and h.lower() not in {x.lower() for x in handles}:
            handles.append(h)
    out.proposed_x_handles = handles
    return out
