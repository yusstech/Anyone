"""Citation and quote validator — the trust core of Anyone.

The model *proposes* labeled segments; this module decides what the user sees.
Rules (applied in order, every decision recorded in the report):

1. Unknown segment kinds are dropped.
2. Evidence IDs the model was not given are stripped (fabricated citations).
3. ``quote``: after stripping wrapping quote marks, the text must appear
   verbatim (modulo typographic normalization) in at least one cited evidence
   chunk and be at least ``min_quote_words`` long. The chunk must be written by
   the subject (authorship ``by_me``), or be ``secondhand`` reporting in which
   the quote sits inside double quotation marks whose attribution clause names
   the subject (see ``reported_quote``); such quotes carry ``reported_by``.
   Citations are narrowed to the chunks that actually contain it. A failed
   quote is demoted to ``extrapolation``: it is never shown as the subject's words.
4. ``extrapolation`` must cite at least one valid evidence ID, otherwise it is
   removed as unsupported. If the owner disabled extrapolation, all are removed.
5. ``no_evidence`` segments never carry citations.
6. If nothing survives, a single ``no_evidence`` segment is returned.

What this does NOT check: whether an extrapolation is a *fair* reading of its
cited evidence, or whether a quoted remark near the subject's name was really
theirs (the attribution check is a heuristic). The first is measured
by ``anyone eval``; the second is why such quotes are labeled "as reported by".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .store import Evidence
from .text import normalize, strip_wrapping_quotes

Kind = Literal["quote", "extrapolation", "no_evidence"]
KINDS: tuple[str, ...] = ("quote", "extrapolation", "no_evidence")

FALLBACK_NO_EVIDENCE = "I don't have anything on the record that answers this."
ATTRIBUTION_WINDOW = 250
_QUOTED = re.compile(r'"([^"]{1,3000})"')


@dataclass
class Segment:
    kind: Kind
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    verified: bool = False          # True only for quotes that passed the verbatim check
    note: str | None = None         # why a segment was changed, if it was
    reported_by: str | None = None  # outlet, for quotes verified only in secondhand reporting

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "text": self.text, "evidence_ids": self.evidence_ids, "verified": self.verified}
        if self.note:
            d["note"] = self.note
        if self.reported_by:
            d["reported_by"] = self.reported_by
        return d


@dataclass
class ValidationReport:
    proposed: int = 0
    shown: int = 0
    quotes_verified: int = 0
    quotes_demoted: int = 0
    removed: list[str] = field(default_factory=list)
    stripped_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"[^\W\d_][\w'-]*", name) if len(t) >= 3]


_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def reported_quote(needle: str, source: str, subject_name: str) -> bool:
    """True if normalized ``needle`` sits inside a double-quoted span of normalized ``source`` whose
    attribution clause names the subject: one of their name tokens appears between the previous
    sentence end and the opening quote, or between the closing quote and the next sentence end
    (``"...," Reyes said.`` / ``Reyes said: "..."``), within ATTRIBUTION_WINDOW characters.
    Pronoun-only attribution ("she said") is not accepted: false negatives are demoted, never misquoted."""
    tokens = _name_tokens(subject_name)
    if not tokens:
        return False
    name_re = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in tokens) + r")(?!\w)")
    for m in _QUOTED.finditer(source):
        if needle not in m.group(1):
            continue
        before = source[max(0, m.start() - ATTRIBUTION_WINDOW): m.start()]
        ends = list(_SENTENCE_END.finditer(before))
        if ends:
            before = before[ends[-1].end():]
        after = source[m.end(): m.end() + ATTRIBUTION_WINDOW]
        end = _SENTENCE_END.search(after)
        if end:
            after = after[: end.start()]
        if name_re.search(before) or name_re.search(after):
            return True
    return False


def outlet_label(e: Evidence) -> str:
    if e.outlet:
        return e.outlet
    if e.url:
        from urllib.parse import urlparse

        host = urlparse(e.url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    return e.title


def validate_segments(raw_segments: object, evidence: dict[str, Evidence], *, min_quote_words: int = 3,
                      allow_extrapolation: bool = True, subject_name: str = "") -> tuple[list[Segment], ValidationReport]:
    report = ValidationReport()
    out: list[Segment] = []
    items = raw_segments if isinstance(raw_segments, list) else []
    report.proposed = len(items)

    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            report.removed.append(f"segment {i}: not an object")
            continue
        kind = raw.get("kind")
        text = raw.get("text")
        ids = raw.get("evidence_ids", [])
        if kind not in KINDS:
            report.removed.append(f"segment {i}: unknown kind {kind!r}")
            continue
        if not isinstance(text, str) or not text.strip():
            report.removed.append(f"segment {i}: empty text")
            continue
        if not isinstance(ids, list):
            ids = []
        valid_ids: list[str] = []
        for eid in ids:
            if isinstance(eid, str) and eid in evidence:
                if eid not in valid_ids:
                    valid_ids.append(eid)
            else:
                report.stripped_citations.append(f"segment {i}: {eid!r}")

        if kind == "no_evidence":
            out.append(Segment("no_evidence", text.strip()))
            continue

        if kind == "quote":
            quote = strip_wrapping_quotes(text)
            needle = normalize(quote)
            found = [eid for eid in valid_ids if needle and needle in normalize(evidence[eid].text)]
            containing = [eid for eid in found if evidence[eid].quotable]
            reported = [eid for eid in found if evidence[eid].authorship == "secondhand"
                        and reported_quote(needle, normalize(evidence[eid].text), subject_name)]
            long_enough = len(needle.split()) >= min_quote_words
            if containing and long_enough:
                out.append(Segment("quote", quote, containing, verified=True))
                report.quotes_verified += 1
                continue
            if reported and long_enough:
                outlets = list(dict.fromkeys(outlet_label(evidence[eid]) for eid in reported))
                out.append(Segment("quote", quote, reported, verified=True, reported_by=", ".join(outlets)))
                report.quotes_verified += 1
                continue
            report.quotes_demoted += 1
            secondhand = [eid for eid in found if evidence[eid].authorship == "secondhand"]
            if (containing or reported) and not long_enough:
                reason = "quote shorter than minimum length"
            elif secondhand and not containing:
                reason = "found in secondhand reporting, but not as a quoted remark attributed to the subject"
            elif found and not containing:
                reason = "source is marked AI-assisted, so it is not shown as the subject's own words"
            else:
                reason = "quote not found verbatim in cited sources"
            kind, text = "extrapolation", quote
            note = f"demoted: {reason}"
        else:
            note = None

        # extrapolation (original or demoted)
        if not allow_extrapolation:
            report.removed.append(f"segment {i}: extrapolation disabled by owner")
            continue
        if not valid_ids:
            report.removed.append(f"segment {i}: extrapolation without valid evidence")
            continue
        out.append(Segment("extrapolation", text.strip(), valid_ids, note=note))

    if not out:
        out.append(Segment("no_evidence", FALLBACK_NO_EVIDENCE))
    report.shown = len(out)
    return out, report
