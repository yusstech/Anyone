"""Citation and quote validator — the trust core of CiteSelf.

The model *proposes* labeled segments; this module decides what the user sees.
Rules (applied in order, every decision recorded in the report):

1. Unknown segment kinds are dropped.
2. Evidence IDs the model was not given are stripped (fabricated citations).
3. ``quote``: after stripping wrapping quote marks, the text must appear
   verbatim (modulo typographic normalization) in at least one cited evidence
   chunk and be at least ``min_quote_words`` long. Citations are narrowed to the
   chunks that actually contain it. A failed quote is demoted to
   ``extrapolation`` — it is never shown as the owner's words.
4. ``extrapolation`` must cite at least one valid evidence ID, otherwise it is
   removed as unsupported. If the owner disabled extrapolation, all are removed.
5. ``no_evidence`` segments never carry citations.
6. If nothing survives, a single ``no_evidence`` segment is returned.

What this does NOT check: whether an extrapolation is a *fair* reading of its
cited evidence. That is a semantic judgment, measured by ``citeself eval``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .store import Evidence
from .text import normalize, strip_wrapping_quotes

Kind = Literal["quote", "extrapolation", "no_evidence"]
KINDS: tuple[str, ...] = ("quote", "extrapolation", "no_evidence")

FALLBACK_NO_EVIDENCE = "I don't have anything on the record that answers this."


@dataclass
class Segment:
    kind: Kind
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    verified: bool = False          # True only for quotes that passed the verbatim check
    note: str | None = None         # why a segment was changed, if it was

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "text": self.text, "evidence_ids": self.evidence_ids, "verified": self.verified}
        if self.note:
            d["note"] = self.note
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


def validate_segments(raw_segments: object, evidence: dict[str, Evidence], *, min_quote_words: int = 3,
                      allow_extrapolation: bool = True) -> tuple[list[Segment], ValidationReport]:
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
            containing = [eid for eid in valid_ids if needle and needle in normalize(evidence[eid].text)]
            if containing and len(needle.split()) >= min_quote_words:
                out.append(Segment("quote", quote, containing, verified=True))
                report.quotes_verified += 1
                continue
            report.quotes_demoted += 1
            reason = ("quote shorter than minimum length" if containing
                      else "quote not found verbatim in cited sources")
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
