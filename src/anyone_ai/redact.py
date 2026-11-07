"""Redaction applied at ingest, before any text is stored, indexed or sent to a model.

Quotes are verified against the redacted text, so a placeholder can appear inside a
verified quote but redacted details can never come back out.

PII patterns favour precision over recall: a false positive silently changes text that
is later quoted as verbatim, so each pattern needs an unambiguous signal (an ``@`` with
a domain, phone-style grouping, or a capitalised street name followed by a street type).
Not covered: addresses written house-number-last (e.g. "Hauptstraße 5"), postcodes on
their own, obfuscated emails ("name at domain dot com") and unformatted digit runs.
"""
from __future__ import annotations

import re

EMAIL_TOKEN = "[email redacted]"
PHONE_TOKEN = "[phone redacted]"
ADDRESS_TOKEN = "[address redacted]"
NAME_TOKEN = "[redacted]"

_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9][\w.+-]*@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*\.[A-Za-z]{2,}"
    r"(?![\w-])"
)

# (pattern, min digits, max digits). Ends refuse a following digit group so a longer
# number (an ID, an IP address) is never partially redacted.
_END = r"(?![\w]|[.\-]\d)"
_PHONES: list[tuple[re.Pattern, int, int]] = [
    # International: +44 20 7946 0958, +1 (415) 555-0132, +2348031234567
    (re.compile(r"(?<![\w+])\+\d{1,3}(?:[ .\-]?\(?\d{1,5}\)?){2,6}" + _END), 8, 15),
    # North American: (415) 555-0132, 415.555.0132, 1-800-555-0199
    (re.compile(r"(?<![\w.\-+])(?:1[ .\-])?(?:\(\d{3}\)[ .\-]?|\d{3}[ .\-])\d{3}[ .\-]\d{4}" + _END), 10, 11),
    # National trunk prefix: 020 7946 0958, 0803 123 4567, (020) 7946-0958
    (re.compile(r"(?<![\w.\-+])(?:\(0\d{1,4}\)|0\d{1,4})[ \-]\d{3,4}[ \-]?\d{3,4}" + _END), 9, 11),
]

_STREET_TYPES = (
    "Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Terrace|Crescent|"
    "Close|Parkway|Pkwy|Highway|Hwy|Square|Sq|Circle|Cir|Trail|Mews"
)
_DIRECTION = r"(?:N|S|E|W|NE|NW|SE|SW|North|South|East|West)\.?"
_ADDRESS = re.compile(
    r"(?<![\w.,\-])\d{1,6}[A-Za-z]?(?:-\d{1,6})?,?\s+"         # house number: 12, 221B, 10-12
    rf"(?:{_DIRECTION}\s+)?"
    r"(?:[A-Z][A-Za-z'’.\-]*\s+){1,4}"                          # capitalised street name words
    rf"(?:{_STREET_TYPES})\b\.?"
    rf"(?:\s+(?:N|S|E|W|NE|NW|SE|SW)\b\.?)?"
    r"(?:,?\s+(?:Apt|Apartment|Suite|Ste|Unit|Flat|Floor|Fl)\.?\s*#?[\w\-]+)?"
)
_PO_BOX = re.compile(r"(?<!\w)(?:P\.?\s?O\.?\s*Box|Post Office Box)\s+\d+(?!\w)", re.I)


def _phone(min_digits: int, max_digits: int):
    def repl(m: re.Match) -> str:
        n = sum(ch.isdigit() for ch in m.group(0))
        return PHONE_TOKEN if min_digits <= n <= max_digits else m.group(0)
    return repl


def redact_pii(text: str) -> str:
    """Replace email addresses, phone numbers and street addresses with placeholders."""
    text = _EMAIL.sub(EMAIL_TOKEN, text)
    for pattern, lo, hi in _PHONES:
        text = pattern.sub(_phone(lo, hi), text)
    text = _PO_BOX.sub(ADDRESS_TOKEN, text)
    return _ADDRESS.sub(ADDRESS_TOKEN, text)


def redact_names(text: str, names: list[str]) -> str:
    for name in sorted({n.strip() for n in names if n.strip()}, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", NAME_TOKEN, text, flags=re.I)
    return text
