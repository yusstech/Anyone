"""Text extraction, normalization and chunking.

Normalization is deliberately narrow: it only removes differences that are
typographic (curly vs straight quotes, dash variants, whitespace runs, Unicode
compatibility forms). Case and wording are never normalized, so a "verbatim"
match still means the owner actually wrote those words in that order.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

_TYPOGRAPHIC = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
})
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Canonical form used only for verbatim comparison."""
    text = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHIC)
    return _WS.sub(" ", text).strip()


_QUOTE_WRAP = "\"'“”‘’«»"


def strip_wrapping_quotes(text: str) -> str:
    t = text.strip()
    while len(t) >= 2 and t[0] in _QUOTE_WRAP and t[-1] in _QUOTE_WRAP:
        t = t[1:-1].strip()
    return t


# ---------------------------------------------------------------- extraction

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_MD_RULES = [
    (re.compile(r"```.*?```", re.S), lambda m: m.group(0)),       # keep code fences as-is
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),                # images -> alt text
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),                 # links -> text
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),                  # headings
    (re.compile(r"^\s{0,3}>\s?", re.M), ""),                       # blockquotes
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),                      # bold
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"), r"\1"),  # italic *x*
    (re.compile(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])"), r"\1"),    # italic _x_
    (re.compile(r"`([^`\n]+)`"), r"\1"),                           # inline code
]


def parse_front_matter(raw: str) -> tuple[dict, str]:
    m = _FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    import yaml  # local import keeps module import cheap

    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), raw[m.end():]


def markdown_to_text(md: str) -> str:
    for pattern, repl in _MD_RULES:
        md = pattern.sub(repl, md)
    return md


class _HTMLText(HTMLParser):
    """Extracts readable text. If the page has a <main>, only its content is kept;
    navigation, headers, footers and forms are always dropped as site chrome."""
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "blockquote",
             "main", "figcaption", "dd", "dt", "pre"}
    SKIP = {"script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside", "form", "button",
            "iframe", "select"}
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
    DATE_META = ("article:published_time", "datepublished", "date", "dc.date", "pubdate")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.main_parts: list[str] = []
        self.title = ""
        self.date: str | None = None
        self._skip = 0
        self._main = 0
        self._saw_main = False
        self._in_title = False

    def _emit(self, text: str) -> None:
        self.parts.append(text)
        if self._main:
            self.main_parts.append(text)

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            a = {k.lower(): (v or "") for k, v in attrs}
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key in self.DATE_META and a.get("content") and not self.date:
                self.date = a["content"].strip()[:10]
            return
        if tag in self.VOID:
            if tag == "br":
                self._emit("\n")
            return
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "main":
            self._main += 1
            self._saw_main = True
        if tag in self.BLOCK and not self._skip:
            self._emit("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag == "main":
            self._main = max(0, self._main - 1)
        if tag in self.BLOCK and not self._skip:
            self._emit("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self._emit(data)

    @property
    def text(self) -> str:
        return "".join(self.main_parts if self._saw_main and "".join(self.main_parts).strip() else self.parts)


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MAX_DOCX_XML = 50 * 1024 * 1024


def _docx(path: Path) -> "Extracted":
    import zipfile
    import xml.etree.ElementTree as ET

    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError("not a valid .docx file") from exc
    with zf:
        def read(name: str) -> bytes | None:
            try:
                info = zf.getinfo(name)
            except KeyError:
                return None
            if info.file_size > _MAX_DOCX_XML:
                raise ValueError(f"{name} inside the .docx is too large")
            data = zf.read(name)
            if b"<!DOCTYPE" in data[:2048].upper():  # Word never emits DTDs; refuse entity tricks
                raise ValueError("unexpected DTD in .docx")
            return data

        body = read("word/document.xml")
        if body is None:
            raise ValueError("not a Word document (word/document.xml missing)")
        paras: list[str] = []
        for p in ET.fromstring(body).iter(f"{_W}p"):
            buf: list[str] = []
            for node in p.iter():
                if node.tag == f"{_W}t" and node.text:
                    buf.append(node.text)
                elif node.tag == f"{_W}tab":
                    buf.append("\t")
                elif node.tag in (f"{_W}br", f"{_W}cr"):
                    buf.append("\n")
            paras.append("".join(buf))
        title, date = "", None
        core = read("docProps/core.xml")
        if core is not None:
            root = ET.fromstring(core)
            for el in root:
                tag = el.tag.rsplit("}", 1)[-1]
                if tag == "title" and el.text:
                    title = el.text.strip()
                elif tag == "created" and el.text and not date:
                    date = el.text.strip()[:10]
    return Extracted(text="\n\n".join(paras), title=title or path.stem, date=date)


@dataclass
class Extracted:
    text: str
    title: str
    date: str | None


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"}


def extract(path: Path) -> Extracted:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.name} (supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ValueError("PDF support needs the 'pdf' extra: pip install 'anyone-ai[pdf]'") from exc
        reader = PdfReader(str(path))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        title = (reader.metadata.title if reader.metadata and reader.metadata.title else "") or path.stem
        return Extracted(text=text, title=title.strip(), date=None)

    if suffix == ".docx":
        return _docx(path)

    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".html", ".htm"):
        parser = _HTMLText()
        parser.feed(raw)
        parser.close()
        return Extracted(text=parser.text, title=" ".join(parser.title.split()) or path.stem, date=parser.date)

    meta, body = parse_front_matter(raw)
    title = str(meta.get("title") or "").strip()
    if suffix in (".md", ".markdown"):
        if not title:
            h = re.search(r"^\s{0,3}#\s+(.+)$", body, re.M)
            title = h.group(1).strip() if h else ""
        body = markdown_to_text(body)
    date = meta.get("date")
    return Extracted(text=body, title=title or path.stem, date=str(date) if date else None)


# ------------------------------------------------------------------ chunking

@dataclass
class Chunk:
    ordinal: int
    start: int
    end: int
    text: str


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Tidy extracted text while keeping paragraph breaks."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, target: int = 700, hard_max: int = 1200) -> list[Chunk]:
    """Greedy paragraph packing with sentence splitting for oversized paragraphs.

    Offsets refer to positions in ``text`` so a chunk can always be traced back
    to its exact location in the stored source.
    """
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"[^\n](?:.|\n(?!\n))*", text):
        start, end = m.span()
        if end - start <= hard_max:
            spans.append((start, end))
            continue
        cursor = start
        for sm in _SENTENCE_END.finditer(text, start, end):
            if sm.start() - cursor >= target:
                spans.append((cursor, sm.start()))
                cursor = sm.end()
        if cursor < end:
            spans.append((cursor, end))

    chunks: list[Chunk] = []
    cur_start = cur_end = None
    for start, end in spans:
        if cur_start is None:
            cur_start, cur_end = start, end
        elif end - cur_start <= target:
            cur_end = end
        else:
            chunks.append(Chunk(len(chunks), cur_start, cur_end, text[cur_start:cur_end]))
            cur_start, cur_end = start, end
    if cur_start is not None:
        chunks.append(Chunk(len(chunks), cur_start, cur_end, text[cur_start:cur_end]))
    return chunks


# ---------------------------------------------------------------- retrieval

STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if in into is it its "
    "me my no not of on or our she he her his that the their them then there these they this those to was we "
    "were what when where which who whom why will with would you your yours about just also than so very".split()
)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def query_terms(text: str, limit: int = 24) -> list[str]:
    seen: list[str] = []
    for w in _WORD.findall(text.lower()):
        if len(w) > 2 and w not in STOPWORDS and w not in seen:
            seen.append(w)
    return seen[:limit]
