"""Text-shaped formats: txt, md, html/htm, csv/tsv, json.

None of these need a service call, and routing them to Document Intelligence
would be both slower and worse -- DI OCRs a rendering of the bytes, whereas
these formats already state their own structure. The job here is to recover
that structure into the *same* `Block` vocabulary the PDF and DOCX parsers
emit, because everything downstream (`build_section_tree`, the chunker, the
graph writer) is written against blocks, not against formats. A parser that
invented its own conventions would work in isolation and break the pipeline.

The shared conventions come from `rag.parsing.base`: a numbered line is a
heading whose level is its dot depth, and a pipe-delimited metadata line is
lifted out of the block stream into `raw_header` rather than indexed as
content.
"""
from __future__ import annotations

import codecs
import csv
import io
import json
import re
from html.parser import HTMLParser
from typing import Any

from rag.parsing.base import classify_line, is_header_line
from rag.parsing.errors import MalformedDocumentError
from rag.models import Block, BlockType, ParsedDocument

# Ordered longest-first: the UTF-32 marks start with the UTF-16 ones, so
# checking UTF-16 first would mis-decode every UTF-32 file.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def decode_text(data: bytes) -> str:
    """Bytes to text without ever raising.

    Enterprise corpora are full of files exported from Excel and Outlook on
    Windows, which are cp1252 rather than UTF-8; a strict UTF-8 decode turns
    every curly apostrophe into an unhandled `UnicodeDecodeError` and, under
    the never-crash contract, into a document that silently never gets
    indexed. So: honour a byte-order mark if present, then UTF-8, then
    cp1252, and finally latin-1 with replacement -- which cannot fail for any
    byte sequence, so this function always returns a string.

    The BOM is stripped rather than kept. Left in, it becomes the first
    character of the first line, so "\\ufeff# Title" no longer matches the
    heading rule and the document loses its top-level section.
    """
    text: str | None = None
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            try:
                text = data.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                text = None
            break
    if text is None:
        for encoding in ("utf-8", "cp1252"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
    if text is None:
        text = data.decode("latin-1", errors="replace")
    # A BOM can also appear mid-file where files were concatenated, and it is
    # never meaningful content.
    return text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")


# ------------------------------------------------------------- plain text


def _parse_plain(text: str) -> tuple[list[Block], str]:
    blocks: list[Block] = []
    header = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if not header and is_header_line(stripped):
            header = stripped
            continue
        blocks.append(classify_line(stripped, page=1))
    return blocks, header


# --------------------------------------------------------------- markdown

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_BULLET_RE = re.compile(r"^[-*+]\s+(.+)$")
_FENCE_RE = re.compile(r"^(```|~~~)")
_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def _split_md_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_markdown(text: str) -> tuple[list[Block], str]:
    blocks: list[Block] = []
    header = ""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            i += 1
            continue

        if _FENCE_RE.match(line):
            # A fenced block is one unit: splitting code across blocks would
            # scatter it into separate chunks and destroy the only thing that
            # makes it readable.
            fence = line[:3]
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                body.append(lines[i])
                i += 1
            i += 1  # closing fence, or past the end if the block is unterminated
            code = "\n".join(body).strip()
            if code:
                blocks.append(Block(type=BlockType.PARAGRAPH, text=code, page=1))
            continue

        # A GFM pipe table is recognised by its separator row, not by pipes
        # alone -- which also keeps a real table's rows from being mistaken
        # for the document's pipe-delimited metadata header.
        if (line.startswith("|") and i + 1 < len(lines)
                and _TABLE_SEP_RE.match(lines[i + 1].strip())):
            rows = [_split_md_row(line)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_md_row(lines[i]))
                i += 1
            blocks.append(Block(type=BlockType.TABLE, text="", page=1,
                                rows=_rectangular(rows)))
            continue

        atx = _ATX_RE.match(line)
        if atx:
            hashes, title = atx.groups()
            if title:
                blocks.append(Block(type=BlockType.HEADING, text=title,
                                    page=1, level=len(hashes)))
            i += 1
            continue

        if not header and is_header_line(line):
            header = line
            i += 1
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            blocks.append(Block(type=BlockType.LIST, text=bullet.group(1).strip(),
                                page=1))
            i += 1
            continue

        # Numbered section headings ("2.1 Airfare") are the house convention
        # across the corpus and appear in markdown too, so an unprefixed line
        # gets the same treatment it would in a PDF. The cost is that a
        # genuine ordered-list item reads as a heading; the benefit is that
        # markdown without any `#` still produces a section tree, and a
        # document with no sections is a document with no citable location.
        blocks.append(classify_line(line, page=1))
        i += 1
    return blocks, header


# ------------------------------------------------------------------- html

# Content that is markup machinery, never prose. `title` is excluded too: it
# duplicates the h1 in practice and would otherwise land as a stray paragraph
# ahead of the document's real first heading.
_HTML_SKIP = {"script", "style", "noscript", "title"}
_HTML_HEADINGS = {f"h{n}": n for n in range(1, 7)}
# Tags that end whatever text run preceded them.
_HTML_BREAKS = {
    "p", "div", "section", "article", "header", "footer", "main", "aside",
    "blockquote", "pre", "ul", "ol", "dl", "dt", "dd", "figcaption", "hr",
}


class _HtmlBlockBuilder(HTMLParser):
    """Folds an HTML stream into the shared block vocabulary.

    `convert_charrefs=True` (the default) means `handle_data` already
    receives unescaped text, so entities never reach the index as `&amp;`.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.header = ""
        self._buf: list[str] = []
        self._type = BlockType.PARAGRAPH
        self._level = 0
        self._skip = 0
        # Tables nest, so the open ones are a stack; the innermost finishes
        # first and is emitted before its parent.
        self._tables: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    # ---- emission ----

    def _flush(self) -> None:
        if self._cell is not None:
            return  # text inside a cell belongs to the table, not to a block
        text = " ".join("".join(self._buf).split())
        self._buf.clear()
        block_type, level = self._type, self._level
        self._type, self._level = BlockType.PARAGRAPH, 0
        if not text:
            return
        if block_type is BlockType.PARAGRAPH and not self.header and is_header_line(text):
            self.header = text
            return
        self.blocks.append(Block(type=block_type, text=text, page=1, level=level))

    # ---- HTMLParser hooks ----

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _HTML_SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in _HTML_HEADINGS:
            self._flush()
            self._type, self._level = BlockType.HEADING, _HTML_HEADINGS[tag]
        elif tag == "li":
            self._flush()
            self._type = BlockType.LIST
        elif tag == "table":
            self._flush()
            self._tables.append([])
        elif tag == "tr" and self._tables:
            self._row = []
        elif tag in ("td", "th") and self._tables:
            self._cell = []
        elif tag == "br":
            self._buf.append(" ")
        elif tag in _HTML_BREAKS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _HTML_SKIP:
            self._skip = max(self._skip - 1, 0)
            return
        if self._skip:
            return
        if tag in ("td", "th"):
            if self._cell is not None and self._row is not None:
                self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr":
            if self._row is not None and self._tables:
                self._tables[-1].append(self._row)
            self._row = None
        elif tag == "table":
            if self._tables:
                rows = self._tables.pop()
                if rows:
                    self.blocks.append(Block(type=BlockType.TABLE, text="",
                                             page=1, rows=_rectangular(rows)))
        elif tag in _HTML_HEADINGS or tag == "li" or tag in _HTML_BREAKS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._cell is not None:
            self._cell.append(data)
        else:
            self._buf.append(data)

    def close(self) -> None:  # noqa: D102 - inherited contract
        super().close()
        self._flush()


def _parse_html(text: str) -> tuple[list[Block], str]:
    builder = _HtmlBlockBuilder()
    builder.feed(text)
    builder.close()
    return builder.blocks, builder.header


# ---------------------------------------------------------------- csv/tsv


def _rectangular(rows: list[list[str]]) -> list[list[str]]:
    """Pad ragged rows so `Block.to_markdown` renders a valid grid.

    Real CSVs have short rows; a markdown table with a row narrower than its
    header renders as broken markup in the answer the user finally reads.
    """
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def _csv_delimiter(file_format: str, text: str) -> str:
    if file_format == "tsv":
        return "\t"
    # Deliberately not `csv.Sniffer`: it guesses from a sample and is happy to
    # pick "|" or " " on a file whose first line is the pipe-delimited
    # metadata header, silently shredding the whole document. Two explicit
    # cases cover the exports that actually occur (comma, and the semicolon
    # that European Excel locales produce).
    first = next((ln for ln in text.split("\n") if ln.strip()), "")
    if ";" in first and "," not in first:
        return ";"
    return ","


def _parse_delimited(text: str, file_format: str) -> tuple[list[Block], str]:
    from rag.parsing.local_parser import LocalParser

    delimiter = _csv_delimiter(file_format, text)
    blocks: list[Block] = []
    header = ""
    group: list[list[str]] = []

    def flush_group() -> None:
        nonlocal group
        if group:
            # Same grouping rule as the XLSX path, by direct reuse rather than
            # by a second copy: a run is a TABLE only when its first row is a
            # genuine column-header row, otherwise it is label/value prose.
            blocks.extend(LocalParser._xlsx_group_to_blocks(_rectangular(group)))
            group = []

    for row in csv.reader(io.StringIO(text), delimiter=delimiter):
        cells = [(c or "").strip() for c in row]
        if not any(cells):
            flush_group()
            continue
        joined = " ".join(c for c in cells if c)
        if is_header_line(joined):
            # The metadata line is document-level, not a data row; indexing it
            # as one would put "Version 4.2" in the price table.
            if not header:
                header = joined
            continue
        group.append(cells)
    flush_group()
    return blocks, header


# ------------------------------------------------------------------- json


def _is_object_array(node: Any) -> bool:
    """A list of dicts sharing a key set -- i.e. a table someone wrote as JSON."""
    if not isinstance(node, list) or len(node) < 2:
        return False
    if not all(isinstance(item, dict) for item in node):
        return False
    keys = list(node[0].keys())
    return bool(keys) and all(list(item.keys()) == keys for item in node[1:])


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _emit_json(node: Any, path: str, blocks: list[Block], depth: int) -> None:
    if _is_object_array(node):
        rows = [list(node[0].keys())]
        rows.extend([_scalar(item.get(k)) for k in rows[0]] for item in node)
        blocks.append(Block(type=BlockType.TABLE, text="", page=1, rows=rows))
        return
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            # Only top-level containers become headings. Deeper ones would
            # produce a section per nested object, which for a
            # machine-generated document means hundreds of one-line sections
            # and a section tree that carries no information.
            if depth == 0 and isinstance(value, (dict, list)):
                blocks.append(Block(type=BlockType.HEADING, text=str(key),
                                    page=1, level=1))
            _emit_json(value, child, blocks, depth + 1)
        return
    if isinstance(node, list):
        for index, item in enumerate(node):
            _emit_json(item, f"{path}[{index}]", blocks, depth + 1)
        return
    blocks.append(Block(type=BlockType.PARAGRAPH,
                        text=f"{path}: {_scalar(node)}".strip(), page=1))


def _parse_json(text: str, doc_id: str) -> tuple[list[Block], str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedDocumentError(
            f"{doc_id}: not valid JSON ({exc.msg} at line {exc.lineno})", doc_id
        ) from exc
    blocks: list[Block] = []
    _emit_json(payload, "", blocks, 0)
    return blocks, ""


# ----------------------------------------------------------------- parser

_MARKDOWN = {"md", "markdown", "mdown", "mkd"}
_HTML = {"html", "htm", "xhtml"}
_DELIMITED = {"csv", "tsv"}

#: Formats this parser claims. The router consults it; nothing else should
#: need to know how the dispatch below is spelled.
PLAIN_FORMATS: frozenset[str] = frozenset(
    {"txt", "text", "log", "json"} | _MARKDOWN | _HTML | _DELIMITED
)


class PlainParser:
    """Dispatches to the right text-format reader.

    `file_format` overrides the extension, for the case the router cares
    about most: a file whose name says nothing (`invoices/export`, a blob with
    no suffix) but whose bytes say HTML. Sniffing lives in the router so this
    class stays a pure bytes-to-blocks transform.
    """

    def __init__(self, file_format: str = "") -> None:
        self._file_format = file_format.lower().lstrip(".")

    async def parse(self, data: bytes, doc_id: str) -> ParsedDocument:
        file_format = self._file_format or doc_id.rsplit(".", 1)[-1].lower()
        text = decode_text(data)

        if file_format in _MARKDOWN:
            blocks, header = _parse_markdown(text)
        elif file_format in _HTML:
            blocks, header = _parse_html(text)
        elif file_format in _DELIMITED:
            blocks, header = _parse_delimited(text, file_format)
        elif file_format == "json":
            blocks, header = _parse_json(text, doc_id)
        else:
            # Plain text is the safe default rather than an error: an unknown
            # text-ish extension read line by line is still a usable document,
            # and the router has already decided these bytes are text.
            blocks, header = _parse_plain(text)

        return ParsedDocument(doc_id=doc_id, file_format=file_format,
                              blocks=blocks, raw_header=header)
