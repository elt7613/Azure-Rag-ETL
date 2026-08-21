"""Parser interface and the block-emission conventions every parser shares.

Local and Azure implementations are interchangeable, which only holds if they
agree on more than the method signature: `build_section_tree` reads
`BlockType.HEADING` + `level` to build the document hierarchy, and
`extract_metadata` reads `raw_header`. Those two conventions -- numeric
section prefixes as headings, and a pipe-delimited metadata line lifted out of
the block stream -- are therefore defined here once and imported by the
parsers added alongside the router.

`local_parser.py` and `azure_docint.py` predate this module and keep their own
copies of the same rules (with `azure_docint`'s docstring explaining the
match). Consolidating those is a mechanical follow-up, deliberately not done
here so this change stays inside its task boundary.
"""
from __future__ import annotations

import re
from typing import Protocol

from rag.models import Block, BlockType, ParsedDocument

# "2.1 Types of Leave" -- the corpus's section convention, and the only
# heading signal available in formats that carry no styling (txt, OCR output).
_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
# Beyond this a "heading" is really a sentence that happens to start with a
# figure ("2 weeks of notice are required before ...").
_MAX_HEADING_TITLE = 80
_HEADER_HINT = "|"
# Real pipe-delimited metadata lines carry 3-4 fields; a single stray "|" can
# turn up in ordinary prose or a table fragment, so require at least two to
# treat a line as the document's metadata header.
_MIN_HEADER_PIPES = 2


def heading_level(number: str) -> int:
    return number.count(".") + 1


def is_header_line(text: str) -> bool:
    return text.count(_HEADER_HINT) >= _MIN_HEADER_PIPES


def classify_line(line: str, page: int) -> Block:
    """Numbered line -> HEADING with a level; anything else -> PARAGRAPH."""
    text = line.strip()
    match = _HEADING_RE.match(text)
    if match and len(match.group(2)) < _MAX_HEADING_TITLE:
        number, title = match.groups()
        return Block(type=BlockType.HEADING, text=f"{number} {title}",
                     page=page, level=heading_level(number))
    return Block(type=BlockType.PARAGRAPH, text=text, page=page)


class DocumentParser(Protocol):
    async def parse(self, data: bytes, doc_id: str) -> ParsedDocument: ...


def select_parser() -> DocumentParser:
    """The format-agnostic entry point the ETL calls.

    `azure` is answerable without looking at the document -- every format goes
    to Document Intelligence -- so it still returns the concrete parser it
    always did. The other two modes are not: whether a PDF needs OCR is a
    property of its bytes, not its name, and `local` covers seven formats that
    `LocalParser` itself does not implement. Both therefore return a parser
    that defers to `select_parser_for` once the bytes are in hand, and that
    router is where `local` is enforced as "never Document Intelligence".
    Callers keep the one-line `await select_parser().parse(data, doc_id)`.
    """
    from rag.config import get_settings

    if get_settings().doc_parser == "azure":
        from rag.parsing.azure_docint import AzureDocIntParser

        return AzureDocIntParser()
    from rag.parsing.router import AutoParser

    return AutoParser()
