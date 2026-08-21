"""Azure AI Document Intelligence `prebuilt-layout` parser.

Returns structured tables with cell spans and reading order — the production
equivalent of the local pdfplumber path, and it handles DOCX and XLSX
natively (the same `begin_analyze_document` call accepts raw bytes of any
supported format, so unlike `LocalParser` this module does not branch on
file suffix).

Block-emission conventions are matched to `rag.parsing.local_parser.LocalParser`
so the two parsers are genuinely interchangeable for downstream consumers
(`rag.enrich.structure.build_section_tree`, the chunker):

- A paragraph becomes a HEADING only when it starts with a numeric prefix
  ("1", "2.1", ...), with `level` derived from the dot count — exactly
  LocalParser's `_HEADING_RE` / `_level_of` rule. DI's own `role` field
  (title / sectionHeading) is flat: on multi-level documents (e.g.
  LeavePolicy.pdf) DI tags "2 Types of Leave" and its "2.1 ..." children
  with the *same* sectionHeading role, which would collapse the hierarchy
  if used directly. Numeric-prefix detection is used as the primary signal
  instead, with `role` only as a fallback for genuinely unnumbered
  headings (DI's "title" role).
- The document's pipe-delimited metadata line becomes `raw_header` only
  when it has at least two pipe characters, matching LocalParser's
  `_MIN_HEADER_PIPES` threshold (a single stray "|" shouldn't qualify).
- Paragraphs that fall inside a detected table's character span are
  dropped, matching LocalParser's exclusion of table regions from PDF text
  extraction — DI's `result.paragraphs` otherwise re-emits every table
  cell's content as its own paragraph too.
"""
from __future__ import annotations

import re

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from rag.config import get_settings
from rag.models import Block, BlockType, ParsedDocument

_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
_HEADER_HINT = "|"
_MIN_HEADER_PIPES = 2


def _level_of(number: str) -> int:
    return number.count(".") + 1


def _is_header_line(text: str) -> bool:
    return text.count(_HEADER_HINT) >= _MIN_HEADER_PIPES


def _span_offset(spans) -> int | None:
    if not spans:
        return None
    return spans[0].offset


def _in_any_table(offset: int | None, table_spans: list[tuple[int, int]]) -> bool:
    if offset is None:
        return False
    return any(start <= offset < start + length for start, length in table_spans)


class AzureDocIntParser:
    async def parse(self, data: bytes, doc_id: str) -> ParsedDocument:
        settings = get_settings()
        client = DocumentIntelligenceClient(
            endpoint=settings.azure_docint_endpoint,
            credential=AzureKeyCredential(settings.azure_docint_key),
        )
        async with client:
            poller = await client.begin_analyze_document(
                "prebuilt-layout", body=data, content_type="application/octet-stream"
            )
            result = await poller.result()

        blocks: list[Block] = []
        header = ""

        table_spans: list[tuple[int, int]] = []
        for table in result.tables or []:
            grid = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
            page = 1
            for cell in table.cells:
                grid[cell.row_index][cell.column_index] = (cell.content or "").strip()
                if cell.bounding_regions:
                    page = cell.bounding_regions[0].page_number
            blocks.append(Block(type=BlockType.TABLE, text="", page=page, rows=grid))
            for span in table.spans or []:
                table_spans.append((span.offset, span.length))

        for paragraph in result.paragraphs or []:
            text = (paragraph.content or "").strip()
            if not text:
                continue
            if _in_any_table(_span_offset(paragraph.spans), table_spans):
                continue

            page = (
                paragraph.bounding_regions[0].page_number
                if paragraph.bounding_regions
                else 1
            )

            if not header and _is_header_line(text):
                header = text
                continue

            match = _HEADING_RE.match(text)
            if match and len(match.group(2)) < 80:
                number, title = match.groups()
                blocks.append(
                    Block(
                        type=BlockType.HEADING,
                        text=f"{number} {title}",
                        page=page,
                        level=_level_of(number),
                    )
                )
                continue

            role = (paragraph.role or "").lower()
            if role == "title":
                blocks.append(Block(type=BlockType.HEADING, text=text, page=page, level=1))
            elif role in {"pagefooter", "pagenumber", "pageheader", "footnote"}:
                continue
            else:
                blocks.append(Block(type=BlockType.PARAGRAPH, text=text, page=page))

        return ParsedDocument(
            doc_id=doc_id,
            file_format=doc_id.rsplit(".", 1)[-1].lower(),
            blocks=blocks,
            raw_header=header,
        )
