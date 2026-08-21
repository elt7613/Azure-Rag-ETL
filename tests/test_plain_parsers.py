"""Behaviour of the text/office parsers added by the universal router.

Every case ends by asserting the same thing the pipeline needs: that the
blocks survive `build_section_tree` -> `chunk_document`. A parser that emits
a plausible-looking block list but breaks the enrichment contract is not
useful, so the round trip -- not the block list -- is what these tests treat
as the deliverable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.enrich.chunker import chunk_document
from rag.enrich.metadata import extract_metadata
from rag.enrich.structure import build_section_tree
from rag.models import BlockType, ParsedDocument
from rag.parsing.errors import MalformedDocumentError
from rag.parsing.plain import PlainParser, decode_text
from rag.parsing.pptx import PptxParser

FIXTURES = Path(__file__).parent / "fixtures"


def read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def assert_enrichable(doc: ParsedDocument) -> None:
    """The real contract: these blocks must flow through enrichment."""
    sections = build_section_tree(doc)
    assert sections, "no sections built from parsed blocks"
    meta = extract_metadata(doc, doc.doc_id)
    chunks = chunk_document(doc, meta)
    assert chunks, "no chunks produced"
    assert all(c.display_text.strip() for c in chunks)
    assert all(c.embed_text.startswith(meta.title) for c in chunks)


# ---------------------------------------------------------------- markdown


async def test_markdown_atx_headings_carry_level():
    doc = await PlainParser().parse(read("sample.md"), "HR/sample.md")
    headings = {b.text: b.level for b in doc.blocks if b.type is BlockType.HEADING}
    assert headings["Remote Work Policy"] == 1
    assert headings["1 Eligibility"] == 2
    assert headings["2 Equipment Stipend"] == 2


async def test_markdown_bullets_become_list_blocks():
    doc = await PlainParser().parse(read("sample.md"), "HR/sample.md")
    lists = [b.text for b in doc.blocks if b.type is BlockType.LIST]
    assert "Manager approval is required." in lists


async def test_markdown_fenced_code_is_one_paragraph_not_a_table():
    doc = await PlainParser().parse(read("sample.md"), "HR/sample.md")
    code = [b for b in doc.blocks
            if b.type is BlockType.PARAGRAPH and "stipend = 700" in b.text]
    assert len(code) == 1


async def test_markdown_pipe_table_becomes_a_table_block():
    doc = await PlainParser().parse(read("sample.md"), "HR/sample.md")
    tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
    assert len(tables) == 1
    assert tables[0].rows[0] == ["Item", "Amount"]
    assert ["Desk", "$400"] in tables[0].rows


async def test_markdown_metadata_line_becomes_raw_header_not_a_block():
    doc = await PlainParser().parse(read("sample.md"), "HR/sample.md")
    assert "Effective: 2026-01-01" in doc.raw_header
    assert not any("Effective: 2026-01-01" in b.text for b in doc.blocks)
    meta = extract_metadata(doc, "HR/sample.md")
    assert meta.version == "3.1"


async def test_markdown_round_trips_through_enrichment():
    doc = await PlainParser().parse(read("sample.md"), "HR/sample.md")
    assert_enrichable(doc)


# -------------------------------------------------------------------- html


async def test_html_headings_carry_level_and_scripts_are_stripped():
    doc = await PlainParser().parse(read("sample.html"), "IT/sample.html")
    headings = {b.text: b.level for b in doc.blocks if b.type is BlockType.HEADING}
    assert headings["Security Standard"] == 1
    assert headings["1 Password Rules"] == 2
    assert headings["1.1 Lockout Thresholds"] == 3
    body = " ".join(b.text for b in doc.blocks)
    assert "console.log" not in body
    assert "color: #333" not in body


async def test_html_entities_are_decoded_and_lists_preserved():
    doc = await PlainParser().parse(read("sample.html"), "IT/sample.html")
    assert any("14 characters & rotated" in b.text for b in doc.blocks)
    lists = [b.text for b in doc.blocks if b.type is BlockType.LIST]
    assert "MFA is mandatory for administrators." in lists


async def test_html_table_rows_are_recovered():
    doc = await PlainParser().parse(read("sample.html"), "IT/sample.html")
    table = next(b for b in doc.blocks if b.type is BlockType.TABLE)
    assert table.rows[0] == ["Attempts", "Lockout"]
    assert ["10", "24 hours"] in table.rows


async def test_html_round_trips_through_enrichment():
    doc = await PlainParser().parse(read("sample.html"), "IT/sample.html")
    assert_enrichable(doc)


# --------------------------------------------------------------- csv / tsv


async def test_csv_groups_into_tables_like_the_xlsx_path():
    doc = await PlainParser().parse(read("sample.csv"), "sales/sample.csv")
    tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
    assert len(tables) == 2, "blank rows should split the file into two tables"
    prices = {r[0]: r[1] for r in tables[0].rows[1:]}
    assert prices["Professional"] == "$65"
    assert tables[1].rows[0] == ["Region", "Quota"]


async def test_csv_metadata_line_is_lifted_into_raw_header():
    doc = await PlainParser().parse(read("sample.csv"), "sales/sample.csv")
    assert "Version 4.2" in doc.raw_header
    assert not any(
        any("Version 4.2" in cell for cell in row)
        for b in doc.blocks if b.rows for row in b.rows
    )


async def test_tsv_uses_tab_delimiter():
    doc = await PlainParser().parse(read("sample.tsv"), "finance/sample.tsv")
    table = next(b for b in doc.blocks if b.type is BlockType.TABLE)
    assert table.rows[0] == ["Country", "VAT Rate"]
    assert ["Germany", "19%"] in table.rows


async def test_csv_round_trips_through_enrichment():
    doc = await PlainParser().parse(read("sample.csv"), "sales/sample.csv")
    assert_enrichable(doc)


# --------------------------------------------------------------------- txt


async def test_txt_numeric_prefixes_become_headings():
    doc = await PlainParser().parse(read("sample.txt"), "finance/sample.txt")
    headings = {b.text: b.level for b in doc.blocks if b.type is BlockType.HEADING}
    assert headings["1 Scope"] == 1
    assert headings["1.1 Airfare"] == 2


async def test_txt_round_trips_through_enrichment():
    doc = await PlainParser().parse(read("sample.txt"), "finance/sample.txt")
    assert_enrichable(doc)


# -------------------------------------------------------------------- json


async def test_json_scalars_become_flattened_key_path_paragraphs():
    doc = await PlainParser().parse(read("sample.json"), "finance/sample.json")
    paragraphs = [b.text for b in doc.blocks if b.type is BlockType.PARAGRAPH]
    assert "policy: Expense Limits" in paragraphs
    assert "meta.version: 2.3" in paragraphs


async def test_json_uniform_object_array_becomes_a_table():
    doc = await PlainParser().parse(read("sample.json"), "finance/sample.json")
    table = next(b for b in doc.blocks if b.type is BlockType.TABLE)
    assert table.rows[0] == ["category", "daily_cap", "currency"]
    assert ["Lodging", "250", "USD"] in table.rows


async def test_json_round_trips_through_enrichment():
    doc = await PlainParser().parse(read("sample.json"), "finance/sample.json")
    assert_enrichable(doc)


async def test_malformed_json_raises_typed_error_not_value_error():
    with pytest.raises(MalformedDocumentError):
        await PlainParser().parse(read("broken.json"), "finance/broken.json")


# -------------------------------------------------------------------- pptx


async def test_pptx_slide_title_is_a_level_one_heading():
    doc = await PptxParser().parse(read("sample.pptx"), "sales/sample.pptx")
    headings = [b for b in doc.blocks if b.type is BlockType.HEADING]
    assert [h.text for h in headings] == ["Quarterly Business Review",
                                          "Discount Ladder"]
    assert all(h.level == 1 for h in headings)


async def test_pptx_page_is_the_slide_number():
    doc = await PptxParser().parse(read("sample.pptx"), "sales/sample.pptx")
    by_page = {b.page for b in doc.blocks}
    assert by_page == {1, 2}
    ladder = next(b for b in doc.blocks if b.text == "Discount Ladder")
    assert ladder.page == 2


async def test_pptx_body_bullets_and_tables():
    doc = await PptxParser().parse(read("sample.pptx"), "sales/sample.pptx")
    lists = [b.text for b in doc.blocks if b.type is BlockType.LIST]
    assert "Revenue grew 18% year over year." in lists
    table = next(b for b in doc.blocks if b.type is BlockType.TABLE)
    assert table.rows[0] == ["Tier", "Max Discount"]
    assert ["Enterprise", "18%"] in table.rows


async def test_pptx_metadata_line_becomes_raw_header():
    doc = await PptxParser().parse(read("sample.pptx"), "sales/sample.pptx")
    assert "Version 1.2" in doc.raw_header


async def test_pptx_round_trips_through_enrichment():
    doc = await PptxParser().parse(read("sample.pptx"), "sales/sample.pptx")
    assert_enrichable(doc)


# ------------------------------------------------------- encoding hardening


def test_decode_text_falls_back_to_cp1252_for_non_utf8_bytes():
    text = decode_text(read("latin1.txt"))
    assert "employee’s notice period" in text
    assert "“30 days”" in text


def test_decode_text_never_raises_on_arbitrary_bytes():
    assert isinstance(decode_text(bytes(range(256))), str)


def test_decode_text_strips_the_byte_order_mark():
    assert not decode_text(read("bom.md")).startswith("﻿")


async def test_bom_does_not_leak_into_the_first_heading():
    doc = await PlainParser().parse(read("bom.md"), "legal/bom.md")
    first = doc.blocks[0]
    assert first.type is BlockType.HEADING
    assert first.text == "Vendor Onboarding"


async def test_non_utf8_text_file_parses_without_crashing():
    doc = await PlainParser().parse(read("latin1.txt"), "HR/latin1.txt")
    assert any(b.type is BlockType.HEADING and b.text == "1 Notice Period"
               for b in doc.blocks)
