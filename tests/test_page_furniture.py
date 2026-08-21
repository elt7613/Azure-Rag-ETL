"""Running headers and footers must not reach the index.

Left in, a line like "Northwind Traders, Inc. — Internal Use Only Page 2"
becomes its own chunk, dilutes the embedding of every chunk it lands in, and
burns context tokens in the answer prompt. It is noise in three places at once.
"""
from __future__ import annotations

import pathlib

import pytest

from rag.parsing.local_parser import (
    _detect_page_furniture,
    _furniture_signature,
    LocalParser,
)

CORPUS = pathlib.Path("source_data")
MULTI_PAGE_PDFS = sorted(str(p.relative_to(CORPUS)) for p in CORPUS.rglob("*.pdf"))


def test_signature_masks_page_numbers():
    """A footer carrying a page number is unique on every page until masked."""
    assert _furniture_signature("Acme Ltd — Internal Page 1") == _furniture_signature(
        "Acme  Ltd — Internal   Page 12"
    )


def test_signature_is_case_and_whitespace_insensitive():
    assert _furniture_signature("Confidential\tDraft") == _furniture_signature(
        "confidential draft"
    )


def test_line_repeated_in_the_edge_band_is_furniture():
    footer = "Acme Ltd - Internal Use Only Page 1"
    pages = [
        [(40.0, "Heading one"), (700.0, footer)],
        [(40.0, "Heading two"), (700.0, "Acme Ltd - Internal Use Only Page 2")],
    ]
    detected = _detect_page_furniture(pages, [792.0, 792.0])
    assert _furniture_signature(footer) in detected


def test_repeated_body_text_is_not_furniture():
    """Repetition alone is not enough — position matters.

    A sentence that genuinely recurs mid-page (a standard clause, a repeated
    warning) is content. Only the top and bottom bands are treated as furniture.
    """
    clause = "Each Party shall protect the other Party's Confidential Information."
    pages = [
        [(400.0, clause), (700.0, "Page 1")],
        [(380.0, clause), (700.0, "Page 2")],
    ]
    detected = _detect_page_furniture(pages, [792.0, 792.0])
    assert _furniture_signature(clause) not in detected


def test_single_page_document_has_no_furniture():
    """With one page there is no repetition, so nothing can be inferred."""
    pages = [[(700.0, "Acme Ltd - Internal Use Only Page 1")]]
    assert _detect_page_furniture(pages, [792.0]) == set()


def test_line_on_only_one_of_many_pages_is_kept():
    pages = [
        [(700.0, "unique closing note")],
        [(700.0, "Acme footer")],
        [(700.0, "Acme footer")],
        [(700.0, "Acme footer")],
    ]
    detected = _detect_page_furniture(pages, [792.0] * 4)
    assert _furniture_signature("Acme footer") in detected
    assert _furniture_signature("unique closing note") not in detected


@pytest.mark.parametrize("doc_id", MULTI_PAGE_PDFS)
async def test_real_corpus_pdfs_carry_no_running_footer(doc_id):
    parsed = await LocalParser().parse((CORPUS / doc_id).read_bytes(), doc_id)
    leaked = [b.text for b in parsed.blocks if "Internal Use Only" in b.text]
    assert leaked == [], f"page furniture reached the block stream: {leaked}"


@pytest.mark.parametrize("doc_id", MULTI_PAGE_PDFS)
async def test_stripping_furniture_does_not_cost_real_content(doc_id):
    """The metadata header and real body text must survive the filter."""
    parsed = await LocalParser().parse((CORPUS / doc_id).read_bytes(), doc_id)
    assert parsed.raw_header.count("|") >= 2, "document header was stripped as furniture"
    assert len(parsed.blocks) > 10
    assert any(b.type.value == "heading" for b in parsed.blocks)
