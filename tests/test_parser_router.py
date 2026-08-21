"""Routing decisions for the universal parser router (design §4.1).

These are deliberately assertions about *which parser is chosen*, not about
what it produces: the router's job is the decision, and the parsers have
their own tests. Document Intelligence is never actually called here except
in the one live test at the bottom -- routing to DI is proven by identity
of the returned parser, so the unit suite costs nothing and needs no
credentials.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.models import BlockType, ParsedDocument
from rag.parsing.azure_docint import AzureDocIntParser
from rag.parsing.base import select_parser
from rag.parsing.errors import (
    ParsingError,
    ScannedDocumentError,
    UnsupportedFormatError,
)
from rag.parsing.local_parser import LocalParser
from rag.parsing.plain import PlainParser
from rag.parsing.pptx import PptxParser
from rag.parsing.router import (
    is_scanned_pdf,
    pdf_char_density,
    select_parser_for,
    sniff_format,
)
from tests.conftest import azure_configured

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = Path(__file__).parent.parent / "source_data"


def read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


# --------------------------------------------------- scanned-PDF detection


def test_char_density_is_zero_for_an_image_only_pdf():
    assert pdf_char_density(read("scanned.pdf")) == 0.0


def test_char_density_is_high_for_a_born_digital_pdf():
    density = pdf_char_density((CORPUS / "HR" / "LeavePolicy.pdf").read_bytes())
    assert density > 500


def test_is_scanned_pdf_compares_against_the_supplied_threshold():
    scanned = read("scanned.pdf")
    digital = (CORPUS / "HR" / "LeavePolicy.pdf").read_bytes()
    assert is_scanned_pdf(scanned, threshold=100) is True
    assert is_scanned_pdf(digital, threshold=100) is False
    # A threshold above the document's real density reclassifies it, which is
    # what makes the knob meaningful rather than decorative.
    assert is_scanned_pdf(digital, threshold=100_000) is True


def test_unopenable_pdf_bytes_are_treated_as_scanned():
    """Garbage where a PDF should be is DI's problem, not pdfplumber's."""
    assert is_scanned_pdf(b"%PDF-1.4 truncated nonsense", threshold=100) is True


# ------------------------------------------------------------ format sniff


def test_sniff_prefers_the_extension_when_it_is_known():
    assert sniff_format("HR/policy.md", b"anything at all") == "md"
    assert sniff_format("sales/deck.PPTX", b"") == "pptx"


def test_sniff_falls_back_to_magic_bytes_when_the_extension_is_missing():
    assert sniff_format("HR/no_extension", read("scanned.pdf")) == "pdf"
    assert sniff_format("HR/no_extension", read("sample.png")) == "png"
    assert sniff_format("sales/no_extension", read("sample.pptx")) == "pptx"
    assert sniff_format("IT/no_extension", read("sample.html")) == "html"


def test_sniff_returns_the_unknown_extension_verbatim():
    assert sniff_format("x/mystery.xyz", read("mystery.xyz")) == "xyz"


def test_conclusive_magic_bytes_beat_a_lying_extension():
    """Messy corpora contain renamed files; parsing by name alone fails them."""
    assert sniff_format("HR/policy.docx", read("scanned.pdf")) == "pdf"
    assert sniff_format("HR/chart.txt", read("sample.png")) == "png"


def test_equivalent_spellings_are_not_treated_as_a_mismatch():
    assert sniff_format("HR/photo.jpeg", b"\xff\xd8\xff\xe0rest") == "jpeg"


def test_a_weak_content_guess_never_overrules_a_real_extension():
    assert sniff_format("HR/notes.md", b'{"looks": "like json"}') == "md"


async def test_a_pdf_named_docx_is_still_parsed_as_a_pdf():
    """The routing correction has to survive into the parser, not just the log.

    `LocalParser` dispatches on the extension, so a router that decides "this
    is really a PDF" and then hands over the original name has decided
    nothing.
    """
    data = (CORPUS / "HR/LeavePolicy.pdf").read_bytes()
    doc = await select_parser_for("HR/LeavePolicy.docx", data).parse(
        data, "HR/LeavePolicy.docx"
    )
    assert doc.file_format == "pdf"
    assert any(b.type is BlockType.HEADING for b in doc.blocks)
    # doc_id is the identity key for every store and citation: it must come
    # back exactly as it went in, extension lie and all.
    assert doc.doc_id == "HR/LeavePolicy.docx"


# ------------------------------------------------------ routing table, auto


@pytest.mark.parametrize(
    "name,expected",
    [
        ("sample.md", PlainParser),
        ("sample.txt", PlainParser),
        ("sample.html", PlainParser),
        ("sample.csv", PlainParser),
        ("sample.tsv", PlainParser),
        ("sample.json", PlainParser),
        ("sample.pptx", PptxParser),
        ("sample.png", AzureDocIntParser),
        ("scanned.pdf", AzureDocIntParser),
        ("mystery.xyz", AzureDocIntParser),
    ],
)
def test_auto_routes_each_format_to_its_parser(name, expected):
    parser = select_parser_for(f"HR/{name}", read(name))
    assert isinstance(parser, expected)


@pytest.mark.parametrize(
    "path", ["HR/LeavePolicy.pdf", "IT/PasswordPolicy.docx", "sales/Discounts.xlsx"]
)
def test_auto_keeps_the_local_parser_for_born_digital_office_formats(path):
    parser = select_parser_for(path, (CORPUS / path).read_bytes())
    assert isinstance(parser, LocalParser)


def test_extensionless_html_still_routes_by_content():
    parser = select_parser_for("IT/handbook", read("sample.html"))
    assert isinstance(parser, PlainParser)


# ----------------------------------------------------------- hard overrides


def test_azure_override_sends_everything_to_document_intelligence(monkeypatch):
    monkeypatch.setenv("DOC_PARSER", "azure")
    assert isinstance(select_parser_for("HR/sample.md", read("sample.md")),
                      AzureDocIntParser)
    assert isinstance(
        select_parser_for("HR/LeavePolicy.pdf",
                          (CORPUS / "HR/LeavePolicy.pdf").read_bytes()),
        AzureDocIntParser,
    )


def test_local_override_still_uses_the_local_only_parsers(monkeypatch):
    """"local" means "no Document Intelligence", not "only LocalParser"."""
    monkeypatch.setenv("DOC_PARSER", "local")
    assert isinstance(select_parser_for("HR/sample.md", read("sample.md")),
                      PlainParser)
    assert isinstance(select_parser_for("sales/sample.pptx", read("sample.pptx")),
                      PptxParser)
    assert isinstance(
        select_parser_for("HR/LeavePolicy.pdf",
                          (CORPUS / "HR/LeavePolicy.pdf").read_bytes()),
        LocalParser,
    )


def test_local_override_refuses_an_image_with_a_typed_error(monkeypatch):
    monkeypatch.setenv("DOC_PARSER", "local")
    with pytest.raises(UnsupportedFormatError):
        select_parser_for("HR/sample.png", read("sample.png"))


def test_local_override_refuses_a_scanned_pdf_rather_than_emitting_nothing(monkeypatch):
    monkeypatch.setenv("DOC_PARSER", "local")
    with pytest.raises(ScannedDocumentError):
        select_parser_for("HR/scanned.pdf", read("scanned.pdf"))


def test_local_override_refuses_an_unknown_format(monkeypatch):
    monkeypatch.setenv("DOC_PARSER", "local")
    with pytest.raises(UnsupportedFormatError):
        select_parser_for("HR/mystery.xyz", read("mystery.xyz"))


def test_every_router_error_is_a_parsing_error_never_a_bare_value_error(monkeypatch):
    monkeypatch.setenv("DOC_PARSER", "local")
    for name in ("sample.png", "mystery.xyz", "scanned.pdf"):
        with pytest.raises(ParsingError):
            select_parser_for(f"HR/{name}", read(name))
    assert issubclass(ScannedDocumentError, UnsupportedFormatError)
    assert issubclass(UnsupportedFormatError, ParsingError)
    assert not issubclass(ParsingError, ValueError)


def test_auto_refuses_the_di_path_when_di_is_not_configured(monkeypatch):
    """Better a typed error naming the missing config than an SDK auth failure."""
    monkeypatch.setenv("DOC_PARSER", "auto")
    monkeypatch.setenv("AZURE_DOCINT_ENDPOINT", "")
    monkeypatch.setenv("AZURE_DOCINT_KEY", "")
    with pytest.raises(UnsupportedFormatError):
        select_parser_for("HR/sample.png", read("sample.png"))


# ------------------------------------------- select_parser() stays the door


async def test_select_parser_in_auto_mode_dispatches_per_document(monkeypatch):
    monkeypatch.setenv("DOC_PARSER", "auto")
    doc = await select_parser().parse(read("sample.md"), "HR/sample.md")
    assert doc.file_format == "md"
    assert any(b.type is BlockType.HEADING and b.level == 1 for b in doc.blocks)


def test_select_parser_azure_override_is_unchanged(monkeypatch):
    monkeypatch.setenv("DOC_PARSER", "azure")
    assert isinstance(select_parser(), AzureDocIntParser)


async def test_select_parser_in_local_mode_routes_but_never_reaches_azure(monkeypatch):
    monkeypatch.setenv("DOC_PARSER", "local")
    doc = await select_parser().parse(read("sample.txt"), "finance/sample.txt")
    assert any(b.type is BlockType.HEADING for b in doc.blocks)
    with pytest.raises(UnsupportedFormatError):
        await select_parser().parse(read("sample.png"), "HR/sample.png")


async def test_auto_dispatch_reaches_document_intelligence_for_a_scanned_pdf(
    monkeypatch,
):
    """Proves the scanned branch is wired without spending a DI call."""
    called: list[str] = []

    async def fake_parse(self, data: bytes, doc_id: str) -> ParsedDocument:
        called.append(doc_id)
        return ParsedDocument(doc_id=doc_id, file_format="pdf", blocks=[])

    monkeypatch.setattr(AzureDocIntParser, "parse", fake_parse)
    monkeypatch.setenv("DOC_PARSER", "auto")
    await select_parser().parse(read("scanned.pdf"), "HR/scanned.pdf")
    assert called == ["HR/scanned.pdf"]


# ------------------------------------------------------------------- live


@pytest.mark.skipif(
    not azure_configured("azure_docint_endpoint", "azure_docint_key"),
    reason="Azure Document Intelligence not configured",
)
async def test_live_scanned_pdf_is_ocred_by_document_intelligence():
    """The whole point of the scanned branch: pixels in, real text out.

    pdfplumber extracts exactly zero characters from this fixture, so any
    text asserted here can only have come from OCR.
    """
    data = read("scanned.pdf")
    assert pdf_char_density(data) == 0.0
    parser = select_parser_for("HR/scanned.pdf", data)
    assert isinstance(parser, AzureDocIntParser)
    doc = await parser.parse(data, "HR/scanned.pdf")
    text = " ".join(b.text for b in doc.blocks).lower()
    assert "badge access" in text
    assert "north wing" in text
