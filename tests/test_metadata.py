from datetime import date

from rag.enrich.metadata import extract_metadata, link_versions
from rag.models import DocumentMetadata, ParsedDocument


def _doc(header: str, key: str) -> ParsedDocument:
    return ParsedDocument(doc_id=key, file_format="pdf", blocks=[], raw_header=header)


def test_parses_effective_version_and_supersedes():
    header = ("Northwind Traders, Inc. | Sales Operations | Effective: January 1, 2026 "
              "| Version 1.0 | Supersedes: 2025 Rate Card (v1.4)")
    m = extract_metadata(_doc(header, "sales/Pricing2026.pdf"), "sales/Pricing2026.pdf")
    assert m.department == "sales"
    assert m.version == "1.0"
    assert m.effective_from == date(2026, 1, 1)
    assert "2025 Rate Card" in m.supersedes


def test_parses_effective_range():
    header = ("Northwind Traders, Inc. | Sales Operations | Effective: January 1, 2025 "
              "– December 31, 2025 | Version 1.4")
    m = extract_metadata(_doc(header, "sales/Pricing2025.pdf"), "sales/Pricing2025.pdf")
    assert m.effective_from == date(2025, 1, 1)
    assert m.effective_to == date(2025, 12, 31)


def test_department_falls_back_to_folder():
    m = extract_metadata(_doc("", "HR/Benefits.pdf"), "HR/Benefits.pdf")
    assert m.department == "HR"


def test_link_versions_marks_older_superseded_but_keeps_it():
    older = DocumentMetadata(doc_id="sales/Pricing2025.pdf", title="2025 Rate Card",
                             department="sales", version="1.4")
    newer = DocumentMetadata(doc_id="sales/Pricing2026.pdf", title="2026 Rate Card",
                             department="sales", version="1.0",
                             supersedes="2025 Rate Card (v1.4)")
    linked = {m.doc_id: m for m in link_versions([older, newer])}
    assert linked["sales/Pricing2025.pdf"].is_current is False
    assert linked["sales/Pricing2025.pdf"].superseded_by == "sales/Pricing2026.pdf"
    assert linked["sales/Pricing2026.pdf"].is_current is True


def test_link_versions_links_real_pricing_pair():
    import asyncio
    from pathlib import Path

    from rag.parsing.local_parser import LocalParser

    root = Path(__file__).resolve().parent.parent / "source_data" / "sales"
    parser = LocalParser()

    async def _parse(name: str):
        data = (root / name).read_bytes()
        return await parser.parse(data, f"sales/{name}")

    doc_2025 = asyncio.run(_parse("Pricing2025.pdf"))
    doc_2026 = asyncio.run(_parse("Pricing2026.pdf"))

    meta_2025 = extract_metadata(doc_2025, "sales/Pricing2025.pdf")
    meta_2026 = extract_metadata(doc_2026, "sales/Pricing2026.pdf")

    linked = {m.doc_id: m for m in link_versions([meta_2025, meta_2026])}
    assert linked["sales/Pricing2025.pdf"].is_current is False
    assert linked["sales/Pricing2025.pdf"].superseded_by == "sales/Pricing2026.pdf"
    assert linked["sales/Pricing2026.pdf"].is_current is True
