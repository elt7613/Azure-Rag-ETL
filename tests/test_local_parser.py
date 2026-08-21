import pytest
from rag.models import BlockType
from rag.parsing.local_parser import LocalParser


@pytest.fixture
def parser():
    return LocalParser()


async def test_pdf_recovers_table_grid(parser):
    data = open("source_data/sales/Pricing2026.pdf", "rb").read()
    doc = await parser.parse(data, "sales/Pricing2026.pdf")
    tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
    tier = next(t for t in tables if t.rows[0][0] == "Tier")
    row = {r[0]: r[1] for r in tier.rows[1:]}
    assert row["Starter"] == "$32"
    assert row["Professional"] == "$65"


async def test_docx_emits_headings(parser):
    data = open("source_data/IT/PasswordPolicy.docx", "rb").read()
    doc = await parser.parse(data, "IT/PasswordPolicy.docx")
    headings = [b.text for b in doc.blocks if b.type is BlockType.HEADING]
    assert any("Password Requirements" in h for h in headings)


async def test_header_line_is_captured(parser):
    data = open("source_data/HR/Benefits.pdf", "rb").read()
    doc = await parser.parse(data, "HR/Benefits.pdf")
    assert "Human Resources" in doc.raw_header
