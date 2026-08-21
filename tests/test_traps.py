import re

from rag.models import BlockType
from rag.parsing.local_parser import LocalParser


async def test_xlsx_yields_values_not_formulas():
    """Trap 1: openpyxl default reads '=$C$4*(1-C10)'; data_only=True gives 55.25."""
    data = open("source_data/sales/Discounts.xlsx", "rb").read()
    doc = await LocalParser().parse(data, "sales/Discounts.xlsx")
    text = "\n".join(b.to_markdown() for b in doc.blocks)
    assert "55.25" in text
    assert "$C$4" not in text
    assert "VLOOKUP" not in text


async def test_pdf_table_not_flattened():
    """Trap 2: pypdf flattens the grid so $32 can bind to the wrong tier."""
    data = open("source_data/sales/Pricing2026.pdf", "rb").read()
    doc = await LocalParser().parse(data, "sales/Pricing2026.pdf")
    md = "\n".join(b.to_markdown() for b in doc.blocks if b.rows)
    assert "| Starter | $32 |" in md


async def test_leave_policy_headings_are_real_sections():
    """Regression for review Finding A.

    pdfplumber's extract_text() re-emits table-cell text as plain lines in
    addition to the gridded extract_tables() output. A row like
    "0 - 2 years 15 days 5 days 22.5 days" starts with a digit + whitespace
    and used to false-match the heading regex, becoming a bogus level-1
    HEADING interleaved with the genuine 2.1/2.2 section headings.
    """
    data = open("source_data/HR/LeavePolicy.pdf", "rb").read()
    doc = await LocalParser().parse(data, "HR/LeavePolicy.pdf")
    headings = [b.text for b in doc.blocks if b.type is BlockType.HEADING]

    # No heading should be a table row like "0 - 2 years ...".
    assert not any(re.match(r"^\d+\s*[–-]\s", h) for h in headings)

    # The real section/subsection numbers, in order, and nothing else.
    assert headings == [
        "1 Purpose & Scope",
        "2 Types of Leave",
        "2.1 Annual / Paid Time Off (PTO)",
        "2.2 Sick Leave",
        "2.3 Parental Leave",
        "2.4 Bereavement Leave",
        "2.5 Jury Duty & Civic Leave",
        "2.6 Company-Observed Holidays",
        "3 Requesting Leave",
        "4 Unplanned Absences",
        "5 Leave Without Pay",
        "6 Questions",
    ]


async def test_pdf_table_content_not_duplicated_as_paragraphs():
    """Regression for review Finding D (same root cause as Finding A).

    Every table's cell text used to also appear a second time as flat
    PARAGRAPH/HEADING blocks from extract_text(), doubling what the chunker
    and embedder would see. Table cell content must appear exactly once, as
    part of a TABLE block.
    """
    for path, doc_id in [
        ("source_data/HR/LeavePolicy.pdf", "HR/LeavePolicy.pdf"),
        ("source_data/sales/Pricing2026.pdf", "sales/Pricing2026.pdf"),
        ("source_data/finance/ExpensePolicy.pdf", "finance/ExpensePolicy.pdf"),
    ]:
        data = open(path, "rb").read()
        doc = await LocalParser().parse(data, doc_id)
        tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
        prose = [b.text for b in doc.blocks if b.type is not BlockType.TABLE]
        for table in tables:
            header_row = " ".join(c for c in table.rows[0] if c)
            assert not any(header_row in p for p in prose), (
                f"{doc_id}: table header row leaked into prose: {header_row!r}"
            )


async def test_xlsx_metadata_line_not_duplicated_across_sheets():
    """Regression for review Finding B (part 1).

    The "Sales Operations | Effective: ... | Version ..." metadata line
    repeats at the top of every sheet in Discounts.xlsx. The header-capture
    used to be a workbook-level one-shot flag, so only sheet 1's copy was
    stripped -- the same line leaked in as a bogus data row on every other
    sheet's table.
    """
    data = open("source_data/sales/Discounts.xlsx", "rb").read()
    doc = await LocalParser().parse(data, "sales/Discounts.xlsx")
    for table in (b for b in doc.blocks if b.type is BlockType.TABLE):
        for row in table.rows:
            for cell in row:
                # No legitimate data cell in this workbook contains a pipe
                # or the "Effective: <date> | Version" marker; only the
                # metadata line does.
                assert "|" not in cell
                assert "Effective: January 1, 2026" not in cell


async def test_xlsx_table_header_row_is_column_names():
    """Regression for review Finding B (part 2).

    No xlsx table has a real header row at rows[0] naturally -- the sheet
    title text occupied that slot -- so Block.to_markdown() used to render
    the sheet title as the table header, burying the real column names.
    """
    data = open("source_data/sales/Discounts.xlsx", "rb").read()
    doc = await LocalParser().parse(data, "sales/Discounts.xlsx")
    tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
    seat_tier_table = next(t for t in tables if t.rows[0][0] == "Seat Count Tier")
    assert seat_tier_table.rows[0] == [
        "Seat Count Tier", "Min Seats", "Discount %", "Discounted Price ($/seat/mo)",
    ]
    term_table = next(t for t in tables if t.rows[0][0] == "Billing Term")
    assert term_table.rows[0] == ["Billing Term", "Discount %", "Illustrative Price ($/seat/mo)"]


async def test_pdf_header_not_truncated_by_line_wrap():
    """Regression for review Finding C.

    VendorContract.pdf's metadata line wraps mid-sentence onto the next
    physical PDF line ("... not a binding" / "agreement"). The header
    heuristic used to capture only the first line, truncating the header.
    """
    data = open("source_data/legal/VendorContract.pdf", "rb").read()
    doc = await LocalParser().parse(data, "legal/VendorContract.pdf")
    assert "not a binding agreement" in doc.raw_header
    assert "Legal Department" in doc.raw_header


async def test_discount_row_chunk_is_retrievable_as_value():
    """Trap 1 end-to-end: the 100-249 seat row reaches a chunk as 55.25."""
    from rag.enrich.chunker import chunk_document
    from rag.enrich.metadata import extract_metadata
    from rag.parsing.local_parser import LocalParser

    data = open("source_data/sales/Discounts.xlsx", "rb").read()
    doc = await LocalParser().parse(data, "sales/Discounts.xlsx")
    chunks = chunk_document(doc, extract_metadata(doc, "sales/Discounts.xlsx"))
    assert any("55.25" in c.display_text for c in chunks)
    assert not any("$C$4" in c.display_text for c in chunks)


async def test_pricing_table_follows_its_heading_not_preamble():
    """Regression for review Finding E.

    _parse_pdf used to emit every page's tables before any of its text, so
    a table always preceded the heading it visually sits under. Downstream,
    build_section_tree files anything before the first heading under a
    synthetic "Preamble" section -- so the tier pricing table (and every
    other table in the corpus) lost its real section and retrieved with
    breadcrumb ".../Preamble" instead of ".../3 Subscription Tiers".
    """
    from rag.enrich.chunker import chunk_document
    from rag.enrich.metadata import extract_metadata

    data = open("source_data/sales/Pricing2026.pdf", "rb").read()
    doc = await LocalParser().parse(data, "sales/Pricing2026.pdf")

    heading_idx = next(
        i for i, b in enumerate(doc.blocks)
        if b.type is BlockType.HEADING and b.text == "3 Subscription Tiers"
    )
    table_idx = next(
        i for i, b in enumerate(doc.blocks)
        if b.type is BlockType.TABLE and b.rows[0][0] == "Tier"
    )
    assert table_idx > heading_idx, (
        "the tier table must appear after its '3 Subscription Tiers' heading, "
        f"not before it (heading at {heading_idx}, table at {table_idx})"
    )

    chunks = chunk_document(doc, extract_metadata(doc, "sales/Pricing2026.pdf"))
    tier_chunk = next(c for c in chunks if c.content_type == "table"
                       and "Starter" in c.display_text)
    assert tier_chunk.section_path != ["Preamble"]


async def test_no_cid_glyph_artifacts_in_any_document():
    """Regression for review Finding F.

    pdfplumber sometimes fails to decode a bullet glyph and emits the raw
    PDF font code instead, e.g. "(cid:127) List prices increased ...". Left
    in place this lands in both display_text (shown to users in citations)
    and embed_text (pollutes the vector). No block's text may contain a
    "(cid:N)" artifact, across every document in the corpus.
    """
    import glob

    cid_re = re.compile(r"cid:\d+")
    for path in sorted(glob.glob("source_data/**/*.*", recursive=True)):
        doc_id = path[len("source_data/"):]
        data = open(path, "rb").read()
        doc = await LocalParser().parse(data, doc_id)
        for b in doc.blocks:
            assert not cid_re.search(b.text), f"{doc_id}: cid artifact in {b.text!r}"
            for row in (b.rows or []):
                for cell in row:
                    assert not cid_re.search(cell), f"{doc_id}: cid artifact in table cell {cell!r}"
        assert not cid_re.search(doc.raw_header), f"{doc_id}: cid artifact in raw_header"
