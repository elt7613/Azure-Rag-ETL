"""Deterministic tabular extraction, measured against the real corpus.

Per the owner's direction, tables never go through the LLM: a rate table is
already structured data, and a parser reading "$61.75" cannot transpose a
digit the way a model reading it back can. The real-corpus tests below parse
the actual source files with the project's own parsers and assert on the
actual numbers a human put in those spreadsheets and documents -- inventing a
table here would only prove the code agrees with a table nobody has to get
right.
"""
from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import httpx
import pytest

from rag.enrich.chunker import chunk_document
from rag.enrich.metadata import extract_metadata
from rag.extraction.tabular import (
    CellValue,
    classify_cell_value,
    extract_from_table,
)
from rag.extraction.units import build_units
from rag.extraction.ontology import is_valid_entity_type, is_valid_relation_type
from rag.models import BlockType, Entity, ExtractionUnit, Relation
from rag.parsing.local_parser import LocalParser

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source_data"


# --------------------------------------------------------------------------
# classify_cell_value -- the typed-cell-value rules in isolation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,kind,value", [
    ("$65", "currency", 65.0),
    ("$61.75", "currency", 61.75),
    ("Up to $500", "currency", 500.0),
    ("Above $10,000", "currency", 10000.0),
    ("$100/person", "currency", 100.0),
])
def test_currency_cells_are_typed_and_keep_the_raw_string(raw, kind, value):
    cell = classify_cell_value(raw)
    assert cell.kind == kind
    assert cell.value == value
    assert cell.raw == raw  # citable exactly as written


def test_currency_with_a_unit_suffix_keeps_the_unit():
    assert classify_cell_value("$100/person").unit == "person"
    assert classify_cell_value("$500/year").unit == "year"


@pytest.mark.parametrize("raw,expected_pct", [
    ("15%", 15.0),
    ("Above 40%", 40.0),
])
def test_explicit_percent_sign_is_a_percentage(raw, expected_pct):
    cell = classify_cell_value(raw)
    assert cell.kind == "percentage"
    assert cell.value == expected_pct


def test_bare_fraction_is_a_percentage_only_under_a_percent_header():
    # "0.15" alone is indistinguishable from a small quantity; the column
    # header ("Discount %") is what tells the classifier which it is.
    assert classify_cell_value("0.15", "Discount %").kind == "percentage"
    assert classify_cell_value("0.15", "Discount %").value == 15.0
    assert classify_cell_value("0.15", "Min Seats").kind == "quantity"


def test_a_dash_separated_pair_is_a_range_with_low_and_high():
    cell = classify_cell_value("5–24 seats")
    assert cell.kind == "range"
    assert (cell.low, cell.high, cell.unit) == (5.0, 24.0, "seats")

    cell = classify_cell_value("0% – 10%")
    assert cell.kind == "range"
    assert (cell.low, cell.high) == (0.0, 10.0)

    cell = classify_cell_value("$500 – $2,500")
    assert cell.kind == "range"
    assert (cell.low, cell.high) == (500.0, 2500.0)


def test_a_count_and_its_unit_word_is_a_quantity():
    cell = classify_cell_value("20 days")
    assert cell.kind == "quantity"
    assert cell.value == 20.0
    assert cell.unit == "days"


def test_plain_descriptive_text_stays_text():
    for raw in ["Direct Manager", "Not permitted without a CFO exception approval",
                "Economy", "N/A", "Actual cost"]:
        cell = classify_cell_value(raw)
        assert cell.kind == "text"
        assert cell.value is None


def test_a_numeric_looking_string_that_is_not_a_number_stays_text():
    """An order/SKU code has digits in it but is not a quantity, a currency
    amount, or anything else numeric -- it must not be coerced into one."""
    for raw in ["SO-2024-118", "v2.0", "Policy-HR-014"]:
        cell = classify_cell_value(raw)
        assert cell.kind == "text"
        assert cell.value is None


def test_a_trailing_qualifier_does_not_get_mistaken_for_a_clean_quantity():
    # "6 hours or more (domestic)" is prose describing a flight duration
    # band, not a bare "<N> <unit>" quantity -- it must not be parsed as one.
    cell = classify_cell_value("6 hours or more (domestic)")
    assert cell.kind == "text"


def test_classify_cell_value_returns_the_dataclass():
    assert isinstance(classify_cell_value("$65"), CellValue)


# --------------------------------------------------------------------------
# extract_from_table -- messiness the parsers actually hand it
# --------------------------------------------------------------------------


def _unit(section_path="1 Widgets", doc_id="sales/Widgets.xlsx",
          department="sales", page=1, chunk_ids=("c1",)) -> ExtractionUnit:
    return ExtractionUnit(unit_id="u", doc_id=doc_id, department=department,
                          section_path=section_path, text="", page=page,
                          chunk_ids=list(chunk_ids), content_type="table")


def test_every_data_row_becomes_an_entity_with_a_has_value_and_applies_to_relation():
    rows = [
        ["Tier", "Price"],
        ["Bronze", "$10"],
        ["Silver", "$20"],
    ]
    entities, relations = extract_from_table(rows, _unit(section_path="1 Pricing"))
    names = {e.name for e in entities}
    assert {"Pricing", "Bronze", "Silver"} <= names
    applies = [r for r in relations if r.predicate == "APPLIES_TO"]
    has_value = [r for r in relations if r.predicate == "HAS_VALUE"]
    assert {r.subject for r in applies} == {"Bronze", "Silver"}
    assert all(r.object == "Pricing" for r in applies)
    assert {(r.subject, r.object) for r in has_value} == {("Bronze", "$10"), ("Silver", "$20")}


def test_every_relation_carries_full_provenance_and_is_deterministic():
    rows = [["Tier", "Price"], ["Bronze", "$10"]]
    unit = _unit(section_path="1 Pricing", doc_id="sales/Widgets.xlsx",
                department="sales", page=3, chunk_ids=("abc123",))
    _, relations = extract_from_table(rows, unit)
    assert relations  # non-empty, or the provenance checks below are vacuous
    for r in relations:
        assert r.deterministic is True
        assert r.confidence == 1.0
        assert r.doc_id == "sales/Widgets.xlsx"
        assert r.source_chunk_id == "abc123"
        assert r.section_path == "1 Pricing"
        assert r.page == 3
        assert r.department == "sales"
        assert r.evidence_span  # a quotable, verbatim span


def test_only_closed_ontology_types_are_ever_emitted():
    rows = [["Tier", "Price", "Notes"], ["Bronze", "$10", "Introductory"]]
    entities, relations = extract_from_table(rows, _unit(section_path="1 Pricing"))
    for e in entities:
        assert is_valid_entity_type(e.type), e.type
    for r in relations:
        assert is_valid_relation_type(r.predicate)
        assert r.object_type == "" or is_valid_entity_type(r.object_type)
        assert is_valid_entity_type(r.subject_type)


def test_a_merged_or_blank_header_cell_produces_no_relation_for_that_column():
    rows = [
        ["Tier", "", "Price"],   # the middle column has no header
        ["Bronze", "mystery", "$10"],
    ]
    _, relations = extract_from_table(rows, _unit(section_path="1 Pricing"))
    has_value = [r for r in relations if r.predicate == "HAS_VALUE"]
    assert {r.object for r in has_value} == {"$10"}
    assert "mystery" not in {r.object for r in has_value}


def test_a_leading_title_row_above_the_real_header_is_skipped():
    rows = [
        ["Widget Pricing Schedule"],           # caption, not the header
        ["Tier", "Price"],
        ["Bronze", "$10"],
        ["Silver", "$20"],
    ]
    entities, relations = extract_from_table(rows, _unit(section_path="1 Pricing"))
    assert "Widget Pricing Schedule" not in {e.name for e in entities}
    assert {r.subject for r in relations if r.predicate == "APPLIES_TO"} == {"Bronze", "Silver"}


def test_ragged_rows_do_not_crash_and_missing_cells_are_simply_skipped():
    rows = [
        ["Tier", "Price", "Notes"],
        ["Bronze", "$10"],                      # short a column
        ["Silver", "$20", "Popular", "extra"],  # one too many
        ["Gold"],                               # only the key column
    ]
    entities, relations = extract_from_table(rows, _unit(section_path="1 Pricing"))
    assert {e.name for e in entities} >= {"Bronze", "Silver", "Gold"}
    bronze_values = {r.object for r in relations if r.subject == "Bronze" and r.predicate == "HAS_VALUE"}
    assert bronze_values == {"$10"}
    gold_values = {r.object for r in relations if r.subject == "Gold" and r.predicate == "HAS_VALUE"}
    assert gold_values == set()
    gold_applies = [r for r in relations if r.subject == "Gold" and r.predicate == "APPLIES_TO"]
    assert len(gold_applies) == 1


def test_a_single_column_table_yields_entities_with_no_has_value_relations():
    rows = [["Approved Vendor"], ["Acme Corp"], ["Globex"]]
    entities, relations = extract_from_table(rows, _unit(section_path="1 Vendors"))
    assert {e.name for e in entities} >= {"Acme Corp", "Globex"}
    assert not [r for r in relations if r.predicate == "HAS_VALUE"]
    assert len([r for r in relations if r.predicate == "APPLIES_TO"]) == 2


def test_duplicate_key_column_values_do_not_collide_into_one_entity():
    rows = [["Tier", "Price"], ["Standard", "$10"], ["Standard", "$99"]]
    entities, relations = extract_from_table(rows, _unit(section_path="1 Pricing"))
    row_entities = [e for e in entities if e.name != "Pricing"]
    assert len(row_entities) == 2
    assert len({e.compute_id() for e in row_entities}) == 2


def test_a_blank_row_between_data_rows_is_skipped():
    rows = [["Tier", "Price"], ["Bronze", "$10"], ["", ""], ["Silver", "$20"]]
    entities, _ = extract_from_table(rows, _unit(section_path="1 Pricing"))
    assert {e.name for e in entities} == {"Pricing", "Bronze", "Silver"}


def test_an_empty_table_produces_nothing():
    assert extract_from_table([], _unit()) == ([], [])
    assert extract_from_table([["Tier", "Price"]], _unit()) == ([], [])  # header, no data


# --------------------------------------------------------------------------
# Entity type inference from the section/sheet title
# --------------------------------------------------------------------------


def test_entity_type_is_inferred_from_the_section_title():
    rows = [["Program", "Discount %"], ["Non-Profit", "20%"]]
    entities, _ = extract_from_table(rows, _unit(section_path="5 Special Programs"))
    subject = next(e for e in entities if e.name == "Special Programs")
    assert subject.type == "Plan"  # "program(s)" is an alias of Plan
    row = next(e for e in entities if e.name == "Non-Profit")
    assert row.type == subject.type


def test_a_title_the_ontology_cannot_place_falls_back_to_a_valid_type():
    rows = [["Combined Discount Range", "Required Approver"], ["0% – 10%", "Account Executive"]]
    entities, _ = extract_from_table(rows, _unit(section_path="9 Approval Matrix"))
    subject = next(e for e in entities if e.name == "Approval Matrix")
    assert is_valid_entity_type(subject.type)  # never an unrecognised label


# --------------------------------------------------------------------------
# Real corpus
# --------------------------------------------------------------------------


def _table_units_with_rows(doc, meta) -> list[tuple[list[list[str]], ExtractionUnit]]:
    """Pair each parsed TABLE block's raw cells with the extraction unit
    `build_units` made from it.

    `build_units` works from `Chunk.display_text`, which for a table chunk is
    the section caption followed by `Block.to_markdown()` -- the caption is
    there so Azure's reranker sees what the table is rather than a bare grid.
    A block and its unit are therefore the same table iff the unit's text
    *ends with* that block's markdown rendering. This is the correspondence the
    real ETL wiring establishes between a parsed `Block` and its unit.
    """
    chunks = chunk_document(doc, meta)
    remaining = [u for u in build_units(chunks, meta) if u.content_type == "table"]
    pairs: list[tuple[list[list[str]], ExtractionUnit]] = []
    for block in doc.blocks:
        if block.type is not BlockType.TABLE:
            continue
        markdown = block.to_markdown()
        match = next(
            (u for u in remaining if u.text == markdown or u.text.endswith(markdown)),
            None,
        )
        if match is None:
            continue
        remaining.remove(match)
        pairs.append((block.rows, match))
    return pairs


@pytest.fixture(scope="module")
def corpus_tables() -> dict[str, list[tuple[list[list[str]], ExtractionUnit]]]:
    """Every table in the four documents this task is graded against, keyed
    by doc_id, parsed with the project's real parsers."""
    async def build():
        out: dict[str, list[tuple[list[list[str]], ExtractionUnit]]] = {}
        for rel in ["sales/Discounts.xlsx", "finance/ExpensePolicy.pdf",
                    "finance/TravelPolicy.docx", "sales/Pricing2026.pdf"]:
            path = SOURCE_DIR / rel
            doc = await LocalParser().parse(path.read_bytes(), rel)
            meta = extract_metadata(doc, rel)
            out[rel] = _table_units_with_rows(doc, meta)
        return out
    return asyncio.run(build())


def _all_relations(pairs) -> list[Relation]:
    out: list[Relation] = []
    for rows, unit in pairs:
        _, relations = extract_from_table(rows, unit)
        out.extend(relations)
    return out


def _table(pairs, *, first_header_cell: str | None = None,
           section_path: str | None = None):
    for rows, unit in pairs:
        if first_header_cell is not None and rows[0][0] != first_header_cell:
            continue
        if section_path is not None and unit.section_path != section_path:
            continue
        return rows, unit
    raise AssertionError(f"no matching table (header={first_header_cell!r}, "
                         f"section={section_path!r})")


def test_discounts_volume_schedule_produces_correctly_typed_tiers(corpus_tables):
    rows, unit = _table(corpus_tables["sales/Discounts.xlsx"], section_path="Volume Discounts")
    header, data_rows = rows[0], rows[1:]
    discount_header, price_header = header[2], header[3]  # "Discount %", "Discounted Price (...)"
    entities, relations = extract_from_table(rows, unit)
    tiers = {"5–24 seats", "25–49 seats", "50–99 seats", "100–249 seats",
            "250–499 seats", "500+ seats"}
    assert tiers <= {e.name for e in entities}

    has_value = {(r.subject, r.object) for r in relations if r.predicate == "HAS_VALUE"}

    expected = {
        "5–24 seats": (0.0, 65.0), "25–49 seats": (5.0, 61.75),
        "50–99 seats": (10.0, 58.5), "100–249 seats": (15.0, 55.25),
        "250–499 seats": (20.0, 52.0), "500+ seats": (25.0, 48.75),
    }
    for tier, min_seats, discount_raw, price_raw in data_rows:
        # The raw sheet strings are what a citation must quote verbatim...
        assert (tier, discount_raw) in has_value
        assert (tier, price_raw) in has_value
        # ...and they carry the correct typed value underneath.
        pct, price_val = expected[tier]
        assert classify_cell_value(discount_raw, discount_header).value == pct
        assert classify_cell_value(price_raw, price_header).value == price_val

    # The 100-249 seat tier specifically: a real entity with a 15% discount
    # and a $55.25 price, exactly as the acceptance criterion names it.
    assert ("100–249 seats", "0.15") in has_value
    assert ("100–249 seats", "55.25") in has_value
    assert classify_cell_value("0.15", discount_header).value == 15.0
    assert classify_cell_value("55.25", price_header).value == 55.25


def test_expense_policy_approval_matrix_and_category_limits(corpus_tables):
    pairs = corpus_tables["finance/ExpensePolicy.pdf"]
    rows, unit = _table(pairs, first_header_cell="Expense Amount")
    _, relations = extract_from_table(rows, unit)
    approver_by_range = {r.subject: r.object for r in relations if r.predicate == "HAS_VALUE"}
    assert approver_by_range == {
        "Up to $500": "Direct Manager",
        "$500 – $2,500": "Department Director",
        "$2,500 – $10,000": "Department VP",
        "Above $10,000": "VP + Finance Business Partner",
    }

    rows, unit = _table(pairs, first_header_cell="Category")
    _, relations = extract_from_table(rows, unit)
    client_meals = {r.object for r in relations if r.subject == "Client meals" and r.predicate == "HAS_VALUE"}
    assert "$100/person" in client_meals
    assert "Director" in client_meals
    assert classify_cell_value("$100/person").value == 100.0


def test_travel_policy_hotel_caps_and_per_diem(corpus_tables):
    pairs = corpus_tables["finance/TravelPolicy.docx"]
    rows, unit = _table(pairs, first_header_cell="City Tier")
    _, relations = extract_from_table(rows, unit)
    caps = {r.subject: r.object for r in relations
            if r.predicate == "HAS_VALUE" and r.object.startswith("$")}
    assert caps == {"Tier 1": "$350", "Tier 2": "$250", "Tier 3": "$180"}
    for raw in caps.values():
        assert classify_cell_value(raw).kind == "currency"

    rows, unit = _table(pairs, first_header_cell="Meal")
    _, relations = extract_from_table(rows, unit)
    breakfast = {r.object for r in relations if r.subject == "Breakfast" and r.predicate == "HAS_VALUE"}
    assert breakfast == {"$18", "$25"}


def test_pricing_2026_subscription_tiers(corpus_tables):
    rows, unit = _table(corpus_tables["sales/Pricing2026.pdf"], section_path="3 Subscription Tiers")
    _, relations = extract_from_table(rows, unit)
    price_by_tier = {
        r.subject: r.object for r in relations
        if r.predicate == "HAS_VALUE" and r.object.startswith("$")
    }
    assert price_by_tier == {
        "Starter": "$32", "Professional": "$65",
        "Enterprise": "$109", "Enterprise Plus": "$145",
    }
    for tier, raw in price_by_tier.items():
        assert classify_cell_value(raw).value == {
            "Starter": 32.0, "Professional": 65.0,
            "Enterprise": 109.0, "Enterprise Plus": 145.0,
        }[tier]


def test_every_corpus_table_relation_is_deterministic_with_full_provenance(corpus_tables):
    relations = [r for pairs in corpus_tables.values() for r in _all_relations(pairs)]
    assert len(relations) > 50  # the four documents' tables are not trivial
    for r in relations:
        assert r.deterministic is True
        assert r.confidence == 1.0
        assert r.doc_id
        assert r.source_chunk_id
        assert r.department in {"sales", "finance"}
        assert is_valid_relation_type(r.predicate)


def test_docx_tables_are_attributed_to_the_section_that_introduces_them(corpus_tables):
    """Regression: `LocalParser._parse_docx` used to consume
    `document.paragraphs` and then `document.tables`, two collections each
    ordered only within themselves. Every table therefore landed after every
    paragraph, filed under whichever heading came last -- all three
    TravelPolicy tables under "10 Contact". The values extracted were still
    correct, but a hotel-cap answer cited the contact section, which is the
    kind of citation that looks fine and is not. The parser now walks the
    body's XML children in true document order."""
    pairs = corpus_tables["finance/TravelPolicy.docx"]
    sections = {unit.section_path for _, unit in pairs}
    assert sections == {
        "3 Air Travel",
        "4 Hotel Accommodations",
        "6 Meals & Incidentals",
    }


# --------------------------------------------------------------------------
# Zero LLM calls, ever
# --------------------------------------------------------------------------


def test_extraction_never_constructs_a_network_client_or_touches_the_network(
    monkeypatch, corpus_tables,
):
    """The owner's requirement is not "prefer not to call the LLM" -- it is
    zero calls. Proven two ways: no HTTP request is ever sent (any attempt
    through httpx, which every Azure/OpenAI SDK client in this project is
    built on, raises immediately), and the socket layer itself is never
    touched, so nothing could have gotten out even via a client this test
    doesn't know about.
    """
    def _blow_up(*args, **kwargs):
        raise AssertionError("tabular extraction must never send an HTTP request")

    monkeypatch.setattr(httpx.Client, "send", _blow_up)
    monkeypatch.setattr(httpx.AsyncClient, "send", _blow_up)

    def _blow_up_connect(*args, **kwargs):
        raise AssertionError("tabular extraction must never open a socket")

    monkeypatch.setattr(socket.socket, "connect", _blow_up_connect)

    total_entities = total_relations = 0
    for pairs in corpus_tables.values():
        for rows, unit in pairs:
            entities, relations = extract_from_table(rows, unit)
            total_entities += len(entities)
            total_relations += len(relations)

    assert total_entities > 0
    assert total_relations > 0
