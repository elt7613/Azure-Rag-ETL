"""The triage gate, measured against the real corpus rather than invented text.

Every assertion below about "keeps this / skips that" names an actual section
of an actual file in `source_data/`. Inventing strings here would only prove
that the scorer matches the strings I imagined; the question that matters is
whether it separates obligation-bearing prose from document furniture in
prose a human actually wrote.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rag.config import get_settings
from rag.enrich.metadata import extract_metadata
from rag.extraction.triage import (
    BoilerplateIndex,
    SkipReason,
    TriageDecision,
    count_acronyms,
    count_dates,
    count_deontic_verbs,
    count_durations,
    count_money,
    count_percentages,
    count_proper_noun_phrases,
    signal_components,
    signal_score,
    triage_unit,
    triage_units,
)
from rag.extraction.units import build_units
from rag.models import ExtractionUnit
from rag.parsing.local_parser import LocalParser
from rag.enrich.chunker import chunk_document

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source_data"


# ---- individual scorer components (pure functions) ----

def test_counts_capitalized_multi_word_phrases_not_sentence_starts():
    assert count_proper_noun_phrases("Meridian Health Group administers the plans.") == 1
    assert count_proper_noun_phrases("Employees may claim reimbursement.") == 0


def test_counts_acronyms():
    assert count_acronyms("MFA is required for all Company accounts.") == 1
    assert count_acronyms("The CISO and the IT Service Desk sign off.") == 2


@pytest.mark.parametrize("text,expected", [
    ("Employees may claim up to $600 per year.", 1),
    ("Client meals $100/person and client gifts $75/recipient.", 2),
    ("No amounts here.", 0),
    ("insurance of at least $1,000,000 per occurrence", 1),
])
def test_counts_currency_amounts(text, expected):
    assert count_money(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("The Company matches 100% of contributions up to 4% of pay.", 2),
    ("Annual prepaid contracts receive a 15% discount.", 1),
    ("interest at 1.0% per month", 1),
    ("No percentages.", 0),
])
def test_counts_percentages(text, expected):
    assert count_percentages(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("effective for contracts signed on or after January 1, 2026.", 1),
    ("credited in full on January 1", 1),
    ("No dates at all.", 0),
])
def test_counts_dates(text, expected):
    assert count_dates(text) >= expected


@pytest.mark.parametrize("text", [
    "12 weeks of paid parental leave",
    "within 30 calendar days of the purchase date",
    "rotate credentials every 90 days",
    "an initial term of twelve (12) months",
])
def test_counts_durations(text):
    assert count_durations(text) >= 1


def test_durations_are_not_found_in_prose_without_them():
    assert count_durations("Expenses must be reasonable and necessary.") == 0


@pytest.mark.parametrize("text", [
    "All expenses must be submitted through ExpensePath.",
    "Recipient shall promptly return all materials.",
    "Employees may claim up to $600 per year.",
    "Eligible employees are entitled to 12 weeks.",
    "Bookings above the cap require pre-approval.",
    "Employees are eligible for tuition reimbursement.",
])
def test_counts_deontic_verbs(text):
    assert count_deontic_verbs(text) >= 1


def test_deontic_count_is_zero_for_descriptive_prose():
    assert count_deontic_verbs("This guide covers installation and troubleshooting.") == 0


def test_signal_score_is_density_not_raw_count():
    dense = "Employees must submit receipts within 30 days for expenses over $25."
    padded = dense + " " + " ".join(["filler"] * 500)
    assert signal_score(dense) > signal_score(padded)


def test_signal_components_are_exposed_for_inspection():
    counts = signal_components("Employees must claim up to $600 per year by January 1.")
    assert counts.deontic >= 1
    assert counts.money == 1
    assert counts.words > 0
    assert counts.weighted > 0


def test_page_furniture_is_not_mistaken_for_signal():
    """pdfplumber leaks running headers/footers into the body; a footer full of
    capitalized words must not read as entity density."""
    footer = "Northwind Traders, Inc. — Internal Use Only Page 2"
    assert signal_score(footer) == 0.0


# ---- the gate itself ----

def _unit(text, content_type="prose", doc_id="HR/LeavePolicy.pdf") -> ExtractionUnit:
    return ExtractionUnit(unit_id="u", doc_id=doc_id, department="HR",
                          section_path="2.3 Parental Leave", text=text, page=1,
                          chunk_ids=["c1"], content_type=content_type)


def test_decision_is_typed_not_a_bare_bool():
    decision = triage_unit(_unit("| A |\n| --- |\n| 1 |", content_type="table"))
    assert isinstance(decision, TriageDecision)
    assert decision.extract is False
    assert decision.reason is SkipReason.TABLE


def test_tables_are_skipped_for_the_deterministic_path():
    rows = "\n".join(f"| Tier {i} | ${i}00 | {i}% |" for i in range(50))
    decision = triage_unit(_unit(rows, content_type="table"))
    assert decision.reason is SkipReason.TABLE


def test_short_units_are_skipped_before_anything_else_is_measured():
    decision = triage_unit(_unit("Employees must submit receipts."))
    assert decision.extract is False
    assert decision.reason is SkipReason.TOO_SHORT
    assert decision.tokens < get_settings().graph_extract_min_tokens


def test_low_signal_prose_above_the_token_floor_is_skipped():
    filler = ("This guide covers installation and troubleshooting of the client "
              "and describes what you will see on screen when you open it. " * 4)
    decision = triage_unit(_unit(filler))
    assert decision.tokens >= get_settings().graph_extract_min_tokens
    assert decision.reason is SkipReason.LOW_SIGNAL


def test_boilerplate_is_skipped_once_it_is_seen_in_enough_documents():
    text = ("Northwind Traders, Inc. considers this document Internal Use Only. "
            "Employees must not distribute it externally without written approval "
            "from the Chief Information Security Officer within 30 days. " * 2)
    threshold = get_settings().boilerplate_doc_threshold
    units = [_unit(text, doc_id=f"HR/Doc{i}.pdf") for i in range(threshold)]
    index = BoilerplateIndex()
    for u in units:
        index.observe(u)
    # It clears every other gate; only its ubiquity disqualifies it.
    assert triage_unit(units[0]).extract is True
    assert triage_unit(units[0], boilerplate=index).reason is SkipReason.BOILERPLATE


def test_boilerplate_needs_distinct_documents_not_repeats_in_one():
    text = "Northwind Traders, Inc. — Confidential. " * 20
    index = BoilerplateIndex()
    for _ in range(20):
        index.observe(_unit(text, doc_id="HR/Only.pdf"))
    assert index.document_count(_unit(text).content_hash()) == 1
    assert index.is_boilerplate(_unit(text).content_hash()) is False


# ---- real corpus ----

@pytest.fixture(scope="module")
def corpus_units() -> dict[str, ExtractionUnit]:
    """Every section of all 11 real source documents, keyed doc_id::section_path."""
    async def build() -> dict[str, ExtractionUnit]:
        out: dict[str, ExtractionUnit] = {}
        for path in sorted(SOURCE_DIR.rglob("*")):
            if not path.is_file():
                continue
            doc_id = path.relative_to(SOURCE_DIR).as_posix()
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                # Other suites drop probe documents into source_data/ and
                # delete them again; a corpus directory under live monitoring
                # can change under a scan, and neither should fail this one.
                continue
            parsed = await LocalParser().parse(data, doc_id)
            meta = extract_metadata(parsed, doc_id)
            for unit in build_units(chunk_document(parsed, meta), meta):
                out[f"{unit.doc_id}::{unit.section_path}"] = unit
        return out

    units = asyncio.run(build())
    assert len(units) > 60, f"corpus fixture looks wrong: {len(units)} units"
    return units


OBLIGATION_BEARING = [
    "HR/LeavePolicy.pdf::2 Types of Leave > 2.3 Parental Leave",
    "HR/LeavePolicy.pdf::3 Requesting Leave",
    "HR/LeavePolicy.pdf::4 Unplanned Absences",
    "HR/Benefits.pdf::5 Wellness Program",
    "HR/Benefits.pdf::6 Tuition Reimbursement",
    "HR/Benefits.pdf::8 Enrollment",
    "finance/ExpensePolicy.pdf::4 Submission Process",
    "finance/ExpensePolicy.pdf::5 Receipt Requirements",
    "finance/TravelPolicy.docx::2 Booking Travel",
    "finance/TravelPolicy.docx::7 International Travel",
    "IT/PasswordPolicy.docx::3 Password Rotation",
    "IT/PasswordPolicy.docx::4 Multi-Factor Authentication (MFA)",
    # Prohibitions written as bare gerund bullets -- the rule is in the heading.
    "IT/PasswordPolicy.docx::7 Prohibited Practices",
    "IT/PasswordPolicy.docx::8 Reporting a Compromised Password",
    "legal/VendorContract.pdf::2 Term & Termination",
    "legal/VendorContract.pdf::3 Payment Terms",
    "legal/VendorContract.pdf::9 Insurance",
    # Contract register: one modal governing eighty words of sub-clauses.
    "legal/NDA.docx::2 Obligations of Receiving Party",
    "legal/NDA.docx::3 Exclusions",
    "legal/NDA.docx::4 Term",
    "sales/Pricing2026.pdf::6 Billing Terms",
]

DOCUMENT_FURNITURE = [
    "HR/Benefits.pdf::9 Contact",
    "HR/LeavePolicy.pdf::6 Questions",
    "finance/ExpensePolicy.pdf::9 Contact",
    "IT/VPNGuide.pdf::6 Troubleshooting",   # the leaked page footer, alone
]


@pytest.mark.parametrize("key", OBLIGATION_BEARING)
def test_triage_keeps_obligation_bearing_corpus_prose(corpus_units, key):
    unit = corpus_units[key]
    decision = triage_unit(unit)
    assert decision.extract is True, (
        f"{key} was dropped as {decision.reason} "
        f"(tokens={decision.tokens}, signal={decision.signal:.4f})\n{unit.text}"
    )


@pytest.mark.parametrize("key", DOCUMENT_FURNITURE)
def test_triage_skips_corpus_document_furniture(corpus_units, key):
    decision = triage_unit(corpus_units[key])
    assert decision.extract is False, (
        f"{key} was kept (tokens={decision.tokens}, signal={decision.signal:.4f})"
    )


def test_the_token_floor_is_set_where_a_real_short_rule_survives(corpus_units):
    """Measured, not assumed.

    `5 Account Lockout Policy` states a genuine rule -- lock after 5 failed
    attempts, for 30 minutes -- in 38 tokens. At the original floor of 40 it
    was dropped, which is why the floor was lowered to 30: the recall cost was
    real and this section is exactly the kind of terse, high-value policy
    statement the graph exists to capture. This test pins the trade in both
    directions so neither the floor nor the section can drift silently.
    """
    unit = corpus_units["IT/PasswordPolicy.docx::5 Account Lockout Policy"]
    decision = triage_unit(unit)
    assert decision.tokens == 38
    assert decision.extract is True, "the configured floor must keep this rule"

    # Raise the floor back to where it started and the same unit is lost.
    strict = get_settings().model_copy(update={"graph_extract_min_tokens": 40})
    dropped = triage_unit(unit, settings=strict)
    assert dropped.extract is False
    assert dropped.reason is SkipReason.TOO_SHORT


def test_every_corpus_table_goes_to_the_deterministic_path(corpus_units):
    tables = [u for u in corpus_units.values() if u.content_type == "table"]
    assert tables
    for unit in tables:
        assert triage_unit(unit).reason is SkipReason.TABLE


def test_triage_over_the_whole_corpus_saves_calls_without_losing_substance(corpus_units):
    decisions = triage_units(list(corpus_units.values()))
    kept = [d for d in decisions if d.extract]
    assert 0 < len(kept) < len(decisions), "triage must be a filter, not a pass-through"
    # Tables and furniture alone are a third of this corpus.
    assert len(kept) / len(decisions) < 0.75
    for decision in kept:
        assert decision.signal >= get_settings().graph_extract_min_signal
        assert decision.tokens >= get_settings().graph_extract_min_tokens


def test_triage_units_returns_one_decision_per_unit_in_order(corpus_units):
    units = list(corpus_units.values())
    decisions = triage_units(units)
    assert [d.unit_id for d in decisions] == [u.unit_id for u in units]
