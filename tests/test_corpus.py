"""The document corpus that ships with the project must actually work.

`source_data/` is what someone gets when they clone this, so it is the first
thing they will run the pipeline over, and every measured figure in the docs
comes from it. These tests are the contract for that: the files are all there,
the departments are the ones the default configuration monitors, and the
formats span more than one parsing path.

What each document *contains* is asserted by the parsing suites --
`test_local_parser`, `test_page_furniture`, `test_traps`, `test_metadata`.
This file only guards the shape of the corpus itself. No network.
"""
from __future__ import annotations

from tests.conftest import CORPUS

from rag.config import get_settings

EXPECTED = {
    "HR/Benefits.pdf",
    "HR/LeavePolicy.pdf",
    "finance/ExpensePolicy.pdf",
    "finance/TravelPolicy.docx",
    "IT/PasswordPolicy.docx",
    "IT/VPNGuide.pdf",
    "legal/NDA.docx",
    "legal/VendorContract.pdf",
    "sales/Discounts.xlsx",
    "sales/Pricing2025.pdf",
    "sales/Pricing2026.pdf",
}


def documents() -> list[str]:
    return sorted(
        p.relative_to(CORPUS).as_posix()
        for p in CORPUS.rglob("*")
        if p.is_file()
    )


def test_the_corpus_is_complete():
    assert set(documents()) == EXPECTED


def test_it_spans_five_departments_and_three_formats():
    """Enough variety that a first run exercises more than one code path."""
    docs = documents()
    assert {d.split("/")[0] for d in docs} == {"HR", "finance", "IT", "legal", "sales"}
    assert {d.rsplit(".", 1)[1] for d in docs} == {"pdf", "docx", "xlsx"}


def test_every_folder_is_a_configured_department():
    """A folder no department claims is silently invisible: the ETL skips it,
    so the documents index nowhere and no query can ever reach them."""
    configured = {d.lower() for d in get_settings().departments}
    folders = {d.split("/")[0].lower() for d in documents()}
    assert folders <= configured, f"not in DEPARTMENTS: {sorted(folders - configured)}"


def test_the_pricing_pair_is_present_for_version_resolution():
    """Superseded-version handling needs two documents that actually conflict;
    without both, the tests that cover it pass by retrieving nothing."""
    docs = set(documents())
    assert {"sales/Pricing2025.pdf", "sales/Pricing2026.pdf"} <= docs
