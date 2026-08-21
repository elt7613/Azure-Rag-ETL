"""Version conflict detection and resolution — the 2025/2026 rate card case."""
from __future__ import annotations

from datetime import date

from rag.retrieval import RetrievedChunk
from rag.retrieval.conflict import (
    conflict_note,
    detect_conflicts,
    prefer_current,
    resolve,
)


def pricing(year: int, *, current: bool, superseded_by: str = "",
            cid: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid or f"c{year}",
        doc_id=f"sales/Pricing{year}.pdf",
        title=f"Pricing{year}",
        department="sales",
        section_path="3 Subscription Tiers",
        content=f"Enterprise ${99 if year == 2025 else 109}/seat/month",
        page=1,
        version="1.4" if year == 2025 else "1.0",
        is_current=current,
        superseded_by=superseded_by,
        effective_from=date(year, 1, 1),
        reranker_score=2.5,
    )


def test_explicit_superseded_by_link_is_detected():
    chunks = [
        pricing(2025, current=False, superseded_by="sales/Pricing2026.pdf"),
        pricing(2026, current=True),
    ]
    conflicts = detect_conflicts(chunks)

    assert len(conflicts) == 1
    assert conflicts[0].current_doc_id == "sales/Pricing2026.pdf"
    assert conflicts[0].superseded_doc_ids == ["sales/Pricing2025.pdf"]


def test_unlinked_versions_are_detected_by_title_stem_and_date():
    """The fallback path: no `superseded_by` was ever resolved."""
    chunks = [pricing(2025, current=True), pricing(2026, current=True)]
    conflicts = detect_conflicts(chunks)

    assert len(conflicts) == 1
    assert conflicts[0].current_doc_id == "sales/Pricing2026.pdf"


def test_unrelated_documents_are_not_treated_as_versions():
    leave = RetrievedChunk(
        chunk_id="l", doc_id="HR/LeavePolicy.pdf", title="LeavePolicy",
        department="HR", section_path="2", content="PTO", page=1,
        effective_from=date(2026, 1, 1),
    )
    benefits = RetrievedChunk(
        chunk_id="b", doc_id="HR/Benefits.pdf", title="Benefits",
        department="HR", section_path="1", content="401k", page=1,
        effective_from=date(2025, 1, 1),
    )
    assert detect_conflicts([leave, benefits]) == []


def test_documents_from_different_departments_are_never_versions():
    a = RetrievedChunk(
        chunk_id="a", doc_id="HR/Policy2025.pdf", title="Policy2025",
        department="HR", section_path="1", content="x", page=1,
        effective_from=date(2025, 1, 1),
    )
    b = RetrievedChunk(
        chunk_id="b", doc_id="IT/Policy2026.pdf", title="Policy2026",
        department="IT", section_path="1", content="y", page=1,
        effective_from=date(2026, 1, 1),
    )
    assert detect_conflicts([a, b]) == []


def test_superseded_evidence_is_reordered_not_discarded():
    """The old rate card still governs contracts signed while it was current."""
    old = pricing(2025, current=False, superseded_by="sales/Pricing2026.pdf")
    new = pricing(2026, current=True)

    ordered, conflicts = resolve([old, new])

    assert conflicts
    assert [c.doc_id for c in ordered] == [
        "sales/Pricing2026.pdf", "sales/Pricing2025.pdf",
    ]
    # Nothing was dropped.
    assert len(ordered) == 2


def test_prefer_current_orders_by_currency_then_date():
    old = pricing(2025, current=False)
    new = pricing(2026, current=True)
    assert [c.doc_id for c in prefer_current([old, new])][0] == "sales/Pricing2026.pdf"
    assert [c.doc_id for c in prefer_current([new, old])][0] == "sales/Pricing2026.pdf"


def test_no_conflict_leaves_ordering_untouched():
    chunks = [pricing(2026, current=True, cid="a"), pricing(2026, current=True, cid="b")]
    ordered, conflicts = resolve(chunks)
    assert conflicts == []
    assert [c.chunk_id for c in ordered] == ["a", "b"]


def test_conflict_note_tells_the_model_what_to_disclose():
    conflicts = detect_conflicts([
        pricing(2025, current=False, superseded_by="sales/Pricing2026.pdf"),
        pricing(2026, current=True),
    ])
    note = conflict_note(conflicts)

    assert "sales/Pricing2026.pdf" in note
    assert "sales/Pricing2025.pdf" in note
    assert "state which version" in note


def test_conflict_note_is_empty_when_there_is_no_conflict():
    assert conflict_note([]) == ""
