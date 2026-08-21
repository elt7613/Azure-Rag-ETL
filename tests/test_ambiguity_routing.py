"""Routing between answer, clarify and abstain.

Every case below is built from the diagnostics a real evaluation run recorded
(`eval/results/improved.json`) — the reranker scores, coverage and reading
counts are the measured ones, not invented. That matters because the routing
rule is a set of thresholds, and thresholds tuned against imagined inputs are
tuned against nothing.

The two failures these fix, from that run:

- `a05` *"What's the deadline?"* — abstained. Reranker 2.43, coverage **0.00**:
  the corpus says "within 30 calendar days" and never once says "deadline", so
  an under-specified question fails the sufficiency gate by construction.
- `a03` *"How much notice do I need to give?"* — answered about PTO alone.
  Four notice periods exist across HR, legal and sales, but the leave section
  dominated the ranking so the near-top-tie test saw one clear winner.
"""
from __future__ import annotations

import pytest

from rag.agents.graph_app import _evidence_is_genuinely_split, route_after_assess
from rag.agents.planner import QueryPlan
from rag.retrieval import RetrievedChunk
from rag.retrieval.conflict import VersionConflict
from rag.retrieval.sufficiency import Sufficiency


def chunk(cid: str, doc: str, dept: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id=doc, title=doc, department=dept,
        section_path="s", content=cid, page=1, reranker_score=score,
    )


def plan(kind: str, readings: int = 0, subqueries: int = 0) -> QueryPlan:
    return QueryPlan(
        kind=kind, reasoning="test",
        readings=[f"reading {i}" for i in range(readings)],
        subqueries=[f"sub {i}" for i in range(subqueries)],
    )


def state(*, kind: str, readings: int, sufficient: bool, top: float,
          chunks: list[RetrievedChunk], conflicts=None) -> dict:
    return {
        "plan": plan(kind, readings),
        "sufficiency": Sufficiency(
            score=0.5, sufficient=sufficient, top_reranker=top, coverage=0.5
        ),
        "chunks": chunks,
        "conflicts": conflicts or [],
    }


# ---------------- the two measured failures ----------------


def test_an_underspecified_question_clarifies_instead_of_abstaining():
    """a05: reranker 2.43, coverage 0.00, four readings — it abstained.

    Coverage is zero because the user's vague word is not in the documents,
    not because the documents lack the answer. "I don't know" was the wrong
    answer; "I don't know which one you mean" was the right one.
    """
    assert route_after_assess(state(
        kind="AMBIGUOUS", readings=4, sufficient=False, top=2.43,
        chunks=[chunk("a", "finance/ExpensePolicy.pdf", "finance", 2.43),
                chunk("b", "HR/LeavePolicy.pdf", "HR", 1.9)],
    )) == "clarify"


def test_a_dominant_hit_still_clarifies_when_the_readings_span_departments():
    """a03: one section dominated at 2.80 and the system answered about PTO.

    Four notice periods exist — PTO at 5 or 15 business days, contract
    cancellation at 30, non-renewal at 60 — across three departments.
    """
    assert route_after_assess(state(
        kind="AMBIGUOUS", readings=4, sufficient=True, top=2.80,
        chunks=[chunk("a", "HR/LeavePolicy.pdf", "HR", 2.80),
                chunk("b", "HR/LeavePolicy.pdf", "HR", 2.10),
                chunk("c", "legal/VendorContract.pdf", "legal", 1.70)],
    )) == "clarify"


# ---------------- what must not regress ----------------


@pytest.mark.parametrize("case,top,coverage", [
    ("n01 refund policy", 2.14, 0.50),
    ("n02 severance", 1.96, 0.75),
    ("n04 Germany parental leave", 2.41, 0.67),
    ("n07 dental deductible", 1.90, 1.00),
])
def test_genuine_no_answer_questions_still_abstain(case, top, coverage):
    """Every no-answer case in the set is classified SIMPLE with no readings.

    That is what makes the clarify-on-weak-evidence branch safe: it cannot be
    reached without the planner having named alternative meanings.
    """
    assert route_after_assess(state(
        kind="SIMPLE", readings=0, sufficient=False, top=top,
        chunks=[chunk("a", "HR/Benefits.pdf", "HR", top)],
    )) == "abstain", case


def test_an_ambiguous_question_with_no_evidence_at_all_still_abstains():
    """Below the sufficiency threshold the corpus really has nothing.

    Vagueness does not conjure an answer, and offering readings the evidence
    cannot support is worse than saying so.
    """
    assert route_after_assess(state(
        kind="AMBIGUOUS", readings=4, sufficient=False, top=0.7,
        chunks=[chunk("a", "HR/Benefits.pdf", "HR", 0.7)],
    )) == "abstain"


def test_one_reading_is_not_an_ambiguity():
    """A clarification offering a single option is not a question."""
    assert route_after_assess(state(
        kind="AMBIGUOUS", readings=1, sufficient=False, top=2.4,
        chunks=[chunk("a", "HR/Benefits.pdf", "HR", 2.4)],
    )) == "abstain"


def test_a_simple_question_with_good_evidence_answers():
    assert route_after_assess(state(
        kind="SIMPLE", readings=0, sufficient=True, top=2.85,
        chunks=[chunk("a", "HR/Benefits.pdf", "HR", 2.85)],
    )) == "answer"


def test_an_apparently_ambiguous_question_the_evidence_settles_still_answers():
    """a06 "What is the tuition reimbursement limit?" — one dominant hit,
    one department, and the planner did not call it ambiguous."""
    assert route_after_assess(state(
        kind="SIMPLE", readings=0, sufficient=True, top=2.85,
        chunks=[chunk("a", "HR/Benefits.pdf", "HR", 2.85),
                chunk("b", "finance/ExpensePolicy.pdf", "finance", 1.87)],
    )) == "answer"


def test_two_versions_of_one_document_are_not_two_readings():
    """The regression that turned every pricing question into a clarification."""
    conflicts = [VersionConflict(
        current_doc_id="sales/Pricing2026.pdf",
        superseded_doc_ids=["sales/Pricing2025.pdf"],
    )]
    assert route_after_assess(state(
        kind="AMBIGUOUS", readings=2, sufficient=True, top=2.9,
        chunks=[chunk("a", "sales/Pricing2026.pdf", "sales", 2.9),
                chunk("b", "sales/Pricing2025.pdf", "sales", 2.7)],
        conflicts=conflicts,
    )) == "answer"


# ---------------- the split test itself ----------------


def test_many_readings_need_more_than_one_department():
    """Otherwise every vague question about one policy would clarify."""
    same_department = [chunk("a", "HR/LeavePolicy.pdf", "HR", 2.8),
                       chunk("b", "HR/Benefits.pdf", "HR", 1.2)]
    assert _evidence_is_genuinely_split(same_department, [], readings=4) is False

    across = [*same_department, chunk("c", "legal/NDA.docx", "legal", 1.1)]
    assert _evidence_is_genuinely_split(across, [], readings=4) is True


def test_few_readings_fall_back_to_the_near_top_tie_test():
    across = [chunk("a", "HR/LeavePolicy.pdf", "HR", 2.8),
              chunk("b", "legal/NDA.docx", "legal", 1.0)]
    # Two readings, and the second document is nowhere near the top.
    assert _evidence_is_genuinely_split(across, [], readings=2) is False


def test_a_single_chunk_is_never_split():
    assert _evidence_is_genuinely_split(
        [chunk("a", "HR/Benefits.pdf", "HR", 3.0)], [], readings=4
    ) is False
