"""The abstention gate: does the evidence actually support an answer?"""
from __future__ import annotations

from rag.retrieval import RetrievedChunk
from rag.retrieval.sufficiency import (
    assess,
    content_terms,
    score_concentration,
    term_coverage,
)

THRESHOLD = 1.6


def ev(content: str, reranker: float | None = 2.5, cid: str = "c") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id="HR/LeavePolicy.pdf", title="LeavePolicy",
        department="HR", section_path="2.1", content=content, page=1,
        reranker_score=reranker,
    )


def test_content_terms_drops_stopwords_and_keeps_figures():
    terms = content_terms("How many days of PTO do I get after 5 years?")
    assert "days" in terms and "pto" in terms and "years" in terms
    assert "how" not in terms and "do" not in terms and "of" not in terms
    assert "5" in terms


def test_term_coverage_is_full_when_everything_appears():
    coverage, missing = term_coverage(
        "PTO accrual days", [ev("Employees accrue PTO; the annual accrual is 15 days.")]
    )
    assert coverage == 1.0
    assert missing == []


def test_term_coverage_reports_what_is_missing():
    coverage, missing = term_coverage(
        "severance pay policy", [ev("Employees accrue PTO on a bi-weekly basis.")]
    )
    assert coverage < 0.5
    assert "severance" in missing


def test_term_coverage_matches_a_formatted_amount():
    """"$5,250" in the question must match "$5,250" in the evidence.

    Punctuation-sensitive matching would score this as a miss and refuse a
    question the corpus answers verbatim.
    """
    coverage, missing = term_coverage(
        "tuition reimbursement limit $5,250",
        [ev("Tuition reimbursement is limited to $5,250 per calendar year.")],
    )
    assert "$5,250" not in missing
    assert coverage == 1.0


def test_concentration_is_high_when_one_hit_dominates():
    chunks = [ev("a", 3.8, "1"), ev("b", 0.4, "2"), ev("c", 0.3, "3")]
    assert score_concentration(chunks) > 0.8


def test_concentration_is_low_when_scores_are_flat():
    chunks = [ev("a", 1.0, "1"), ev("b", 1.0, "2"), ev("c", 1.0, "3")]
    assert score_concentration(chunks) == 0.0


def test_no_evidence_is_never_sufficient():
    verdict = assess("what is the severance policy", [], threshold=THRESHOLD)
    assert verdict.sufficient is False
    assert verdict.score == 0.0
    assert "no evidence" in verdict.reason


def test_strong_on_topic_evidence_is_sufficient():
    verdict = assess(
        "how many sick days per year",
        [ev("Employees receive 10 days of paid sick leave per calendar year.", 3.1)],
        threshold=THRESHOLD,
    )
    assert verdict.sufficient is True
    assert verdict.top_reranker == 3.1
    assert "3.10" in verdict.reason


def test_weak_evidence_is_refused_and_says_why():
    verdict = assess(
        "what is the severance policy",
        [ev("Employees accrue PTO on a bi-weekly basis.", 0.6)],
        threshold=THRESHOLD,
    )
    assert verdict.sufficient is False
    assert "0.60" in verdict.reason
    assert "required 1.60" in verdict.reason


def test_topically_adjacent_evidence_is_refused_on_coverage():
    """The reranker liked it, but it does not mention what was asked about.

    This is the case a score threshold alone misses: a passage about leave
    scores well against a question about leave-adjacent severance, and would be
    answered from confidently and wrongly.
    """
    verdict = assess(
        "severance entitlement redundancy payout",
        [ev("Employees accrue paid time off on a bi-weekly basis.", 2.4)],
        threshold=THRESHOLD,
    )
    assert verdict.sufficient is False
    assert "does not mention" in verdict.reason
    assert "severance" in verdict.missing_terms


def test_missing_reranker_scores_fall_back_to_coverage_and_say_so():
    verdict = assess(
        "sick leave days",
        [ev("Employees receive 10 days of paid sick leave.", None)],
        threshold=THRESHOLD,
    )
    assert verdict.sufficient is True
    assert verdict.top_reranker is None
    assert "term coverage only" in verdict.reason


def test_score_is_bounded_and_monotonic_in_evidence_quality():
    weak = assess("sick leave days", [ev("sick leave days", 1.0)], threshold=0.0)
    strong = assess("sick leave days", [ev("sick leave days", 4.0)], threshold=0.0)
    assert 0.0 <= weak.score <= 1.0
    assert 0.0 <= strong.score <= 1.0
    assert strong.score > weak.score
