"""Evaluation metrics, checked against hand-computed values.

These numbers are the argument that the improved system beats the baseline, so
every formula is verified against an arithmetic result worked out by hand
rather than against whatever the implementation happens to return.
"""
from __future__ import annotations

import math

import pytest

from eval.metrics import (
    BehaviourMatrix,
    citation_accuracy,
    contains_expected,
    hit_rate,
    ndcg_at_k,
    observed_behaviour,
    percentile,
    recall_at_k,
    reciprocal_rank,
    score_behaviour,
)


# ---------------- retrieval ----------------


def test_hit_rate_is_binary_within_k():
    retrieved = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
    assert hit_rate(retrieved, ["c.pdf"], 3) == 1.0
    assert hit_rate(retrieved, ["d.pdf"], 3) == 0.0
    assert hit_rate(retrieved, ["d.pdf"], 4) == 1.0


def test_hit_rate_with_no_expectation_is_vacuously_satisfied():
    """A no-answer case expects no documents, so retrieval cannot fail it."""
    assert hit_rate(["a.pdf"], [], 5) == 1.0


def test_recall_distinguishes_partial_from_complete():
    """The metric that separates a half-answered comparison from a full one."""
    retrieved = ["travel.docx", "unrelated.pdf"]
    expected = ["travel.docx", "expense.pdf"]
    assert recall_at_k(retrieved, expected, 5) == pytest.approx(0.5)
    # Hit rate cannot see this: it scores the same case as a success.
    assert hit_rate(retrieved, expected, 5) == 1.0


def test_recall_is_complete_when_every_expected_document_returns():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], 5) == 1.0


def test_reciprocal_rank_is_one_over_the_first_hit():
    assert reciprocal_rank(["x", "y", "target"], ["target"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["target", "y"], ["target"]) == 1.0
    assert reciprocal_rank(["x", "y"], ["target"]) == 0.0


def test_ndcg_rewards_ranking_the_answer_first():
    top = ndcg_at_k(["target", "x", "y"], ["target"], 5)
    third = ndcg_at_k(["x", "y", "target"], ["target"], 5)
    assert top == 1.0
    assert third < top
    # Hand-computed: DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.0
    assert third == pytest.approx(0.5)


def test_ndcg_with_two_expected_documents():
    # Retrieved at ranks 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
    # Ideal (ranks 1 and 2):      1/log2(2) + 1/log2(3) = 1.0 + 0.63093 = 1.63093
    value = ndcg_at_k(["a", "x", "b"], ["a", "b"], 5)
    assert value == pytest.approx(1.5 / (1.0 + 1.0 / math.log2(3)))


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved():
    assert ndcg_at_k(["x", "y"], ["target"], 5) == 0.0


# ---------------- behaviour ----------------


def test_observed_behaviour_reads_the_result_shape():
    assert observed_behaviour({"answered": True}) == "answer"
    assert observed_behaviour({"answered": False, "abstained": True}) == "abstain"
    assert observed_behaviour(
        {"answered": False, "clarification": "which limit?"}
    ) == "clarify"


def test_clarification_wins_over_answered_flag():
    """A clarification is not an answer, however the flags happen to be set."""
    assert observed_behaviour(
        {"answered": True, "clarification": "which one?"}
    ) == "clarify"


def test_answering_an_unanswerable_question_is_recorded_as_wrongly_answered():
    """The dangerous failure: it looks like a good answer, with citations."""
    matrix = BehaviourMatrix()
    assert score_behaviour("abstain", "answer", matrix) is False
    assert matrix.wrongly_answered == 1
    assert matrix.wrongly_abstained == 0


def test_refusing_an_answerable_question_is_recorded_separately():
    """Over-refusal is the standard way of gaming a hallucination metric."""
    matrix = BehaviourMatrix()
    assert score_behaviour("answer", "abstain", matrix) is False
    assert matrix.wrongly_abstained == 1
    assert matrix.wrongly_answered == 0


def test_behaviour_accuracy_counts_all_three_correct_kinds():
    matrix = BehaviourMatrix()
    score_behaviour("answer", "answer", matrix)
    score_behaviour("abstain", "abstain", matrix)
    score_behaviour("clarify", "clarify", matrix)
    score_behaviour("answer", "abstain", matrix)
    assert matrix.total == 4
    assert matrix.correct == 3
    assert matrix.accuracy() == pytest.approx(0.75)


def test_empty_matrix_scores_zero_not_one():
    assert BehaviourMatrix().accuracy() == 0.0


# ---------------- citations and facts ----------------


def test_citation_accuracy_is_precision_over_cited_documents():
    citations = [{"doc_id": "a.pdf"}, {"doc_id": "b.pdf"}, {"doc_id": "wrong.pdf"}]
    assert citation_accuracy(citations, ["a.pdf", "b.pdf"]) == pytest.approx(2 / 3)


def test_citing_nothing_is_not_scored_at_all():
    """An abstention has nothing to cite, and must not be scored zero for it.

    Scoring it zero makes the metric invert: on the first run of this harness
    the improved pipeline scored *below* the baseline purely because it refused
    five questions the baseline answered wrongly -- each refusal scoring 0, each
    wrong answer scoring 1.
    """
    assert citation_accuracy([], ["a.pdf"]) is None
    assert citation_accuracy([], []) is None


def test_cases_expecting_no_citations_are_not_scored():
    assert citation_accuracy([{"doc_id": "a.pdf"}], []) is None


def test_fact_match_ignores_currency_and_thousands_formatting():
    """"$5,250" in the corpus and "5250" in an answer are the same fact."""
    assert contains_expected("The limit is $5,250 per year.", ["5250"]) is True
    assert contains_expected("The limit is 5250 per year.", ["$5,250"]) is True


def test_fact_match_requires_every_term_by_default():
    answer = "Bronze costs $45 per month."
    assert contains_expected(answer, ["45", "160"]) is False
    assert contains_expected(answer, ["45", "160"], any_of=True) is True


def test_fact_match_with_nothing_required_passes():
    assert contains_expected("anything", []) is True


def test_a_wrong_figure_fails_the_fact_check():
    """The check the LLM judge can be talked out of, and this cannot."""
    assert contains_expected("Tuition is capped at $7,500.", ["5,250"]) is False


# ---------------- aggregation ----------------


def test_percentile_picks_real_observations():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 0.5) == 30.0
    assert percentile(values, 1.0) == 50.0


def test_percentile_of_nothing_is_zero():
    assert percentile([], 0.95) == 0.0
