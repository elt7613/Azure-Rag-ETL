"""Evaluation metrics.

Everything here is pure and hand-checkable. That matters more than it sounds:
these numbers are the argument that the improved architecture is actually
better than the baseline, and a metric nobody can verify by hand is a number
nobody should believe.

Three families, measuring different failures:

- **Retrieval** — did the evidence containing the answer come back at all, and
  how far up? A generation failure downstream of a retrieval failure is not a
  generation problem, and separating them is the whole point of measuring both.
- **Behaviour** — did the system do the right *kind* of thing? Answering a
  question the corpus cannot answer and refusing one it can are both failures,
  and both are invisible to an accuracy score computed only over answered
  questions. Scored as a confusion matrix, not folded into accuracy.
- **System** — latency, tokens, cost. Reported as percentiles because LLM
  latency is heavy-tailed and a mean hides exactly the tail users complain
  about.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------- retrieval


def hit_rate(retrieved: list[str], expected: list[str], k: int) -> float:
    """1.0 if any expected document appears in the top k, else 0.0."""
    if not expected:
        return 1.0
    return 1.0 if set(retrieved[:k]) & set(expected) else 0.0


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of the expected documents present in the top k.

    Distinct from hit rate: a comparison question needing two documents scores
    0.5 here when only one came back, and 1.0 on hit rate. That gap is exactly
    where multi-document questions fail.
    """
    if not expected:
        return 1.0
    found = set(retrieved[:k]) & set(expected)
    return len(found) / len(set(expected))


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """1 / rank of the first expected document, 0 if it never appears."""
    if not expected:
        return 1.0
    wanted = set(expected)
    for index, item in enumerate(retrieved, start=1):
        if item in wanted:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Rewards putting the right document first rather than fifth, which hit rate
    cannot see and which matters because the answer prompt reads the context
    top-down and the context is truncated.
    """
    if not expected:
        return 1.0
    wanted = set(expected)
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, item in enumerate(retrieved[:k], start=1)
        if item in wanted
    )
    ideal = sum(
        1.0 / math.log2(index + 1)
        for index in range(1, min(len(wanted), k) + 1)
    )
    return 0.0 if ideal == 0 else dcg / ideal


# ---------------------------------------------------------------- behaviour


@dataclass
class BehaviourMatrix:
    """How often the system chose the right *kind* of response.

    The two off-diagonal cells are the ones that matter:

    - `wrongly_answered` — it answered a question the corpus cannot answer.
      This is hallucination in its most damaging form, because the answer
      arrives with citations and looks exactly like a good one.
    - `wrongly_abstained` — it refused a question the corpus does answer.
      Less dangerous, more corrosive: a system that refuses too often stops
      being used, and over-refusal is the standard way of gaming a
      hallucination metric.
    """

    correctly_answered: int = 0
    correctly_abstained: int = 0
    correctly_clarified: int = 0
    wrongly_answered: int = 0
    wrongly_abstained: int = 0
    wrongly_clarified: int = 0

    @property
    def total(self) -> int:
        return (
            self.correctly_answered + self.correctly_abstained
            + self.correctly_clarified + self.wrongly_answered
            + self.wrongly_abstained + self.wrongly_clarified
        )

    @property
    def correct(self) -> int:
        return (
            self.correctly_answered + self.correctly_abstained
            + self.correctly_clarified
        )

    def accuracy(self) -> float:
        return 0.0 if not self.total else self.correct / self.total

    def as_dict(self) -> dict:
        return {
            "correctly_answered": self.correctly_answered,
            "correctly_abstained": self.correctly_abstained,
            "correctly_clarified": self.correctly_clarified,
            "wrongly_answered": self.wrongly_answered,
            "wrongly_abstained": self.wrongly_abstained,
            "wrongly_clarified": self.wrongly_clarified,
            "behaviour_accuracy": round(self.accuracy(), 4),
        }


def observed_behaviour(result: dict) -> str:
    """Classify what the system actually did: answer, clarify or abstain."""
    if result.get("clarification"):
        return "clarify"
    if result.get("answered"):
        return "answer"
    return "abstain"


_PARTICIPLE = {"answer": "answered", "abstain": "abstained", "clarify": "clarified"}


def score_behaviour(expected: str, observed: str, matrix: BehaviourMatrix) -> bool:
    """Record one case in the matrix, keyed on what the system actually did.

    Note the cell is chosen by the *observed* behaviour, not the expected one:
    "it answered when it should have abstained" and "it abstained when it
    should have answered" are different failures with different remedies, and
    collapsing them into a single error count loses the distinction.
    """
    correct = expected == observed
    prefix = "correctly" if correct else "wrongly"
    field_name = f"{prefix}_{_PARTICIPLE[observed]}"
    setattr(matrix, field_name, getattr(matrix, field_name) + 1)
    return correct


# ---------------------------------------------------------------- citations


def citation_accuracy(citations: list[dict], expected_docs: list[str]) -> float | None:
    """Fraction of cited documents that are ones the answer should rest on.

    Precision rather than recall, deliberately: citing a document that does not
    support the claim is the failure being measured. Citing fewer than expected
    is a retrieval or answer-completeness problem and shows up in those metrics.

    Returns **None** when there was nothing to cite -- an abstention or a
    clarifying question. This is not a detail. Scoring those 0.0 punishes the
    system for correctly declining, and rewards a system that answers anyway
    and cites something: on the first run of this harness the improved
    pipeline's citation accuracy came out *below* the baseline's purely because
    it refused five questions the baseline answered wrongly, each refusal
    scoring zero while each wrong answer scored one. A metric that inverts on
    the behaviour it is meant to encourage is worse than no metric.
    """
    if not citations:
        return None
    if not expected_docs:
        return None
    wanted = set(expected_docs)
    correct = sum(1 for c in citations if c.get("doc_id") in wanted)
    return correct / len(citations)


def contains_expected(answer: str, must_contain: list[str], any_of: bool = False) -> bool:
    """Whether the answer states the required facts.

    A crude but unfakeable check that sits underneath the LLM judge: a judge can
    be talked into calling a wrong number right, but "$5,250" either appears in
    the answer or it does not. `any_of` covers facts the documents state in more
    than one form ("two years" / "2 years").
    """
    if not must_contain:
        return True
    normalized = answer.lower().replace(",", "").replace("$", "")
    checks = [
        term.lower().replace(",", "").replace("$", "") in normalized
        for term in must_contain
    ]
    return any(checks) if any_of else all(checks)


# ---------------------------------------------------------------- aggregation


@dataclass
class RunTotals:
    """Accumulated scores for one evaluation run."""

    cases: int = 0
    hit_rate_3: list[float] = field(default_factory=list)
    hit_rate_5: list[float] = field(default_factory=list)
    hit_rate_10: list[float] = field(default_factory=list)
    recall_5: list[float] = field(default_factory=list)
    mrr: list[float] = field(default_factory=list)
    ndcg_5: list[float] = field(default_factory=list)
    correctness: list[float] = field(default_factory=list)
    groundedness: list[float] = field(default_factory=list)
    citation_accuracy: list[float] = field(default_factory=list)
    fact_match: list[float] = field(default_factory=list)
    hallucinations: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    behaviour: BehaviourMatrix = field(default_factory=BehaviourMatrix)


def mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(totals: RunTotals, *, cost_per_1m_input: float,
              cost_per_1m_output: float) -> dict:
    """The report's numbers, rounded once, here, so nothing rounds twice."""
    cost = (
        totals.prompt_tokens / 1_000_000 * cost_per_1m_input
        + totals.completion_tokens / 1_000_000 * cost_per_1m_output
    )
    return {
        "cases": totals.cases,
        "retrieval": {
            "hit_rate@3": round(mean(totals.hit_rate_3), 4),
            "hit_rate@5": round(mean(totals.hit_rate_5), 4),
            "hit_rate@10": round(mean(totals.hit_rate_10), 4),
            "recall@5": round(mean(totals.recall_5), 4),
            "mrr": round(mean(totals.mrr), 4),
            "ndcg@5": round(mean(totals.ndcg_5), 4),
        },
        "generation": {
            "answer_correctness": round(mean(totals.correctness), 4),
            "fact_match": round(mean(totals.fact_match), 4),
            "groundedness": round(mean(totals.groundedness), 4),
            "citation_accuracy": round(mean(totals.citation_accuracy), 4),
            "hallucination_rate": round(
                totals.hallucinations / totals.cases if totals.cases else 0.0, 4
            ),
        },
        "behaviour": totals.behaviour.as_dict(),
        "system": {
            "latency_p50_ms": round(percentile(totals.latencies_ms, 0.50), 1),
            "latency_p95_ms": round(percentile(totals.latencies_ms, 0.95), 1),
            "latency_mean_ms": round(mean(totals.latencies_ms), 1),
            "prompt_tokens": totals.prompt_tokens,
            "completion_tokens": totals.completion_tokens,
            "estimated_cost_usd": round(cost, 6),
            "cost_per_query_usd": round(cost / totals.cases, 6) if totals.cases else 0.0,
        },
    }
