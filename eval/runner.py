"""Run the golden set against two architectures and score them identically.

The comparison is only worth anything if it is fair, so three things are held
constant:

- **The same corpus, index and questions.** Both configurations retrieve from
  the same live Azure AI Search index.
- **The same measuring instruments.** Groundedness is measured by running the
  claim-level verifier over *both* systems' answers. In the improved pipeline
  the verifier is also a gate; here it is only a thermometer. Using a stricter
  instrument on one side than the other would manufacture the result.
- **The same judge, prompt and temperature** for answer correctness.

What differs is only the architecture:

| | baseline | improved |
|---|---|---|
| retrieval | vector-only, top 5 | hybrid + semantic reranking + neighbour expansion |
| query handling | the raw message | condensed, classified, decomposed |
| ranking | none | reranked, version conflicts resolved |
| grounding | always answers | sufficiency gate, verification, abstention |

The baseline is a real naive RAG implementation, not a strawman: it is the
pipeline this repository had before the retrieval work, and it is the thing
most RAG tutorials produce.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from rag.agents import USAGE, build_agent, record_usage
from rag.agents.answerer import answer as answer_question, resolve_citations
from rag.agents.condense import Turn
from rag.agents.graph_app import RetrievalContext, build_graph, run_turn
from rag.agents.verifier import check_citations, verify
from rag.config import get_settings
from rag.retrieval.searcher import HybridSearcher

from eval.metrics import (
    RunTotals,
    citation_accuracy,
    contains_expected,
    hit_rate,
    ndcg_at_k,
    observed_behaviour,
    recall_at_k,
    reciprocal_rank,
    score_behaviour,
    summarize,
)

# Backoff between attempts at a single case. Three tries then give up: a
# genuine outage should stop the run quickly rather than retrying 51 times.
_RETRY_DELAYS = (5, 20)

DATASET = pathlib.Path(__file__).parent / "dataset.jsonl"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

BASELINE_TOP_K = 5


# ---------------------------------------------------------------- judge


_JUDGE_SYSTEM = """\
You score a candidate answer against a reference answer for a question about \
internal company policy documents.

Score 0 to 4 on factual agreement ONLY:
- 4: states the same facts as the reference. Extra correct detail is fine.
- 3: correct but incomplete -- part of the reference's substance is missing.
- 2: partly correct, with a material omission or a vague figure where the \
reference gives a specific one.
- 1: mostly wrong, or right topic with wrong facts.
- 0: wrong, contradicts the reference, or does not answer the question.

Rules that override your instincts:
- A wrong number, date or threshold is a 0 or 1, never a 3. Looking confident \
and well-written does not raise the score.
- Length, tone, formatting and the presence of citations are irrelevant. Do \
not reward a longer answer.
- If the reference says the corpus does not contain the answer, then a \
candidate that correctly declines scores 4, and one that supplies an answer \
anyway scores 0 however plausible it sounds.
- If the reference expects a clarifying question, a candidate that asks a \
relevant clarifying question scores 4.
"""


class Judgement(BaseModel):
    score: int = Field(ge=0, le=4)
    reason: str = Field(description="One sentence.")


_judge = None


def _get_judge():
    global _judge
    if _judge is None:
        _judge = build_agent(Judgement, _JUDGE_SYSTEM)
    return _judge


async def judge_answer(question: str, reference: str, candidate: str) -> Judgement:
    result = await _get_judge().run(
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Candidate answer: {candidate}"
    )
    record_usage("judge", result)
    return result.output


# ---------------------------------------------------------------- baseline


async def baseline_turn(
    searcher: HybridSearcher, message: str, *, departments: list[str], history=None
) -> dict[str, Any]:
    """Naive RAG: embed the question, take the top 5, answer from them.

    No query rewriting, no reranking, no sufficiency gate, no verification, and
    critically no abstention path -- the answerer is still told it may decline,
    but nothing upstream stops a weak retrieval reaching it and nothing
    downstream checks what comes back. That is the shape of the failure this
    project is about.

    History is passed through raw, which is exactly how conversational RAG goes
    wrong: the retriever sees the whole transcript rather than a resolved
    question.
    """
    started = time.perf_counter()
    raw_query = message
    if history:
        raw_query = " ".join([*(t.content for t in history), message])

    chunks = await searcher.search(
        raw_query, departments=departments, top=BASELINE_TOP_K, semantic=False
    )
    draft = await answer_question(message, chunks, history=history)
    latency_ms = (time.perf_counter() - started) * 1000

    return {
        "answer": draft.answer,
        "answered": bool(draft.answered),
        "abstained": not draft.answered,
        "clarification": "",
        "citations": resolve_citations(draft, chunks),
        "confidence": 0.0,
        "chunks": chunks,
        "draft": draft,
        "latency_ms": latency_ms,
        "diagnostics": {"mode": "baseline", "candidates": len(chunks)},
    }


async def improved_turn(
    pipeline, message: str, *, departments: list[str], history=None
) -> dict[str, Any]:
    started = time.perf_counter()
    state = await run_turn(pipeline, message, departments=departments, history=history)
    return {
        "answer": state.get("answer", ""),
        "answered": bool(state.get("answered")),
        "abstained": bool(state.get("abstained")),
        "clarification": state.get("clarification", ""),
        "citations": state.get("citations", []),
        "confidence": float(state.get("confidence", 0.0)),
        "chunks": state.get("chunks", []),
        "draft": state.get("draft"),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "diagnostics": state.get("diagnostics", {}),
    }


# ---------------------------------------------------------------- scoring


@dataclass
class CaseResult:
    case_id: str
    category: str
    mode: str
    question: str
    answer: str
    expected_behaviour: str
    observed_behaviour: str
    behaviour_correct: bool
    judge_score: int
    judge_reason: str
    fact_match: bool
    retrieved_docs: list[str]
    expected_docs: list[str]
    hit_rate_5: float
    recall_5: float
    mrr: float
    ndcg_5: float
    citation_accuracy: float | None
    groundedness: float
    contradicted_claims: int
    latency_ms: float
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _errored_case(case: dict, mode: str, exc: Exception) -> CaseResult:
    """A case that could not be run at all.

    Scored as a failure on every axis, and visibly labelled, so it drags the
    numbers down rather than quietly vanishing from the denominator -- a run
    with three silently-dropped cases reports a better score than it earned.
    """
    return CaseResult(
        case_id=case["id"], category=case["category"], mode=mode,
        question=case["question"],
        answer=f"[harness error] {type(exc).__name__}: {exc}",
        expected_behaviour=case["expected_behavior"], observed_behaviour="abstain",
        behaviour_correct=case["expected_behavior"] == "abstain",
        judge_score=0, judge_reason="case failed to run", fact_match=False,
        retrieved_docs=[], expected_docs=case.get("expected_docs", []),
        hit_rate_5=0.0, recall_5=0.0, mrr=0.0, ndcg_5=0.0,
        citation_accuracy=None, groundedness=0.0, contradicted_claims=0,
        latency_ms=0.0, diagnostics={"harness_error": type(exc).__name__},
    )


async def score_case(case: dict, result: dict, mode: str) -> CaseResult:
    expected_docs = case.get("expected_docs", [])
    retrieved_docs: list[str] = []
    for chunk in result.get("chunks", []):
        doc_id = getattr(chunk, "doc_id", None)
        if doc_id and doc_id not in retrieved_docs:
            retrieved_docs.append(doc_id)

    expected_behaviour = case["expected_behavior"]
    observed = observed_behaviour(result)

    judgement = await judge_answer(
        case["question"], case["expected_answer"], result["answer"]
    )

    fact_match = contains_expected(
        result["answer"],
        case.get("must_contain", []),
        any_of=bool(case.get("match_any")),
    )

    # Groundedness is measured for BOTH modes with the same instrument. In the
    # improved pipeline the verifier is also a gate; here it is only a
    # thermometer, so the comparison is not rigged by measuring one side more
    # strictly than the other.
    groundedness, contradicted = 1.0, 0
    draft = result.get("draft")
    if result.get("answered") and draft is not None and result.get("chunks"):
        verification = await verify(case["question"], draft, result["chunks"])
        groundedness = verification.groundedness()
        contradicted = verification.contradicted

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        mode=mode,
        question=case["question"],
        answer=result["answer"],
        expected_behaviour=expected_behaviour,
        observed_behaviour=observed,
        behaviour_correct=expected_behaviour == observed,
        judge_score=judgement.score,
        judge_reason=judgement.reason,
        fact_match=fact_match,
        retrieved_docs=retrieved_docs,
        expected_docs=expected_docs,
        hit_rate_5=hit_rate(retrieved_docs, expected_docs, 5),
        recall_5=recall_at_k(retrieved_docs, expected_docs, 5),
        mrr=reciprocal_rank(retrieved_docs, expected_docs),
        ndcg_5=ndcg_at_k(retrieved_docs, expected_docs, 5),
        citation_accuracy=citation_accuracy(
            result.get("citations", []), expected_docs
        ),
        groundedness=groundedness,
        contradicted_claims=contradicted,
        latency_ms=result["latency_ms"],
        diagnostics=result.get("diagnostics", {}),
    )


def accumulate(totals: RunTotals, scored: CaseResult, retrieved: list[str]) -> None:
    totals.cases += 1
    expected = scored.expected_docs
    totals.hit_rate_3.append(hit_rate(retrieved, expected, 3))
    totals.hit_rate_5.append(scored.hit_rate_5)
    totals.hit_rate_10.append(hit_rate(retrieved, expected, 10))
    totals.recall_5.append(scored.recall_5)
    totals.mrr.append(scored.mrr)
    totals.ndcg_5.append(scored.ndcg_5)
    totals.correctness.append(scored.judge_score / 4.0)
    totals.fact_match.append(1.0 if scored.fact_match else 0.0)
    totals.groundedness.append(scored.groundedness)
    if scored.citation_accuracy is not None:
        totals.citation_accuracy.append(scored.citation_accuracy)
    totals.latencies_ms.append(scored.latency_ms)
    # A hallucination is a claim the evidence contradicts, or an answer given
    # to a question the corpus cannot answer. Both are the system asserting
    # something it has no support for.
    if scored.contradicted_claims or (
        scored.expected_behaviour == "abstain" and scored.observed_behaviour == "answer"
    ):
        totals.hallucinations += 1
    score_behaviour(scored.expected_behaviour, scored.observed_behaviour, totals.behaviour)


# ---------------------------------------------------------------- driver


def load_cases(path: pathlib.Path = DATASET) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def run_mode(mode: str, cases: list[dict], *, concurrency: int = 3) -> dict:
    """Run every case through one architecture and return the scored report."""
    settings = get_settings()
    searcher = HybridSearcher()

    # The improved configuration is the *whole* system, graph retrieval
    # included -- measuring it without the graph would understate what is
    # actually deployed and make the comparison a different one from the claim.
    graph_retriever = None
    pipeline = None
    if mode == "improved":
        if settings.graph_enabled:
            try:
                from rag.retrieval.graph_retriever import GraphRetriever

                graph_retriever = GraphRetriever(searcher)
                await graph_retriever.ping()
            except Exception:
                # Report vector-only rather than silently claiming the graph
                # contributed: a benchmark that quietly loses a component is
                # worse than one that says so.
                print("  ! graph retriever unavailable; running vector-only")
                graph_retriever = None
        pipeline = build_graph(RetrievalContext(searcher, graph_retriever))

    USAGE.reset()
    totals = RunTotals()
    scored_cases: list[CaseResult] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def attempt(case: dict) -> CaseResult:
        history = [Turn(role=t["role"], content=t["content"])
                   for t in case.get("history", [])]
        departments = case.get("departments") or settings.departments
        if mode == "baseline":
            result = await baseline_turn(
                searcher, case["question"],
                departments=departments, history=history,
            )
        else:
            result = await improved_turn(
                pipeline, case["question"],
                departments=departments, history=history,
            )
        return await score_case(case, result, mode)

    async def one(case: dict) -> CaseResult:
        """Run one case, surviving a transient failure.

        A full run is ~45 minutes of live API calls, and an unretried
        `APITimeoutError` on case 12 destroys all of it -- which is exactly
        what happened once. A benchmark that cannot survive one network blip
        is not a benchmark you can rely on, so each case gets its own retries,
        and a case that still fails is recorded as a failure rather than
        taking down the run: losing one data point is a footnote, losing the
        run is an hour.
        """
        async with semaphore:
            for delay in _RETRY_DELAYS:
                try:
                    return await attempt(case)
                except Exception as exc:  # noqa: BLE001 - retried, then recorded
                    print(f"  ! {case['id']}: {type(exc).__name__}; retrying in {delay}s")
                    await asyncio.sleep(delay)
            try:
                return await attempt(case)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! {case['id']} failed permanently: {type(exc).__name__}: {exc}")
                return _errored_case(case, mode, exc)

    try:
        results = await asyncio.gather(*(one(c) for c in cases))
    finally:
        if graph_retriever is not None:
            await graph_retriever.aclose()
        await searcher.aclose()

    for scored in results:
        scored_cases.append(scored)
        accumulate(totals, scored, scored.retrieved_docs)

    usage = USAGE.snapshot()
    totals.prompt_tokens = usage["input_tokens"]
    totals.completion_tokens = usage["output_tokens"]

    report = summarize(
        totals,
        cost_per_1m_input=settings.cost_per_1m_input,
        cost_per_1m_output=settings.cost_per_1m_output,
    )
    report["mode"] = mode
    report["graph_retrieval"] = graph_retriever is not None
    report["usage"] = usage
    report["by_category"] = _by_category(scored_cases)
    report["cases_detail"] = [c.as_dict() for c in scored_cases]
    return report


def _by_category(cases: list[CaseResult]) -> dict:
    """Per-category scores.

    An aggregate hides which kind of question a change actually helped, and
    'the average went up' is not a finding.
    """
    grouped: dict[str, list[CaseResult]] = {}
    for case in cases:
        grouped.setdefault(case.category, []).append(case)

    out: dict[str, dict] = {}
    for category, group in sorted(grouped.items()):
        out[category] = {
            "cases": len(group),
            "behaviour_correct": sum(1 for c in group if c.behaviour_correct),
            "mean_judge_score": round(
                sum(c.judge_score for c in group) / len(group), 3
            ),
            "fact_match": round(
                sum(1 for c in group if c.fact_match) / len(group), 3
            ),
            "hit_rate@5": round(sum(c.hit_rate_5 for c in group) / len(group), 3),
            "mean_latency_ms": round(
                sum(c.latency_ms for c in group) / len(group), 1
            ),
        }
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation.")
    parser.add_argument(
        "--mode", choices=["baseline", "improved", "both"], default="both"
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only N cases.")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--category", default="", help="Run only one category.")
    args = parser.parse_args()

    cases = load_cases()
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    modes = ["baseline", "improved"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print(f"\n=== {mode}: {len(cases)} cases ===")
        report = await run_mode(mode, cases, concurrency=args.concurrency)
        out = RESULTS_DIR / f"{mode}.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(
            {k: v for k, v in report.items() if k != "cases_detail"},
            indent=2, default=str,
        ))
        print(f"written to {out}")


if __name__ == "__main__":
    asyncio.run(main())
