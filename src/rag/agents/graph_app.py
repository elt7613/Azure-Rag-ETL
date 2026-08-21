"""The retrieval pipeline as a LangGraph state machine.

Why a graph rather than a function: the pipeline is not a straight line. It
branches on what the question turns out to be, it can decide mid-flight that it
has nothing worth answering from, and it can loop once when verification
catches a problem. Expressing that as nested conditionals hides the control
flow; expressing it as a graph makes each decision a named edge you can trace,
which is what you need when someone reports that one answer in fifty is wrong.

```
condense ─► plan ─┬─► out_of_scope ──────────────────────────► respond
                  ├─► clarify* ────────────────────────────► respond
                  └─► retrieve ─► expand ─► resolve ─► assess
                                                        │
                                    ┌── insufficient ───┴── sufficient ──┐
                                    ▼                                     ▼
                                 abstain ─► respond                    answer
                                                                          │
                                                                       verify
                                                                          │
                                          ┌── failed (first try) ─────────┤
                                          ▼                               │
                                    answer (retry, stricter) ─► verify ───┤
                                                                          ▼
                                          failed again ─► abstain ─► respond
                                                                    pass ─► respond
```

*`clarify` is reached only after retrieval, never before. An ambiguous question
is a suspicion the evidence has to confirm: if every retrieved passage points at
one reading, answering it is better than interrogating the user.

Cost shape: a SIMPLE question costs one condense (skipped on the first turn),
one plan, one search, one answer, one verify. Decomposition, graph traversal
and the corrective retry are all paid for only when the question earns them.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from rag.agents.answerer import GroundedAnswer, answer as answer_question, resolve_citations
from rag.agents.condense import Turn, condense, retrieval_query
from rag.agents.planner import QueryPlan, plan as plan_query
from rag.agents.verifier import Verification, check_citations, passes, verify
from rag.config import get_settings
from rag.retrieval import RetrievedChunk
from rag.retrieval.conflict import conflict_note, resolve as resolve_conflicts
from rag.retrieval.fusion import (
    dedupe_by_content,
    merge_expansion,
    neighbour_ids,
    reciprocal_rank_fusion,
    rerank,
    top_n,
)
from rag.retrieval.sufficiency import Sufficiency, assess

logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    """Everything the pipeline reads or writes for one turn.

    Kept flat and explicit rather than nested: a checkpointed state that is
    hard to read is hard to debug, and this one is written to a store and read
    back on the next turn.
    """

    # Input
    message: str
    history: list[Turn]
    departments: list[str]

    # Working state
    standalone_query: str
    plan: QueryPlan | None
    chunks: list[RetrievedChunk]
    conflicts: list[Any]
    sufficiency: Sufficiency | None
    draft: GroundedAnswer | None
    verification: Verification | None
    retry_count: int

    # Output
    answer: str
    answered: bool
    abstained: bool
    clarification: str
    citations: list[dict]
    confidence: float
    diagnostics: dict


class RetrievalContext:
    """The live dependencies the graph nodes call out to.

    Injected rather than imported so the API can own their lifecycle (open
    once, close at shutdown) and so tests can substitute a fake retriever
    without patching module globals.
    """

    def __init__(self, searcher, graph_retriever=None) -> None:
        self.searcher = searcher
        self.graph_retriever = graph_retriever


# ---------------------------------------------------------------- nodes


def _mark(state: ChatState, key: str, value: Any) -> None:
    state.setdefault("diagnostics", {})[key] = value


async def condense_node(state: ChatState) -> dict:
    started = time.perf_counter()
    condensed = await condense(state["message"], state.get("history"))
    query = retrieval_query(condensed, state["message"])
    return {
        "standalone_query": query,
        "diagnostics": {
            **state.get("diagnostics", {}),
            "condensed": condensed.rewritten,
            "standalone_query": query,
            "condense_ms": round((time.perf_counter() - started) * 1000),
        },
    }


async def plan_node(state: ChatState) -> dict:
    started = time.perf_counter()
    query_plan = await plan_query(
        state["standalone_query"], known_departments=state.get("departments")
    )
    return {
        "plan": query_plan,
        "diagnostics": {
            **state.get("diagnostics", {}),
            "plan_kind": query_plan.kind,
            "plan_reason": query_plan.reasoning,
            "subqueries": query_plan.subqueries,
            "readings": query_plan.readings,
            "plan_ms": round((time.perf_counter() - started) * 1000),
        },
    }


def route_after_plan(state: ChatState) -> str:
    query_plan = state.get("plan")
    if query_plan is not None and query_plan.kind == "OUT_OF_SCOPE":
        return "out_of_scope"
    return "retrieve"


def make_retrieve_node(context: RetrievalContext):
    async def retrieve_node(state: ChatState) -> dict:
        started = time.perf_counter()
        settings = get_settings()
        query_plan = state["plan"]
        queries = query_plan.search_queries(state["standalone_query"])
        departments = state.get("departments") or []

        async def vector(query: str) -> list[RetrievedChunk]:
            return await context.searcher.search(
                query, departments=departments, top=settings.rerank_top_k
            )

        tasks = [vector(q) for q in queries]
        if context.graph_retriever is not None:
            tasks.append(
                context.graph_retriever.retrieve(
                    state["standalone_query"],
                    departments=departments,
                    top=settings.rerank_top_k,
                )
            )

        # One slow or failing retriever must not sink the turn: a graph outage
        # should degrade the answer, not prevent it.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ranked: list[list[RetrievedChunk]] = []
        failures: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                failures.append(f"{type(result).__name__}: {result}")
                logger.warning("retriever failed", exc_info=result)
            else:
                ranked.append(result)

        fused = dedupe_by_content(reciprocal_rank_fusion(ranked))
        return {
            "chunks": fused,
            "diagnostics": {
                **state.get("diagnostics", {}),
                "queries_run": queries,
                "retrievers_failed": failures,
                "candidates": len(fused),
                "retrieve_ms": round((time.perf_counter() - started) * 1000),
            },
        }

    return retrieve_node


def make_expand_node(context: RetrievalContext):
    async def expand_node(state: ChatState) -> dict:
        settings = get_settings()
        chunks = state.get("chunks") or []
        if not chunks:
            return {"chunks": chunks}

        # Only widen around the hits that will actually survive into the
        # context; expanding around a chunk that gets cut is wasted work.
        head = top_n(chunks, settings.retrieval_top_k)
        wanted = neighbour_ids(head)
        if not wanted:
            return {"chunks": chunks}

        try:
            neighbours = await context.searcher.fetch_chunks(
                wanted, departments=state.get("departments") or []
            )
        except Exception:
            logger.warning("neighbour expansion failed", exc_info=True)
            return {"chunks": chunks}

        widened = merge_expansion(chunks, neighbours)
        return {
            "chunks": widened,
            "diagnostics": {
                **state.get("diagnostics", {}),
                "neighbours_added": len(widened) - len(chunks),
            },
        }

    return expand_node


async def rerank_node(state: ChatState) -> dict:
    """Let the cross-encoder lead the final ordering.

    Separate from fusion on purpose: fusion decides which candidates survive,
    reranking decides what the answer prompt reads first. Collapsing the two
    means RRF's position-only view silently overrides the only signal that
    actually read the query against the passage.
    """
    return {"chunks": rerank(state.get("chunks") or [])}


async def resolve_node(state: ChatState) -> dict:
    ordered, conflicts = resolve_conflicts(state.get("chunks") or [])
    return {
        "chunks": ordered,
        "conflicts": conflicts,
        "diagnostics": {
            **state.get("diagnostics", {}),
            "version_conflicts": [c.describe() for c in conflicts],
        },
    }


# A decomposed question needs evidence for each of its parts, so the context
# budget grows with the decomposition. Fixed at the single-question budget, a
# three-part question gets under three chunks per part and the last part is
# simply not in the context -- observed directly: "what cabin am I allowed and
# who approves the expense?" retrieved the approval matrix and then truncated
# it away, and the answer said the approval authority was not stated.
_CHUNKS_PER_SUBQUERY = 3


def _context_budget(state: ChatState, settings) -> int:
    query_plan = state.get("plan")
    budget = settings.retrieval_top_k
    if query_plan is not None and query_plan.subqueries:
        budget += _CHUNKS_PER_SUBQUERY * len(query_plan.subqueries)
    # Never past what the reranker actually scored: beyond that the ordering is
    # RRF's, not the cross-encoder's, and the extra chunks are unranked filler.
    return min(budget, settings.rerank_top_k)


async def assess_node(state: ChatState) -> dict:
    settings = get_settings()
    chunks = top_n(state.get("chunks") or [], _context_budget(state, settings))
    verdict = assess(
        state["standalone_query"], chunks, threshold=settings.sufficiency_threshold
    )
    return {
        "chunks": chunks,
        "sufficiency": verdict,
        "confidence": verdict.score,
        "diagnostics": {
            **state.get("diagnostics", {}),
            "sufficiency": verdict.score,
            "sufficiency_reason": verdict.reason,
            "top_reranker": verdict.top_reranker,
            "coverage": round(verdict.coverage, 3),
            "context_chunks": len(chunks),
        },
    }


def route_after_assess(state: ChatState) -> str:
    query_plan = state.get("plan")
    verdict = state.get("sufficiency")
    ambiguous = (
        query_plan is not None
        and query_plan.kind == "AMBIGUOUS"
        and len(query_plan.readings) >= _MIN_READINGS_TO_CLARIFY
    )

    if verdict is None or not verdict.sufficient:
        # An under-specified question fails the sufficiency gate *by
        # construction*: the user's vague word is not in the documents, so
        # coverage is near zero even though the documents are full of the
        # thing they mean. "What's the deadline?" scored 2.43 on the reranker
        # with 0.00 term coverage -- the corpus says "within 30 calendar days",
        # never "deadline" -- and abstained. That answer is "I don't know",
        # when the truth is "I don't know which one you mean", and the planner
        # had already produced four concrete readings to offer.
        #
        # Gated on there being topically relevant evidence at all. Below the
        # sufficiency threshold the corpus really has nothing, and abstaining
        # is right however vague the question was. Every genuine no-answer
        # case in the evaluation set is classified SIMPLE with no readings, so
        # this branch cannot reach them.
        if ambiguous and (verdict is not None and (verdict.top_reranker or 0.0)
                          >= get_settings().sufficiency_threshold):
            return "clarify"
        return "abstain"

    if ambiguous and _evidence_is_genuinely_split(
        state.get("chunks") or [],
        state.get("conflicts") or [],
        readings=len(query_plan.readings),
    ):
        return "clarify"
    return "answer"


# How close to the best score a second passage must come to count as a rival
# reading rather than background. Azure's reranker runs 0-4, and in practice a
# passage scoring below ~80% of the winner is not competing with it.
_RIVAL_SCORE_RATIO = 0.8


# Below this the planner has not really identified alternative meanings, and a
# clarification would offer the user a choice of one.
_MIN_READINGS_TO_CLARIFY = 2
# When the planner sees this many distinct readings, evidence spanning more than
# one department is enough to confirm them -- see `_evidence_is_genuinely_split`.
_MANY_READINGS = 3


def _evidence_is_genuinely_split(
    chunks: list[RetrievedChunk], conflicts, *, readings: int = 0
) -> bool:
    """Whether the retrieved evidence really does support several readings.

    The planner's AMBIGUOUS verdict is a suspicion formed without seeing any
    documents. Confirming it needs *competing* evidence: two or more passages
    scoring near the top that answer different questions. A single dominant hit
    plus a tail of weaker, topically-adjacent passages is not ambiguity -- it is
    a clear answer with context, and asking the user to choose between readings
    the evidence has already settled is the exact failure mode that makes
    clarification prompts irritating.

    Two documents are only rivals if they are actually different documents.
    A superseded version and its successor are one document in two states, and
    they retrieve together with near-identical scores by construction -- so
    counting them as competing readings turned every version-sensitive
    question into a clarification. Observed directly: "what is the current
    Enterprise price?" asked the user to choose between "sales pricing
    documents" and "finance price lists" when the only thing that had actually
    happened was that both rate cards came back. Version conflicts are
    resolved, not asked about.
    """
    scored = [c for c in chunks if c.rank_score() > 0]
    if len(scored) < 2:
        return False

    # Collapse each superseded document onto its successor before counting.
    canonical: dict[str, str] = {}
    for conflict in conflicts:
        for superseded in conflict.superseded_doc_ids:
            canonical[superseded] = conflict.current_doc_id

    best = max(c.rank_score() for c in scored)
    rivals = [c for c in scored if c.rank_score() >= best * _RIVAL_SCORE_RATIO]
    if len(rivals) >= 2:
        if len({canonical.get(c.doc_id, c.doc_id) for c in rivals}) > 1:
            return True

    # A near-top tie is not the only shape ambiguity takes. "How much notice do
    # I need to give?" has four answers in this corpus -- PTO at 5 or 15
    # business days, contract cancellation at 30, non-renewal at 60 -- but the
    # leave section dominated the ranking, so the rival test saw one clear
    # winner and the system answered about PTO alone. When the planner has
    # already named several distinct readings, evidence drawn from more than
    # one *department* confirms them: "notice" in HR and "notice" in legal are
    # different facts, where two chunks of one policy are the same fact twice.
    if readings >= _MANY_READINGS:
        return len({c.department for c in scored if c.department}) > 1
    return False


async def answer_node(state: ChatState) -> dict:
    started = time.perf_counter()
    chunks = state.get("chunks") or []
    note = conflict_note(state.get("conflicts") or [])

    retry = state.get("retry_count", 0)
    if retry:
        # The corrective pass. Rather than re-prompting identically and hoping,
        # the failed audit is handed back as the thing to fix.
        verification = state.get("verification")
        problems = "; ".join(
            f"{c.claim} ({c.verdict})"
            for c in (verification.claims if verification else [])
            if c.verdict != "SUPPORTED"
        )
        note = (
            f"{note}\n\nA previous draft failed verification on: {problems}. "
            "Write an answer containing only statements the passages support, "
            "and cite each one. If the passages do not support an answer, say so."
        ).strip()

    draft = await answer_question(
        state["standalone_query"],
        chunks,
        history=state.get("history"),
        conflict_note=note,
    )
    return {
        "draft": draft,
        "diagnostics": {
            **state.get("diagnostics", {}),
            f"answer_ms{'_retry' if retry else ''}": round(
                (time.perf_counter() - started) * 1000
            ),
        },
    }


async def verify_node(state: ChatState) -> dict:
    started = time.perf_counter()
    draft = state["draft"]
    chunks = state.get("chunks") or []
    citation_check = check_citations(draft, chunks)
    verification = await verify(state["standalone_query"], draft, chunks)
    return {
        "verification": verification,
        "diagnostics": {
            **state.get("diagnostics", {}),
            "groundedness": round(verification.groundedness(), 3),
            "claims_supported": verification.supported,
            "claims_contradicted": verification.contradicted,
            "claims_unsupported": verification.unsupported,
            "citation_problems": citation_check.problems,
            "citation_warnings": citation_check.warnings,
            "citations_ok": citation_check.ok,
            "verify_ms": round((time.perf_counter() - started) * 1000),
        },
    }


def route_after_verify(state: ChatState) -> str:
    draft = state.get("draft")
    if draft is not None and not draft.answered:
        # The model declined. That is a valid outcome, not a verification
        # failure, and retrying it would just pressure it into answering.
        return "abstain"

    diagnostics = state.get("diagnostics", {})
    verification = state.get("verification")
    if verification is not None and passes(
        verification, diagnostics.get("citations_ok", False)
    ):
        return "respond"

    if state.get("retry_count", 0) < 1:
        return "retry"
    return "abstain"


async def retry_node(state: ChatState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


async def respond_node(state: ChatState) -> dict:
    draft = state["draft"]
    chunks = state.get("chunks") or []
    return {
        "answer": draft.answer,
        "answered": True,
        "abstained": False,
        "citations": resolve_citations(draft, chunks),
    }


async def abstain_node(state: ChatState) -> dict:
    """Say plainly that the knowledge base does not answer this.

    The reason is included because "I don't know" without a reason is
    indistinguishable from a broken system, and because the nearest documents
    are often exactly what the person needs to go read.
    """
    verdict = state.get("sufficiency")
    draft = state.get("draft")
    chunks = state.get("chunks") or []

    if draft is not None and not draft.answered:
        detail = draft.missing or "the retrieved passages do not cover it"
    elif verdict is not None:
        detail = verdict.reason
    else:
        detail = "no supporting evidence was found"

    nearest = sorted({c.doc_id for c in chunks[:3]})
    suffix = (
        f" The closest documents I found were: {', '.join(nearest)}." if nearest else ""
    )
    return {
        "answer": (
            "I don't have enough in the knowledge base to answer that "
            f"({detail}).{suffix}"
        ),
        "answered": False,
        "abstained": True,
        "citations": [],
        "confidence": verdict.score if verdict else 0.0,
    }


async def clarify_node(state: ChatState) -> dict:
    """Ask a specific question naming the options actually found.

    A bare "could you clarify?" pushes the work back onto the user. Naming the
    concrete readings the evidence contains turns it into a one-word answer.
    """
    query_plan = state.get("plan")
    readings = list(query_plan.readings) if query_plan else []
    if not readings:
        readings = sorted({c.section_path or c.title for c in (state.get("chunks") or [])[:4]})

    options = "\n".join(f"- {r}" for r in readings[:4])
    return {
        "answer": (
            "That could mean a few different things in these documents. "
            f"Which did you mean?\n{options}"
        ),
        "answered": False,
        "abstained": False,
        "clarification": "; ".join(readings[:4]),
        "citations": [],
    }


async def out_of_scope_node(state: ChatState) -> dict:
    query_plan = state.get("plan")
    reason = query_plan.reasoning if query_plan else ""
    return {
        "answer": (
            "I answer questions about the documents in this knowledge base. "
            "I can't help with that one."
        ),
        "answered": False,
        "abstained": True,
        "citations": [],
        "confidence": 0.0,
        "diagnostics": {**state.get("diagnostics", {}), "out_of_scope_reason": reason},
    }


# ---------------------------------------------------------------- assembly


def build_graph(context: RetrievalContext, *, checkpointer=None):
    """Compile the pipeline. `checkpointer` persists conversation state."""
    graph = StateGraph(ChatState)

    graph.add_node("condense", condense_node)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", make_retrieve_node(context))
    graph.add_node("expand", make_expand_node(context))
    graph.add_node("rerank", rerank_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("assess", assess_node)
    graph.add_node("answer", answer_node)
    graph.add_node("verify", verify_node)
    graph.add_node("retry", retry_node)
    graph.add_node("respond", respond_node)
    graph.add_node("abstain", abstain_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.add_edge(START, "condense")
    graph.add_edge("condense", "plan")
    graph.add_conditional_edges(
        "plan", route_after_plan, {"retrieve": "retrieve", "out_of_scope": "out_of_scope"}
    )
    graph.add_edge("retrieve", "expand")
    graph.add_edge("expand", "rerank")
    graph.add_edge("rerank", "resolve")
    graph.add_edge("resolve", "assess")
    graph.add_conditional_edges(
        "assess",
        route_after_assess,
        {"answer": "answer", "abstain": "abstain", "clarify": "clarify"},
    )
    graph.add_edge("answer", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"respond": "respond", "retry": "retry", "abstain": "abstain"},
    )
    graph.add_edge("retry", "answer")
    graph.add_edge("respond", END)
    graph.add_edge("abstain", END)
    graph.add_edge("clarify", END)
    graph.add_edge("out_of_scope", END)

    return graph.compile(checkpointer=checkpointer)


async def run_turn(
    compiled,
    message: str,
    *,
    departments: list[str],
    history: list[Turn] | None = None,
    conversation_id: str | None = None,
) -> ChatState:
    """Answer one message. `conversation_id` keys the checkpointed history."""
    started = time.perf_counter()
    config = (
        {"configurable": {"thread_id": conversation_id}} if conversation_id else None
    )
    initial: ChatState = {
        "message": message,
        "history": history or [],
        "departments": departments,
        "retry_count": 0,
        "diagnostics": {},
    }
    result = await compiled.ainvoke(initial, config=config)
    result.setdefault("diagnostics", {})["total_ms"] = round(
        (time.perf_counter() - started) * 1000
    )
    return result
