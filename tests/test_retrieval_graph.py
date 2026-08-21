"""The LangGraph pipeline, end to end against the live services."""
from __future__ import annotations

import pytest

from tests.conftest import azure_configured

from rag.agents.condense import Turn
from rag.agents.graph_app import RetrievalContext, build_graph, run_turn
from rag.retrieval.searcher import HybridSearcher

live = pytest.mark.skipif(
    not azure_configured(), reason="Azure credentials not configured"
)


@pytest.fixture
async def pipeline():
    searcher = HybridSearcher()
    compiled = build_graph(RetrievalContext(searcher))
    try:
        yield compiled
    finally:
        await searcher.aclose()


@live
async def test_simple_question_is_answered_with_citations(pipeline, facts):
    state = await run_turn(
        pipeline,
        "How many days of paid sick leave do employees get each year?",
        departments=facts.departments,
    )
    assert state["answered"] is True
    assert facts.sick_leave_days in state["answer"]
    assert state["citations"]
    assert any(c["doc_id"] == facts.leave_doc for c in state["citations"])
    # A single-fact lookup must not pay for decomposition.
    assert state["diagnostics"]["plan_kind"] == "SIMPLE"


@live
async def test_answer_cites_a_resolvable_source(pipeline, facts):
    state = await run_turn(
        pipeline,
        facts.learning_query,
        departments=facts.departments,
    )
    assert state["answered"] is True
    assert facts.learning_limit in state["answer"]
    for citation in state["citations"]:
        assert citation["chunk_id"]
        assert citation["doc_id"]


@live
async def test_out_of_scope_question_never_reaches_retrieval(pipeline, facts):
    state = await run_turn(pipeline, "Hi, how are you today?",
                           departments=facts.departments)
    assert state["answered"] is False
    assert state["diagnostics"]["plan_kind"] == "OUT_OF_SCOPE"
    # Nothing was searched for, so nothing was spent on it.
    assert "queries_run" not in state["diagnostics"]


@live
async def test_question_with_no_answer_in_the_corpus_abstains(pipeline, facts):
    state = await run_turn(
        pipeline, facts.unanswerable, departments=facts.departments
    )
    assert state["answered"] is False
    assert state["abstained"] is True
    # The refusal explains itself rather than being a bare "I don't know".
    assert len(state["answer"]) > 40


@live
async def test_multi_part_question_is_decomposed(pipeline, facts):
    state = await run_turn(
        pipeline,
        f"Compare the {facts.entry_tier} and {facts.top_tier} subscription tiers.",
        departments=facts.departments,
    )
    assert state["diagnostics"]["plan_kind"] == "MULTI_PART"
    assert len(state["diagnostics"]["queries_run"]) > 1
    assert state["answered"] is True


@live
async def test_department_scope_is_enforced_end_to_end(pipeline, facts):
    """A caller scoped elsewhere must not receive HR content, whatever they ask."""
    state = await run_turn(
        pipeline,
        "How many days of paid sick leave do employees get each year?",
        departments=[facts.other_department],
    )
    assert all(c["department"] == facts.other_department for c in state["citations"])
    assert all("HR/" not in c["doc_id"] for c in state["citations"])


@live
async def test_empty_department_scope_retrieves_nothing(pipeline):
    state = await run_turn(pipeline, "What is the leave accrual rate?", departments=[])
    assert state["answered"] is False


@live
async def test_follow_up_is_resolved_against_history(pipeline, facts):
    history = [
        Turn(role="user",
             content=f"What is the {facts.top_tier} tier price per seat?"),
        Turn(role="assistant",
             content=f"{facts.top_tier} is ${facts.top_tier_price} per seat per month."),
    ]
    state = await run_turn(
        pipeline, f"What about {facts.entry_tier}?",
        departments=facts.departments, history=history,
    )
    assert state["diagnostics"]["condensed"] is True
    assert facts.entry_tier.lower() in state["diagnostics"]["standalone_query"].lower()
    assert state["answered"] is True
    assert facts.entry_tier_price in state["answer"]


@live
async def test_verification_runs_on_every_answered_turn(pipeline, facts):
    state = await run_turn(
        pipeline, facts.leave_query, departments=facts.departments,
    )
    assert state["answered"] is True
    assert "groundedness" in state["diagnostics"]
    assert state["diagnostics"]["groundedness"] >= 0.8
    assert state["diagnostics"]["claims_contradicted"] == 0


@live
async def test_diagnostics_expose_the_whole_path(pipeline, facts):
    """Every stage must be traceable — this is the debugging contract."""
    state = await run_turn(
        pipeline, "What receipts are required for expenses?",
        departments=facts.departments,
    )
    diagnostics = state["diagnostics"]
    for key in (
        "plan_kind", "queries_run", "candidates", "sufficiency",
        "sufficiency_reason", "retrieve_ms", "total_ms",
    ):
        assert key in diagnostics, f"missing diagnostic: {key}"
