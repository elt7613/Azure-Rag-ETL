"""Hybrid searcher: filter construction (pure) and live retrieval quality."""
from __future__ import annotations

import pytest

from tests.conftest import azure_configured

from rag.retrieval.searcher import HybridSearcher, build_filter


# ---------------- filter construction (no network) ----------------


def test_filter_denies_when_no_department_scope():
    # The critical property: no scope means no results, never all results.
    assert build_filter(None) == "department eq ''"
    assert build_filter([]) == "department eq ''"


def test_filter_scopes_to_requested_departments():
    assert build_filter(["HR", "finance"]) == "search.in(department, 'HR,finance', ',')"


def test_filter_escapes_single_quotes():
    # A department name with an apostrophe must not break out of the literal.
    assert build_filter(["O'Brien"]) == "search.in(department, 'O''Brien', ',')"


def test_filter_injection_attempt_stays_inside_the_literal():
    hostile = "x' or department ne '"
    built = build_filter([hostile])
    assert built == "search.in(department, 'x'' or department ne ''', ',')"
    # No unescaped quote can terminate the literal early.
    assert built.count("'") % 2 == 0


def test_filter_composes_optional_clauses():
    built = build_filter(["sales"], current_only=True, content_types=["table"])
    assert "is_current eq true" in built
    assert "search.in(content_type, 'table', ',')" in built
    assert built.startswith("search.in(department, 'sales', ',')")


# ---------------- live retrieval ----------------


@pytest.fixture
async def searcher():
    s = HybridSearcher()
    try:
        yield s
    finally:
        await s.aclose()


@pytest.mark.skipif(not azure_configured(), reason="Azure credentials not configured")
async def test_hybrid_search_finds_the_right_section(searcher, facts):
    hits = await searcher.search(facts.leave_query, departments=["HR"], top=5)
    assert hits, "hybrid search returned nothing"
    assert any(h.doc_id == facts.leave_doc for h in hits)
    # The accrual table is the chunk that actually answers this.
    assert any("Annual Accrual" in h.content for h in hits)


@pytest.mark.skipif(not azure_configured(), reason="Azure credentials not configured")
async def test_semantic_ranking_populates_reranker_score(searcher, facts):
    hits = await searcher.search(
        "annual leave entitlement", departments=["HR"], top=3, semantic=True
    )
    assert hits
    assert all(h.reranker_score is not None for h in hits)
    # Azure's L2 ranker scores on a 0-4 scale.
    assert all(0.0 <= h.reranker_score <= 4.0 for h in hits)
    # Results arrive already ordered by the reranker.
    scores = [h.reranker_score for h in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.skipif(not azure_configured(), reason="Azure credentials not configured")
async def test_exact_token_query_is_found_by_the_keyword_side(searcher, facts):
    """The case pure vector search loses: a literal amount.

    "$5,250" is a token, not a concept. BM25 finds it; embeddings blur it
    into every other dollar figure in the corpus. This is why the searcher is
    hybrid rather than vector-only.
    """
    hits = await searcher.search("$5,250", departments=["HR"], top=5)
    assert any("5,250" in h.content for h in hits)


@pytest.mark.skipif(not azure_configured(), reason="Azure credentials not configured")
async def test_department_scope_is_enforced_server_side(searcher, facts):
    hits = await searcher.search(
        f"leave policy {facts.leave_term} accrual",
        departments=[facts.other_department], top=10
    )
    # Every hit must belong to the requested department -- an HR document
    # reaching a differently-scoped caller is what this filter exists to prevent.
    assert all(h.department == facts.other_department for h in hits)
    assert all(not h.doc_id.startswith("HR/") for h in hits)


@pytest.mark.skipif(not azure_configured(), reason="Azure credentials not configured")
async def test_empty_scope_retrieves_nothing(searcher):
    hits = await searcher.search("PTO accrual", departments=[], top=10)
    assert hits == []


@pytest.mark.skipif(not azure_configured(), reason="Azure credentials not configured")
async def test_superseded_documents_remain_retrievable(searcher, facts):
    """A superseded document stays in the index on purpose.

    The 2025 rate card still governs a contract signed in 2025, so version is
    a ranking and labelling signal, never a hard exclusion. Whether a given
    query *should* prefer the current version is conflict resolution's job,
    not the searcher's.

    Asserted on retrievability rather than on rank position. An earlier version
    of this test required both rate cards inside the top 10 of a competitive
    query, which conflates "is retrievable" with "outranks everything else in
    its department" -- it passed alone and failed under a full-suite run, which
    is the worst kind of flake because it points at the wrong thing.
    """
    hits = await searcher.search(
        f"{facts.top_tier} price per seat",
        departments=["sales"],
        top=10,
        doc_ids=[facts.pricing_superseded, facts.pricing_current],
    )
    found = {h.doc_id for h in hits}
    assert facts.pricing_superseded in found, "superseded document was filtered out"
    assert facts.pricing_current in found

    superseded = next(h for h in hits if h.doc_id == facts.pricing_superseded)
    assert superseded.is_current is False
    assert superseded.superseded_by == facts.pricing_current


@pytest.mark.skipif(not azure_configured(), reason="Azure credentials not configured")
async def test_current_only_filter_is_accepted_by_the_service(searcher, facts):
    """The `is_current` filter is a valid, executable query.

    Whether it actually excludes the 2025 rate card depends on the corpus-wide
    version reconciliation that stamps `is_current`/`superseded_by` onto the
    indexed chunks -- covered by that pass's own end-to-end test, not here.
    """
    hits = await searcher.search(
        "subscription pricing", departments=["sales"], top=10, current_only=True
    )
    assert all(h.is_current for h in hits)


@pytest.mark.skipif(not azure_configured(), reason="Azure credentials not configured")
async def test_fetch_chunks_round_trips_and_stays_scoped(searcher, facts):
    hits = await searcher.search("sick leave days", departments=["HR"], top=3)
    assert hits
    ids = [h.chunk_id for h in hits]

    same = await searcher.fetch_chunks(ids, departments=["HR"])
    assert {c.chunk_id for c in same} == set(ids)

    # The same ids requested by a caller scoped elsewhere return nothing:
    # holding a chunk id is not authorisation to read it.
    denied = await searcher.fetch_chunks(ids, departments=[facts.other_department])
    assert denied == []
