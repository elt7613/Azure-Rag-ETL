"""Entity-anchored retrieval over the knowledge graph."""
from __future__ import annotations

import pytest

from tests.conftest import azure_configured

from rag.retrieval.graph_retriever import GraphRetriever, candidate_mentions
from rag.retrieval.searcher import HybridSearcher

def _neo4j_configured() -> bool:
    return azure_configured("neo4j_uri", "neo4j_user", "neo4j_password")


live = pytest.mark.skipif(
    not (azure_configured() and _neo4j_configured()),
    reason="Azure or Neo4j not configured",
)


# ---------------- mention extraction (pure) ----------------


def test_multi_word_proper_phrases_are_kept_whole():
    mentions = candidate_mentions("What is the Business Plus price per seat?")
    assert "business plus" in mentions
    # The parts are still available as fallbacks.
    assert "business" in mentions


def test_sentence_initial_capital_does_not_invent_a_phrase():
    """"Which MFA factor..." must not yield the phrase "which mfa"."""
    mentions = candidate_mentions(
        "Which MFA factor is required for IT Administrator accounts?"
    )
    assert "which mfa" not in mentions
    assert "it administrator" in mentions or "administrator" in mentions


def test_stopwords_and_short_tokens_are_dropped():
    mentions = candidate_mentions("what about it?")
    assert mentions == []


def test_units_of_measure_and_modals_are_not_entities():
    mentions = candidate_mentions("How much notice must I give per request?")
    for noise in ("much", "must", "per", "give"):
        assert noise not in mentions


def test_mentions_are_capped():
    long_query = " ".join(f"Term{i}" for i in range(40))
    assert len(candidate_mentions(long_query)) <= 8


# ---------------- live retrieval ----------------


@pytest.fixture
async def retriever():
    searcher = HybridSearcher()
    graph = GraphRetriever(searcher)
    try:
        yield graph
    finally:
        await graph.aclose()
        await searcher.aclose()


@live
async def test_ping_reaches_neo4j(retriever):
    assert await retriever.ping() is True


@live
async def test_anchors_resolve_to_real_entities(retriever, facts):
    anchors = await retriever.find_anchors(
        f"{facts.leave_term} accrual", departments=facts.departments
    )
    assert anchors


@live
async def test_exact_name_matches_rank_before_substring_matches(retriever, facts):
    anchors = await retriever.find_anchors(
        "sick leave", departments=facts.departments
    )
    assert anchors
    # An exactly-named entity must be present, not buried behind coincidental
    # substring hits.
    assert any("sick leave" in a["name"].lower() for a in anchors)


@live
async def test_two_lowercase_words_agreeing_on_one_entity_is_a_valid_anchor(
    retriever, facts
):
    """The rule that makes lowercase multi-word questions work.

    "parental leave" has no capitalised phrase to extract and neither word
    matches an entity name exactly — but both land on the same entity, and
    independent agreement between query words is evidence. Chosen because both
    corpora document parental leave in the same file, so the rule is tested
    rather than the corpus.
    """
    hits = await retriever.retrieve("parental leave", departments=facts.departments, top=5)
    assert hits, "a two-word agreeing anchor produced no hits"
    assert hits[0].doc_id == facts.leave_doc


@live
async def test_anchor_credit_is_weighted_by_how_much_of_the_query_agrees(
    retriever, facts
):
    """A chunk matching a two-word anchor must outscore a single direct mention.

    This replaces a corpus-specific test that asserted `Tuition Reimbursement`
    outranked a bare `reimbursement` entity: the mechanism is the same, but the
    entity names were particular to one corpus. Unweighted, a mention of the
    generic entity scored the same as one of the specific entity, and unrelated
    sections tied with the section that answers the question. A score above a
    single direct credit is the observable proof that the weighting fired.
    """
    from rag.retrieval.graph_retriever import _DIRECT_SCORE

    hits = await retriever.retrieve("parental leave", departments=facts.departments, top=5)
    assert hits
    assert hits[0].score > _DIRECT_SCORE


@live
async def test_retrieval_returns_chunks_ranked_by_anchor_agreement(retriever, facts):
    hits = await retriever.retrieve(
        f"What is the {facts.top_tier} tier price?",
        departments=facts.departments, top=5,
    )
    assert len(hits) >= 2
    assert hits[0].doc_id == facts.pricing_current
    # A chunk mentioning more of the query's entities outranks one mentioning
    # fewer, and the superseded 2025 card ranks below the current one.
    assert hits[0].score > hits[-1].score
    assert all(h.retrievers == ["graph"] for h in hits)


@live
async def test_a_generic_word_does_not_anchor_by_coincidence(retriever, facts):
    """"schedule" sits inside "Discount Schedule" and used to anchor on it."""
    hits = await retriever.retrieve(
        "quarterly submarine maintenance schedule", departments=facts.departments, top=5
    )
    assert hits == []


@live
async def test_every_hit_explains_its_path(retriever, facts):
    """"Why is this chunk here?" must be answerable for a graph result."""
    hits = await retriever.retrieve(
        f"{facts.leave_term} accrual", departments=facts.departments, top=5
    )
    assert hits
    assert all(h.graph_path for h in hits)


@live
async def test_department_scope_is_enforced_in_the_cypher(retriever, facts):
    """An HR question asked by an IT-scoped caller must not reach HR entities."""
    unscoped = await retriever.retrieve(
        f"{facts.leave_term} accrual", departments=facts.departments, top=5
    )
    assert any(h.doc_id.startswith("HR/") for h in unscoped)

    scoped = await retriever.retrieve(
        f"{facts.leave_term} accrual", departments=[facts.other_department], top=5
    )
    assert all(h.department == "IT" for h in scoped)
    assert all(not h.doc_id.startswith("HR/") for h in scoped)


@live
async def test_empty_scope_retrieves_nothing(retriever, facts):
    assert await retriever.retrieve(facts.leave_term, departments=[], top=5) == []


@live
async def test_a_query_naming_no_known_entity_returns_nothing(retriever, facts):
    """Contributing nothing is a normal outcome for one retriever among several.

    The graph adds what vector search misses; it does not duplicate it, and
    returning weak matches would only add noise for fusion to demote.
    """
    hits = await retriever.retrieve(
        "photosynthesis in marine phytoplankton", departments=facts.departments, top=5
    )
    assert hits == []


@live
async def test_successors_of_finds_the_replacing_document(retriever, facts):
    """Text similarity cannot find this: the successor may share no wording."""
    hits = await retriever.successors_of(
        [facts.pricing_superseded], departments=["sales"], limit=5
    )
    assert hits
    assert all(h.doc_id == facts.pricing_current for h in hits)
    assert all("supersedes" in h.retrievers[0] for h in hits)
