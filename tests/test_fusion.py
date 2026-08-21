"""RRF fusion, dedupe, and neighbour expansion — all pure."""
from __future__ import annotations

from rag.retrieval import RetrievedChunk
from rag.retrieval.fusion import (
    RRF_K,
    dedupe_by_content,
    merge_expansion,
    neighbour_ids,
    reciprocal_rank_fusion,
)


def chunk(cid: str, *, content: str = "", retriever: str = "vector",
          prev: str = "", nxt: str = "", reranker: float | None = None,
          score: float = 0.0, query: str = "q") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id="d", title="t", department="HR",
        section_path="s", content=content or cid, page=1,
        prev_chunk_id=prev, next_chunk_id=nxt,
        reranker_score=reranker, score=score,
        retrievers=[retriever], matched_queries=[query],
    )


def test_rrf_scores_by_position_not_by_score_magnitude():
    # Two lists on wildly different score scales. RRF must ignore the scales.
    vector = [chunk("a", score=0.03), chunk("b", score=0.02)]
    graph = [chunk("b", retriever="graph", score=980.0),
             chunk("a", retriever="graph", score=12.0)]

    fused = reciprocal_rank_fusion([vector, graph])

    # a: 1/(60+1) + 1/(60+2); b: 1/(60+2) + 1/(60+1) — identical.
    assert {c.chunk_id for c in fused} == {"a", "b"}
    assert abs(fused[0].fused_score - fused[1].fused_score) < 1e-12


def test_agreement_between_retrievers_outranks_a_single_top_hit():
    vector = [chunk("solo"), chunk("agreed")]
    graph = [chunk("agreed", retriever="graph")]

    fused = reciprocal_rank_fusion([vector, graph])

    # "agreed" is 2nd in one list but found by both; "solo" is 1st in one list.
    assert fused[0].chunk_id == "agreed"
    assert fused[0].fused_score > fused[1].fused_score


def test_rrf_formula_matches_the_definition():
    fused = reciprocal_rank_fusion([[chunk("a"), chunk("b"), chunk("c")]])
    expected = [1.0 / (RRF_K + rank) for rank in (1, 2, 3)]
    assert [round(c.fused_score, 12) for c in fused] == [round(e, 12) for e in expected]


def test_merged_chunk_keeps_provenance_from_every_source():
    vector = [chunk("x", query="original question")]
    graph = [chunk("x", retriever="graph", query="decomposed sub-question")]

    fused = reciprocal_rank_fusion([vector, graph])

    assert fused[0].retrievers == ["vector", "graph"]
    assert fused[0].matched_queries == ["original question", "decomposed sub-question"]


def test_merged_chunk_keeps_the_strongest_reranker_score():
    fused = reciprocal_rank_fusion([[chunk("x", reranker=1.1)],
                                    [chunk("x", reranker=3.4)]])
    assert fused[0].reranker_score == 3.4


def test_empty_input_fuses_to_nothing():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_dedupe_drops_identical_text_regardless_of_whitespace():
    kept = dedupe_by_content([
        chunk("a", content="Employees accrue 15 days."),
        chunk("b", content="Employees   accrue\n15 days."),
        chunk("c", content="Employees accrue 20 days."),
    ])
    assert [c.chunk_id for c in kept] == ["a", "c"]


def test_dedupe_keeps_the_first_occurrence():
    kept = dedupe_by_content([chunk("first", content="same"),
                              chunk("second", content="same")])
    assert [c.chunk_id for c in kept] == ["first"]


def test_neighbour_ids_excludes_chunks_already_retrieved():
    hits = [chunk("b", prev="a", nxt="c"), chunk("c", prev="b", nxt="d")]
    assert neighbour_ids(hits) == ["a", "d"]


def test_neighbour_ids_is_capped():
    hits = [chunk(f"c{i}", prev=f"p{i}", nxt=f"n{i}") for i in range(10)]
    assert len(neighbour_ids(hits, limit=3)) == 3


def test_expansion_never_outranks_a_real_hit():
    primary = [chunk("hit1"), chunk("hit2")]
    expanded = [chunk("neighbour1"), chunk("hit1")]

    merged = merge_expansion(primary, expanded)

    assert [c.chunk_id for c in merged] == ["hit1", "hit2", "neighbour1"]
    assert "neighbour" in merged[-1].retrievers
    # The duplicate of an existing hit is not re-added.
    assert [c.chunk_id for c in merged].count("hit1") == 1


# ---------------- reranking after fusion ----------------


def test_rerank_lets_the_cross_encoder_lead():
    """The bug this exists to prevent, reproduced.

    Two retrievers agreeing on a mediocre chunk accumulate RRF score and
    outrank a strong hit only one retriever found — and the cross-encoder's
    score, the best evidence available, is discarded. Measured on the real
    corpus this dropped `VendorContract § 3 Payment Terms` (2.95, ranked first
    by the ranker) out of the context entirely, and the system abstained on a
    question answered in one sentence.
    """
    from rag.retrieval.fusion import rerank

    strong = chunk("payment-terms", reranker=2.95)
    strong.fused_score = 1 / (RRF_K + 1)                 # found by one retriever
    agreed = chunk("liability", reranker=2.13)
    agreed.fused_score = 1 / (RRF_K + 2) + 1 / (RRF_K + 1)  # found by two

    assert agreed.fused_score > strong.fused_score        # RRF prefers the weaker chunk
    assert rerank([agreed, strong])[0].chunk_id == "payment-terms"


def test_rerank_still_gives_fusion_a_say():
    """Agreement between retrievers is evidence the cross-encoder cannot see."""
    from rag.retrieval.fusion import rerank

    lone = chunk("lone", reranker=3.0)
    lone.fused_score = 0.001
    agreed = chunk("agreed", reranker=2.98)
    agreed.fused_score = 0.05

    # Near-identical cross-encoder scores, so the fusion component decides.
    assert rerank([lone, agreed])[0].chunk_id == "agreed"


def test_graph_only_chunks_are_not_demoted_for_lacking_a_reranker_score():
    """They were never sent to the ranker; scoring them zero would mute the graph."""
    from rag.retrieval.fusion import rerank

    graph_hit = chunk("graph", retriever="graph", reranker=None)
    graph_hit.fused_score = 0.05
    weak = chunk("weak", reranker=0.4)
    weak.fused_score = 0.001

    assert rerank([weak, graph_hit])[0].chunk_id == "graph"


def test_rerank_is_a_no_op_on_trivial_input():
    from rag.retrieval.fusion import rerank

    assert rerank([]) == []
    single = [chunk("only")]
    assert [c.chunk_id for c in rerank(single)] == ["only"]
