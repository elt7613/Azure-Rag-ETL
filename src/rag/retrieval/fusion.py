"""Combining evidence from several retrievers and sub-queries into one context.

Two operations, deliberately separate:

**Fusion** merges ranked lists. Reciprocal Rank Fusion is used rather than
score averaging because the lists being merged are not on comparable scales --
Azure's reranker gives 0-4, BM25 gives an unbounded relevance score, the graph
retriever gives a path-derived score. RRF only reads *positions*, so it is
immune to that mismatch, which is exactly why it is the standard choice for
hybrid retrieval.

**Expansion** widens the context around a hit. The single most common RAG
failure is retrieving the right document but the chunk beside the answer;
because chunking already recorded `prev_chunk_id`/`next_chunk_id`, pulling a
neighbour costs one keyed lookup and no extra search.
"""
from __future__ import annotations

from collections.abc import Iterable

from rag.retrieval import RetrievedChunk

# The RRF damping constant. 60 is the value from the original Cormack et al.
# formulation and the one Azure AI Search itself uses internally; keeping it
# means our cross-retriever fusion behaves like the intra-query fusion the
# service already applied, rather than introducing a second, differently-tuned
# ranking regime.
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[RetrievedChunk]], *, k: int = RRF_K
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one, scoring each chunk by sum(1 / (k + rank)).

    A chunk found by two retrievers, or by two different sub-queries,
    accumulates score from each -- which is the point: independent agreement
    is evidence, and it is the cheapest reliable relevance signal available
    without another model call.

    The merged chunk keeps the union of the retrievers and queries that found
    it and the best rank-relevant scores seen, so downstream code can still
    answer "why is this here?".
    """
    merged: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            existing = merged.get(key)
            if existing is None:
                merged[key] = chunk
                continue
            for retriever in chunk.retrievers:
                if retriever not in existing.retrievers:
                    existing.retrievers.append(retriever)
            for query in chunk.matched_queries:
                if query and query not in existing.matched_queries:
                    existing.matched_queries.append(query)
            # Keep the strongest evidence of relevance seen for this chunk.
            if chunk.reranker_score is not None and (
                existing.reranker_score is None
                or chunk.reranker_score > existing.reranker_score
            ):
                existing.reranker_score = chunk.reranker_score
            existing.score = max(existing.score, chunk.score)
            if chunk.graph_path and not existing.graph_path:
                existing.graph_path = chunk.graph_path

    for key, chunk in merged.items():
        chunk.fused_score = scores[key]

    return sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)


def dedupe_by_content(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop chunks whose text duplicates one already kept.

    Overlapping chunk windows and boilerplate repeated across documents both
    put the same sentences in the context twice. That wastes tokens and, worse,
    makes a repeated claim look corroborated when it is one source counted
    twice.
    """
    seen: set[str] = set()
    kept: list[RetrievedChunk] = []
    for chunk in chunks:
        fingerprint = " ".join(chunk.content.split()).lower()
        if fingerprint and fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(chunk)
    return kept


def neighbour_ids(chunks: list[RetrievedChunk], *, limit: int = 6) -> list[str]:
    """Chunk ids adjacent to the given hits that are not already among them.

    Capped, because expansion is a widening heuristic and an uncapped one turns
    a focused context into the whole document.
    """
    present = {c.chunk_id for c in chunks}
    wanted: list[str] = []
    for chunk in chunks:
        for neighbour in (chunk.prev_chunk_id, chunk.next_chunk_id):
            if neighbour and neighbour not in present and neighbour not in wanted:
                wanted.append(neighbour)
                if len(wanted) >= limit:
                    return wanted
    return wanted


def merge_expansion(
    primary: list[RetrievedChunk], expanded: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Append neighbour chunks below the primary hits, never above them.

    A neighbour earned its place by adjacency, not by relevance, so it must not
    outrank a chunk the retrievers actually chose. Ordering matters here
    because the answer prompt reads the context top-down.
    """
    present = {c.chunk_id for c in primary}
    tail: list[RetrievedChunk] = []
    for chunk in expanded:
        if chunk.chunk_id in present:
            continue
        present.add(chunk.chunk_id)
        chunk.retrievers = [*chunk.retrievers, "neighbour"]
        tail.append(chunk)
    return [*primary, *tail]


def top_n(chunks: list[RetrievedChunk], n: int) -> list[RetrievedChunk]:
    return chunks[:n]


# How much of the final ordering the cross-encoder owns. It is the strongest
# single relevance signal available -- it reads the query against the passage
# rather than comparing two independently-computed vectors -- so it leads. RRF
# keeps a smaller share because agreement between independent retrievers is
# evidence the cross-encoder cannot see: it only ever looks at one passage at a
# time.
_RERANK_WEIGHT = 0.85
_FUSION_WEIGHT = 0.15

# Azure's semantic ranker scores on a fixed 0-4 scale, so its scores are used
# **absolutely** rather than min-max normalised. This matters more than it
# looks: min-maxing two candidates scoring 2.98 and 3.00 turns a negligible
# difference into a maximal one, and the fusion component -- which should decide
# a near-tie -- never gets a say. An absolute scale keeps "these two are
# equivalent" and "one of these is far better" distinguishable.
_RERANKER_SCALE = 4.0


def _relative(values: list[float]) -> dict[int, float]:
    """Scale values against the largest, keyed by position.

    Used for fused scores, which -- unlike the reranker's -- have no meaningful
    absolute scale, only relative magnitude.
    """
    if not values:
        return {}
    high = max(values)
    if high <= 0:
        return {i: 0.0 for i in range(len(values))}
    return {i: v / high for i, v in enumerate(values)}


def rerank(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Re-order fused candidates so the cross-encoder's opinion leads.

    Fusion selects *which* candidates survive; it should not decide their final
    order on its own. RRF reads positions only, so two retrievers agreeing on a
    mediocre chunk can outrank one strong hit that only one retriever found --
    and the cross-encoder's score, the best evidence available about whether a
    passage answers the question, is discarded entirely.

    Measured on this corpus: "What are the payment terms in the vendor service
    agreement?" ranks `VendorContract § 3 Payment Terms` **first at 2.95**, with
    the rest of the document below 2.15. After fusion it fell out of the context
    window altogether, and the system abstained on a question the corpus answers
    in one sentence -- the classic right-document-wrong-chunk failure, produced
    by the ranking layer rather than by retrieval.

    Chunks with no cross-encoder score -- graph-only hits, which were never sent
    to the ranker -- fall back to their fused score for both components rather
    than being scored zero. They were found by a different kind of evidence, and
    demoting them for not having been measured on this scale would silently turn
    the graph retriever off.
    """
    if len(chunks) < 2:
        return list(chunks)

    fused = _relative([c.fused_score for c in chunks])
    scores: dict[int, float] = {}
    for i, chunk in enumerate(chunks):
        relevance = (
            fused.get(i, 0.0)
            if chunk.reranker_score is None
            else min(1.0, max(0.0, chunk.reranker_score / _RERANKER_SCALE))
        )
        scores[i] = _RERANK_WEIGHT * relevance + _FUSION_WEIGHT * fused.get(i, 0.0)

    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in order]
