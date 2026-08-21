"""Retrieval layer.

Every retriever -- vector, graph, or anything added later -- returns the same
`RetrievedChunk` shape, so fusion, reranking, conflict resolution and context
assembly downstream never have to care where a piece of evidence came from.
That uniformity is what lets a graph hit and a vector hit be ranked against
each other honestly instead of being stapled together at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class RetrievedChunk:
    """One piece of evidence, with everything needed to rank it and cite it."""

    chunk_id: str
    doc_id: str
    title: str
    department: str
    section_path: str
    content: str
    page: int
    content_type: str = "prose"

    # Version/recency signals, used by conflict resolution rather than by the
    # search engine: a superseded document is still retrievable, because the
    # 2025 rate card still governs a contract signed in 2025.
    version: str = ""
    is_current: bool = True
    superseded_by: str = ""
    effective_from: date | None = None
    effective_to: date | None = None

    # Neighbour links, used to widen context when the right document was found
    # but the retrieved chunk sits just beside the answer.
    prev_chunk_id: str = ""
    next_chunk_id: str = ""

    # Scores. `score` is whatever the originating retriever produced (BM25/RRF
    # for search, path relevance for the graph); `reranker_score` is Azure's
    # semantic ranker on its 0-4 scale, present only when semantic ranking ran;
    # `fused_score` is filled in by RRF fusion across retrievers.
    score: float = 0.0
    reranker_score: float | None = None
    fused_score: float = 0.0

    # Provenance of the retrieval itself -- which retriever(s) surfaced this,
    # and via which sub-query. Kept because "why did this chunk end up in the
    # context?" is the first question when an answer goes wrong.
    retrievers: list[str] = field(default_factory=list)
    matched_queries: list[str] = field(default_factory=list)
    graph_path: str = ""

    def citation(self) -> str:
        """Human-readable source reference for the answer text."""
        where = f" § {self.section_path}" if self.section_path else ""
        page = f", p.{self.page}" if self.page else ""
        return f"{self.doc_id}{where}{page}"

    def rank_score(self) -> float:
        """The score to sort by: semantic rerank when present, else raw score."""
        return self.reranker_score if self.reranker_score is not None else self.score


__all__ = ["RetrievedChunk"]
