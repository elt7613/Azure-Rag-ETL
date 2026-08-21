"""Vector-side retrieval: hybrid search with semantic reranking over Azure AI Search.

Three deliberate choices here, each answering a specific way naive RAG fails:

**Hybrid, not vector-only.** Pure vector search loses on exact tokens -- a
policy number, "$5,250", "Enterprise Plus", "Okta Verify". BM25 finds those and
embeddings find paraphrases; Azure fuses the two ranked lists with Reciprocal
Rank Fusion. This is the single biggest fix for "the right document came back
but the wrong chunk did".

**Semantic reranking over 50 candidates.** Azure's L2 ranker rescores the top
50 fused candidates with a cross-encoder on a 0-4 scale, which is a genuinely
different judgement from cosine similarity -- it reads the query against the
passage rather than comparing two independently-computed vectors. `vector_k`
defaults to 50 for exactly this reason: fewer candidates and the reranker has
nothing to work with.

**Filters are applied server-side, always.** Department scoping is an OData
filter in the query, not a post-processing step. Filtering after the fact means
the content has already crossed into the process that was not allowed to see it.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery

from rag.config import get_settings
from rag.embedding.azure_openai import AzureOpenAIEmbedder
from rag.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

_SELECT = [
    "chunk_id", "doc_id", "title", "department", "section_path", "section_number",
    "version", "effective_from", "effective_to", "is_current", "superseded_by",
    "content_type", "page", "prev_chunk_id", "next_chunk_id", "content",
]


def _escape_odata(value: str) -> str:
    """OData string literals escape a single quote by doubling it.

    Without this a department named `O'Brien Group` -- or a hostile one chosen
    to look like `x' or department ne '` -- would break out of the literal and
    rewrite the filter. Every user-supplied value reaching a filter goes
    through here.
    """
    return value.replace("'", "''")


def build_filter(
    departments: list[str] | None,
    *,
    doc_ids: list[str] | None = None,
    current_only: bool = False,
    content_types: list[str] | None = None,
) -> str:
    """Compose the OData filter for a query.

    Department scoping is mandatory and deny-by-default: an empty or missing
    scope produces a filter that matches nothing, rather than one that matches
    everything. A retrieval layer whose access control fails open is worse than
    one with no access control, because it looks safe.
    """
    clauses: list[str] = []

    if not departments:
        return "department eq ''"  # matches nothing; deny by default
    quoted = ",".join(_escape_odata(d) for d in departments)
    clauses.append(f"search.in(department, '{quoted}', ',')")

    if doc_ids:
        ids = ",".join(_escape_odata(d) for d in doc_ids)
        clauses.append(f"search.in(doc_id, '{ids}', ',')")
    if content_types:
        types = ",".join(_escape_odata(c) for c in content_types)
        clauses.append(f"search.in(content_type, '{types}', ',')")
    if current_only:
        clauses.append("is_current eq true")

    return " and ".join(clauses)


def _parse_date(value) -> date | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _to_chunk(doc: dict, query: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=doc["chunk_id"],
        doc_id=doc.get("doc_id", ""),
        title=doc.get("title", ""),
        department=doc.get("department", ""),
        section_path=doc.get("section_path", ""),
        content=doc.get("content", ""),
        page=doc.get("page") or 0,
        content_type=doc.get("content_type", "prose"),
        version=doc.get("version", "") or "",
        is_current=bool(doc.get("is_current", True)),
        superseded_by=doc.get("superseded_by", "") or "",
        effective_from=_parse_date(doc.get("effective_from")),
        effective_to=_parse_date(doc.get("effective_to")),
        prev_chunk_id=doc.get("prev_chunk_id", "") or "",
        next_chunk_id=doc.get("next_chunk_id", "") or "",
        score=float(doc.get("@search.score") or 0.0),
        reranker_score=(
            float(doc["@search.reranker_score"])
            if doc.get("@search.reranker_score") is not None
            else None
        ),
        retrievers=["vector"],
        matched_queries=[query],
    )


class HybridSearcher:
    """Hybrid + semantically-reranked retrieval against the chunk index.

    Owns its `SearchClient` for its lifetime and closes it once, matching
    `AzureSearchSink`'s lifecycle: the underlying aiohttp transport cannot be
    reopened after close, so a per-call `async with` would break every call
    after the first.
    """

    def __init__(self, embedder: AzureOpenAIEmbedder | None = None) -> None:
        settings = get_settings()
        self._settings = settings
        self._embedder = embedder or AzureOpenAIEmbedder()
        self._semantic_config = settings.azure_search_semantic_config
        self._client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index,
            credential=AzureKeyCredential(settings.azure_search_key),
        )

    async def search(
        self,
        query: str,
        *,
        departments: list[str] | None,
        top: int | None = None,
        vector_k: int | None = None,
        semantic: bool = True,
        current_only: bool = False,
        doc_ids: list[str] | None = None,
        content_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Run one hybrid query and return ranked evidence.

        `semantic=False` gives the plain RRF hybrid ranking; it exists so the
        evaluation harness can measure what the semantic ranker is actually
        worth on this corpus rather than assuming it.
        """
        settings = self._settings
        top = top or settings.retrieval_top_k
        vector_k = vector_k or settings.vector_k
        odata = build_filter(
            departments,
            doc_ids=doc_ids,
            current_only=current_only,
            content_types=content_types,
        )

        vector = await self._embedder.embed_one(query)
        kwargs: dict = {
            "search_text": query,
            "vector_queries": [
                VectorizedQuery(
                    vector=vector, k_nearest_neighbors=vector_k, fields="content_vector"
                )
            ],
            "filter": odata,
            "select": _SELECT,
            "top": top,
        }
        if semantic:
            kwargs["query_type"] = "semantic"
            kwargs["semantic_configuration_name"] = self._semantic_config

        try:
            results = await self._client.search(**kwargs)
            return [_to_chunk(doc, query) async for doc in results]
        except Exception:
            if not semantic:
                raise
            # A semantic-ranking failure (quota exhausted, config missing,
            # transient service error) must degrade to plain hybrid rather than
            # failing the user's question outright. Ranking gets worse; the
            # system keeps answering.
            logger.warning(
                "Semantic ranking failed for %r; falling back to hybrid RRF", query,
                exc_info=True,
            )
            kwargs.pop("query_type", None)
            kwargs.pop("semantic_configuration_name", None)
            results = await self._client.search(**kwargs)
            return [_to_chunk(doc, query) async for doc in results]

    async def fetch_chunks(
        self, chunk_ids: list[str], *, departments: list[str] | None
    ) -> list[RetrievedChunk]:
        """Look up specific chunks by id, still department-scoped.

        Used by neighbour expansion and by the graph retriever, which finds
        chunk ids in Neo4j and needs their full text and metadata. The
        department filter is reapplied here on purpose: an id arriving from
        another subsystem is not a licence to bypass access control.
        """
        if not chunk_ids:
            return []
        allowed = {d.lower() for d in (departments or [])}
        if not allowed:
            return []

        # `chunk_id` is the index key, and Azure AI Search does not allow the
        # key field in a $filter expression -- so this is a direct keyed
        # lookup rather than a filtered query. That makes the department check
        # a post-condition here instead of a server-side filter, which is why
        # it is enforced explicitly below and why callers still pass a scope:
        # a chunk id arriving from the graph is not authorisation to read it.
        async def _one(chunk_id: str) -> dict | None:
            try:
                return await self._client.get_document(key=chunk_id, selected_fields=_SELECT)
            except ResourceNotFoundError:
                return None

        docs = await asyncio.gather(*(_one(c) for c in chunk_ids))
        return [
            _to_chunk(doc, "")
            for doc in docs
            if doc is not None and (doc.get("department") or "").lower() in allowed
        ]

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "HybridSearcher":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
