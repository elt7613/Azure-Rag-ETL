"""Azure AI Search index definition and upsert sink.

CocoIndex has no built-in Azure AI Search target and no public custom-target
SDK, so this is a plain sink invoked from a memoized `@coco.fn`. Consequence:
writes are memoized but NOT auto-reconciled — deletions are handled explicitly
via `delete_document`, which the ETL calls on source removal.

The configured Azure AI Search service is serverless (Basic tier equivalent
with no reserved compute). Serverless services reject `list_indexes()` /
`list_index_names()` with "Serverless services cannot enumerate resources
without paging" — so this module never enumerates indexes. Existence checks
use `get_index(name)` guarded by try/except instead.
"""
from __future__ import annotations

import logging

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from rag.config import get_settings
from rag.models import Chunk, DocumentMetadata

_ALGORITHM = "hnsw-config"
_PROFILE = "hnsw-profile"

logger = logging.getLogger(__name__)


def build_index_definition() -> SearchIndex:
    settings = get_settings()
    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="department", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SearchableField(name="section_path", type=SearchFieldDataType.String),
        SimpleField(name="section_number", type=SearchFieldDataType.String,
                    filterable=True),
        SimpleField(name="version", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="effective_from", type=SearchFieldDataType.DateTimeOffset,
                    filterable=True, sortable=True),
        SimpleField(name="effective_to", type=SearchFieldDataType.DateTimeOffset,
                    filterable=True, sortable=True),
        SimpleField(name="is_current", type=SearchFieldDataType.Boolean,
                    filterable=True),
        SimpleField(name="superseded_by", type=SearchFieldDataType.String,
                    filterable=True),
        SimpleField(name="content_type", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="prev_chunk_id", type=SearchFieldDataType.String),
        SimpleField(name="next_chunk_id", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True, vector_search_dimensions=settings.embedding_dimensions,
            vector_search_profile_name=_PROFILE,
        ),
    ]
    return SearchIndex(
        name=settings.azure_search_index,
        fields=fields,
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name=_ALGORITHM)],
            profiles=[VectorSearchProfile(name=_PROFILE,
                                          algorithm_configuration_name=_ALGORITHM)],
        ),
        semantic_search=SemanticSearch(configurations=[
            SemanticConfiguration(
                name=settings.azure_search_semantic_config,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="section_path")],
                ),
            )
        ]),
    )


def _schema_matches(existing: SearchIndex, desired: SearchIndex) -> bool:
    """Compare field names+types only. A serverless index cannot be patched
    to an incompatible schema in place, so a mismatch here means the caller
    must delete and recreate rather than update."""
    existing_fields = {(f.name, str(f.type)) for f in existing.fields}
    desired_fields = {(f.name, str(f.type)) for f in desired.fields}
    return existing_fields == desired_fields


def ensure_index() -> None:
    """Create the index if absent, or recreate it if its schema doesn't match.

    Azure AI Search cannot apply incompatible field changes to a live index
    via `create_or_update_index()`. When the existing index's fields don't
    match the desired schema (e.g. the pre-existing single-field stub some
    services start with), this DELETES the index first — destroying its
    documents — then creates the correct schema. That deletion is logged
    loudly because it is destructive and deliberate, not a side effect.
    """
    settings = get_settings()
    desired = build_index_definition()
    client = SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(settings.azure_search_key),
    )
    with client:
        try:
            existing = client.get_index(desired.name)
        except ResourceNotFoundError:
            existing = None

        if existing is not None and not _schema_matches(existing, desired):
            logger.warning(
                "Deleting Azure AI Search index %r: existing schema (%d fields) "
                "does not match desired schema (%d fields). All documents in "
                "this index will be lost.",
                desired.name, len(existing.fields), len(desired.fields),
            )
            client.delete_index(desired.name)
            existing = None

        client.create_or_update_index(desired)


def _to_document(chunk: Chunk, vector: list[float], meta: DocumentMetadata) -> dict:
    return {
        "chunk_id": chunk.compute_id(),
        "doc_id": meta.doc_id,
        "title": meta.title,
        "department": meta.department,
        "section_path": " > ".join(chunk.section_path),
        "section_number": chunk.section_number,
        "version": meta.version,
        "effective_from": (meta.effective_from.isoformat() + "T00:00:00Z"
                           if meta.effective_from else None),
        "effective_to": (meta.effective_to.isoformat() + "T00:00:00Z"
                         if meta.effective_to else None),
        "is_current": meta.is_current,
        "superseded_by": meta.superseded_by,
        "content_type": chunk.content_type,
        "page": chunk.page,
        "prev_chunk_id": chunk.prev_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "content": chunk.display_text,
        "content_vector": vector,
    }


class AzureSearchSink:
    """Writes chunks + vectors to Azure AI Search. Does not parse, chunk, or embed.

    The underlying `SearchClient` wraps an aiohttp transport that cannot be
    reopened once closed (`azure.core.pipeline.transport._aiohttp` raises
    "HTTP transport has already been closed" if `open()` is called again
    after `close()`). A sink that lives across multiple calls — the normal
    case here — must therefore NOT wrap each method in `async with
    self._client:`; that pattern closes the transport after the first call
    and breaks every subsequent one. Instead the session is opened lazily on
    first use and left open for the sink's lifetime; call `aclose()` once
    when the owning pipeline/process shuts down.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index,
            credential=AzureKeyCredential(settings.azure_search_key),
        )

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
                     meta: DocumentMetadata) -> int:
        if not chunks:
            return 0
        docs = [_to_document(c, v, meta) for c, v in zip(chunks, vectors)]
        results = await self._client.merge_or_upload_documents(documents=docs)
        return sum(1 for r in results if r.succeeded)

    async def set_version_flags(
        self, doc_id: str, *, is_current: bool, superseded_by: str
    ) -> int:
        """Stamp a document's chunks with its resolved supersession status.

        A merge rather than a re-upload: only the two version fields are sent,
        so the content, the 1536-dimension vector and every other field are
        left untouched. Re-uploading whole documents to change a boolean would
        mean re-embedding or shipping the vectors back, and would rewrite the
        index on every reconciliation pass.

        Called by `rag.targets.version_sync`, which is the only thing that
        knows the answer -- supersession is a fact about a *pair* of documents
        and cannot be decided while processing either one alone.
        """
        found = await self._client.search(
            search_text="*", filter=f"doc_id eq '{doc_id}'",
            select=["chunk_id"], top=1000,
        )
        patch = [
            {
                "chunk_id": item["chunk_id"],
                "is_current": is_current,
                "superseded_by": superseded_by,
            }
            async for item in found
        ]
        if not patch:
            return 0
        results = await self._client.merge_documents(documents=patch)
        return sum(1 for r in results if r.succeeded)

    async def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of a document. Called when a source item disappears."""
        found = await self._client.search(
            search_text="*", filter=f"doc_id eq '{doc_id}'",
            select=["chunk_id"], top=1000,
        )
        keys = [{"chunk_id": item["chunk_id"]} async for item in found]
        if not keys:
            return 0
        results = await self._client.delete_documents(documents=keys)
        return sum(1 for r in results if r.succeeded)

    async def aclose(self) -> None:
        """Release the underlying HTTP session. Call once at process shutdown."""
        await self._client.close()

    async def __aenter__(self) -> "AzureSearchSink":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
