import asyncio

import pytest

from tests.conftest import azure_configured


def test_index_definition_matches_configured_dimensions():
    from rag.config import get_settings
    from rag.targets.azure_search import build_index_definition

    index = build_index_definition()
    vector_field = next(f for f in index.fields if f.name == "content_vector")
    assert vector_field.vector_search_dimensions == get_settings().embedding_dimensions
    assert index.semantic_search is not None
    names = {f.name for f in index.fields}
    assert {"chunk_id", "doc_id", "department", "is_current",
            "effective_from", "section_path", "content"} <= names


@pytest.mark.skipif(
    not azure_configured("azure_search_endpoint", "azure_search_key", "azure_search_index"),
    reason="Azure AI Search not configured",
)
async def test_upsert_then_delete_roundtrip():
    from rag.config import get_settings
    from rag.models import Chunk, DocumentMetadata
    from rag.targets.azure_search import AzureSearchSink, ensure_index

    ensure_index()
    meta = DocumentMetadata(doc_id="test/doc.pdf", title="T", department="HR")
    chunk = Chunk(doc_id=meta.doc_id, section_path=["1 X"], display_text="hello",
                  embed_text="hello", content_type="prose", page=1, chunk_index=0)
    sink = AzureSearchSink()
    vector = [0.0] * get_settings().embedding_dimensions
    try:
        assert await sink.upsert([chunk], [vector], meta) == 1

        # Azure AI Search indexing is asynchronous: a document just written
        # via merge_or_upload is not guaranteed to be immediately queryable,
        # and delete_document locates documents via search. Poll with a
        # small, bounded retry rather than assuming instant visibility or
        # waiting unboundedly.
        deleted = 0
        for _ in range(10):
            deleted = await sink.delete_document(meta.doc_id)
            if deleted >= 1:
                break
            await asyncio.sleep(1)
        assert deleted >= 1
    finally:
        await sink.aclose()
