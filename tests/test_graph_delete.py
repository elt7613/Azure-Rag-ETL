"""Delete propagation: a document removed from the source must disappear from
BOTH stores.

The headline test here is deliberately end-to-end and live. A unit test with a
fake driver would have happily passed against the pre-existing code, which
deleted from Azure AI Search and silently left the whole Neo4j subgraph
behind -- the bug was in the wiring, not in any single function, so the proof
has to run the real ingest against the real stores and then look for the
document in both of them.

`source_data/` is treated as read-only: the document that gets deleted is a
throwaway copy of a real corpus document, and `_source_data_unchanged`
asserts nothing was left removed even if a test fails midway.
"""
import asyncio
import os
import shutil
from pathlib import Path

import pytest

from tests.conftest import azure_configured, neo4j_configured

_SEARCH_FIELDS = ("azure_search_endpoint", "azure_search_key", "azure_search_index")
_EMBED_FIELDS = ("azure_openai_endpoint", "azure_openai_key",
                 "azure_openai_embedding_deployment")

live_graph = pytest.mark.skipif(not neo4j_configured(), reason="Neo4j not configured")
live_search = pytest.mark.skipif(
    not azure_configured(*_SEARCH_FIELDS, *_EMBED_FIELDS),
    reason="Azure AI Search / Azure OpenAI not configured",
)

_PROBE_SOURCE = Path("source_data/HR/LeavePolicy.pdf")
# The pid keeps two simultaneous runs -- another engineer, another agent, CI
# and a laptop -- from deleting each other's probe out from under them. The
# probe file and its Neo4j/Azure Search identity are the same string, so the
# isolation covers all three stores at once.
_PROBE_DOC_ID = f"HR/DeleteProbePolicy-{os.getpid()}.pdf"
_PROBE_PATH = Path("source_data") / _PROBE_DOC_ID

# Azure AI Search indexes asynchronously: a document is accepted before it is
# queryable, and a delete is likewise not immediately reflected. Polling is
# not flake-tolerance here, it is the service's actual contract.
_SEARCH_SETTLE_TIMEOUT = 90.0
_SEARCH_POLL_INTERVAL = 2.0


def _source_tree() -> set[str]:
    return {p.as_posix() for p in Path("source_data").rglob("*") if p.is_file()}


@pytest.fixture(autouse=True)
def _source_data_unchanged():
    """`source_data/` is the user's corpus, not test scratch space.

    Checked as "nothing was removed, and my probe is gone" rather than exact
    set equality: a test that deletes a source document has to prove it put
    every real document back, but it has no business failing because a
    concurrent process happened to be holding a probe of its own.
    """
    before = _source_tree()
    yield
    _PROBE_PATH.unlink(missing_ok=True)
    missing = before - _source_tree()
    assert not missing, f"tests removed files from source_data/: {sorted(missing)}"
    assert not _PROBE_PATH.exists()


@pytest.fixture
def probe_document():
    """A byte-for-byte copy of a real corpus document under a throwaway name,
    so the delete path is exercised on genuinely representative content
    without ever removing something the corpus needs."""
    shutil.copyfile(_PROBE_SOURCE, _PROBE_PATH)
    yield _PROBE_DOC_ID
    _PROBE_PATH.unlink(missing_ok=True)


async def _graph_counts(writer, doc_id: str) -> dict:
    async with writer._driver.session(database=writer._database) as session:
        result = await session.run(
            "OPTIONAL MATCH (d:Document {doc_id:$doc_id}) "
            "OPTIONAL MATCH (d)-[:CONTAINS]->(s:Section) "
            "OPTIONAL MATCH (s)-[:CONTAINS]->(c:Chunk) "
            "RETURN count(DISTINCT d) AS documents, count(DISTINCT s) AS sections, "
            "count(DISTINCT c) AS chunks",
            doc_id=doc_id,
        )
        return dict(await result.single())


async def _department_of(writer, doc_id: str) -> list[str]:
    async with writer._driver.session(database=writer._database) as session:
        result = await session.run(
            "MATCH (dept:Department)-[:HAS_DOCUMENT]->(:Document {doc_id:$doc_id}) "
            "RETURN dept.name AS name ORDER BY name",
            doc_id=doc_id,
        )
        return [r["name"] async for r in result]


async def _department_names(writer) -> set[str]:
    async with writer._driver.session(database=writer._database) as session:
        result = await session.run("MATCH (d:Department) RETURN d.name AS name")
        return {r["name"] async for r in result}


async def _search_chunk_count(sink, doc_id: str) -> int:
    found = await sink._client.search(
        search_text="*", filter=f"doc_id eq '{doc_id}'", select=["chunk_id"], top=1000
    )
    return len([item async for item in found])


async def _await_search_count(sink, doc_id: str, expected: int) -> int:
    deadline = asyncio.get_running_loop().time() + _SEARCH_SETTLE_TIMEOUT
    count = await _search_chunk_count(sink, doc_id)
    while count != expected and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(_SEARCH_POLL_INTERVAL)
        count = await _search_chunk_count(sink, doc_id)
    return count


# ---------------------------------------------------------------------------
# Wiring, without the network. Proves reconciliation reaches both targets --
# the specific thing that was missing -- on every run, live or not.
# ---------------------------------------------------------------------------


class _RecordingTarget:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_document(self, doc_id: str) -> int:
        self.deleted.append(doc_id)
        return 1


async def test_reconciliation_deletes_from_both_targets():
    from rag.etl.app import apply_deletions

    sink, graph = _RecordingTarget(), _RecordingTarget()
    await apply_deletions(["a/one.pdf", "b/two.pdf"], sink=sink, graph_writer=graph)

    assert sink.deleted == ["a/one.pdf", "b/two.pdf"]
    assert graph.deleted == ["a/one.pdf", "b/two.pdf"]


async def test_reconciliation_tolerates_a_disabled_graph():
    from rag.etl.app import apply_deletions

    sink = _RecordingTarget()
    await apply_deletions(["a/one.pdf"], sink=sink, graph_writer=None)

    assert sink.deleted == ["a/one.pdf"]


# ---------------------------------------------------------------------------
# Live end-to-end
# ---------------------------------------------------------------------------


@live_graph
@live_search
async def test_source_deletion_removes_document_from_both_stores(probe_document):
    from rag.embedding.azure_openai import AzureOpenAIEmbedder
    from rag.etl.app import apply_deletions, ingest_one
    from rag.targets.azure_search import AzureSearchSink
    from rag.targets.graph import Neo4jGraphWriter, build_graph_elements

    doc_id = probe_document
    sink, writer = AzureSearchSink(), Neo4jGraphWriter()
    embedder = AzureOpenAIEmbedder()
    try:
        await writer.ensure_departments()

        chunks, meta = await ingest_one(_PROBE_PATH.read_bytes(), doc_id)
        assert chunks, "probe document produced no chunks -- nothing to prove"
        vectors = await embedder.embed([c.embed_text for c in chunks])
        assert await sink.upsert(chunks, vectors, meta) == len(chunks)
        await writer.write(build_graph_elements(chunks, meta))

        assert await _await_search_count(sink, doc_id, len(chunks)) == len(chunks)
        before = await _graph_counts(writer, doc_id)
        assert before["documents"] == 1
        assert before["chunks"] == len(chunks)
        assert before["sections"] > 0
        # The department the document was filed under is config-driven, and
        # the graph has to say so.
        assert await _department_of(writer, doc_id) == ["HR"]

        # The source document goes away; a fresh scan of the source no longer
        # lists it. This is exactly what `_reconcile_deletions` diffs against.
        _PROBE_PATH.unlink()
        assert doc_id not in {
            p.relative_to("source_data").as_posix()
            for p in Path("source_data").rglob("*") if p.is_file()
        }

        await apply_deletions([doc_id], sink=sink, graph_writer=writer)

        assert await _await_search_count(sink, doc_id, 0) == 0
        assert await _graph_counts(writer, doc_id) == {
            "documents": 0, "sections": 0, "chunks": 0,
        }
    finally:
        await sink.aclose()
        await writer.close()


@live_graph
async def test_delete_sweeps_entities_that_lose_their_last_provenance():
    """The extracted-entity layer does not exist yet, so this builds the shape
    Task 8 will write -- (:Chunk)-[:MENTIONS]->(:Entity) -- and proves the
    sweep already handles it: an entity whose only evidence was the deleted
    document goes, one still cited elsewhere stays."""
    from rag.models import Chunk, DocumentMetadata
    from rag.targets.graph import Neo4jGraphWriter, build_graph_elements

    doc_id = f"test/EntitySweepProbe-{os.getpid()}.pdf"
    lonely, shared = f"sweep-lonely-{os.getpid()}", f"sweep-shared-{os.getpid()}"
    keeper = f"sweep-keeper-chunk-{os.getpid()}"
    meta = DocumentMetadata(doc_id=doc_id, title="EntitySweepProbe",
                            department="test", version="1.0")
    chunks = [Chunk(doc_id=doc_id, section_path=["1 Intro"], display_text="a",
                    embed_text="a", content_type="prose", page=1, chunk_index=0,
                    section_number="1")]

    writer = Neo4jGraphWriter()
    try:
        await writer.write(build_graph_elements(chunks, meta))
        async with writer._driver.session(database=writer._database) as session:
            await session.run(
                "MATCH (c:Chunk {chunk_id:$chunk_id}) "
                "MERGE (lonely:Entity {entity_id:$lonely}) "
                "MERGE (shared:Entity {entity_id:$shared}) "
                "MERGE (keeper:Chunk {chunk_id:$keeper}) "
                "MERGE (c)-[:MENTIONS]->(lonely) "
                "MERGE (c)-[:MENTIONS]->(shared) "
                "MERGE (keeper)-[:MENTIONS]->(shared)",
                chunk_id=chunks[0].compute_id(),
                lonely=lonely, shared=shared, keeper=keeper,
            )

        await writer.delete_document(doc_id)

        async with writer._driver.session(database=writer._database) as session:
            result = await session.run(
                "MATCH (e:Entity) WHERE e.entity_id IN $ids "
                "RETURN e.entity_id AS id",
                ids=[lonely, shared],
            )
            surviving = {r["id"] async for r in result}
        assert surviving == {shared}
    finally:
        async with writer._driver.session(database=writer._database) as session:
            await session.run(
                "MATCH (n) WHERE n.entity_id IN $ids OR n.chunk_id = $keeper "
                "DETACH DELETE n",
                ids=[lonely, shared], keeper=keeper,
            )
        await writer.delete_document(doc_id)
        await writer.close()


@live_graph
async def test_department_nodes_mirror_the_configured_registry():
    from rag.departments import get_registry
    from rag.targets.graph import Neo4jGraphWriter

    writer = Neo4jGraphWriter()
    try:
        await writer.ensure_departments()
        assert set(get_registry().names()) <= await _department_names(writer)
    finally:
        await writer.close()
