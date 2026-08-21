"""Live Neo4j tests for Neo4jGraphWriter. Gated on real config being present
(never os.getenv -- see tests/conftest.py's `azure_configured` docstring for
why: .env is read by pydantic-settings without exporting to os.environ)."""
import os
import uuid

import pytest

from tests.conftest import azure_configured

# These tests write to the SHARED live Neo4j database, so a fixed probe doc_id
# means two concurrent pytest processes delete each other's nodes mid-assertion
# -- which showed up as intermittent count mismatches and EntityNotFound errors
# while several agents ran the suite at once. Scoping the probe to this process
# makes the tests independent of who else is running.
PROBE_DOC_ID = f"test/LiveWriterProbe-{os.getpid()}-{uuid.uuid4().hex[:8]}.pdf"


def _neo4j_configured() -> bool:
    return azure_configured("neo4j_uri", "neo4j_user", "neo4j_password", "neo4j_database")


pytestmark = pytest.mark.skipif(not _neo4j_configured(), reason="Neo4j not configured")


@pytest.fixture
async def writer():
    from rag.targets.graph import Neo4jGraphWriter

    w = Neo4jGraphWriter()
    yield w
    # Clean up any nodes this test wrote, then close the driver.
    async with w._driver.session(database=w._database) as session:
        await session.run(
            "MATCH (d:Document {doc_id:$doc_id}) "
            "OPTIONAL MATCH (d)-[:CONTAINS*1..2]->(n) "
            "DETACH DELETE d, n",
            doc_id=PROBE_DOC_ID,
        )
    await w.close()


def _elements():
    from rag.models import Chunk, DocumentMetadata
    from rag.targets.graph import build_graph_elements

    meta = DocumentMetadata(doc_id=PROBE_DOC_ID, title="LiveWriterProbe",
                            department="test", version="1.0")
    chunks = [
        Chunk(doc_id=meta.doc_id, section_path=["1 Intro"], display_text="a",
              embed_text="a", content_type="prose", page=1, chunk_index=0,
              section_number="1"),
        Chunk(doc_id=meta.doc_id, section_path=["1 Intro"], display_text="b",
              embed_text="b", content_type="prose", page=1, chunk_index=1,
              section_number="1"),
        Chunk(doc_id=meta.doc_id, section_path=["2 Details"], display_text="c",
              embed_text="c", content_type="prose", page=2, chunk_index=2,
              section_number="2"),
    ]
    return build_graph_elements(chunks, meta)


async def _counts(writer, doc_id: str) -> dict:
    async with writer._driver.session(database=writer._database) as session:
        result = await session.run(
            "MATCH (d:Document {doc_id:$doc_id}) "
            "OPTIONAL MATCH (d)-[:CONTAINS]->(s:Section) "
            "OPTIONAL MATCH (s)-[:CONTAINS]->(c:Chunk) "
            "OPTIONAL MATCH (c)-[n:NEXT]->() "
            "RETURN count(DISTINCT s) AS sections, count(DISTINCT c) AS chunks, "
            "count(DISTINCT n) AS next_edges",
            doc_id=doc_id,
        )
        record = await result.single()
        return dict(record)


async def test_write_creates_expected_node_and_edge_counts(writer):
    await writer.write(_elements())
    counts = await _counts(writer, PROBE_DOC_ID)
    assert counts == {"sections": 2, "chunks": 3, "next_edges": 2}


async def test_write_is_idempotent_no_duplication(writer):
    await writer.write(_elements())
    first = await _counts(writer, PROBE_DOC_ID)
    await writer.write(_elements())
    second = await _counts(writer, PROBE_DOC_ID)
    assert first == second == {"sections": 2, "chunks": 3, "next_edges": 2}
