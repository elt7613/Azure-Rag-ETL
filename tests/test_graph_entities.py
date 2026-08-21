"""The semantic graph layer: entities, typed relations, and the provenance
rule that makes them auditable.

Three things are under test here, and only the first is about Cypher.

**Provenance is the contract, not a nicety.** The stated risk of an
LLM-extracted graph is an invented relationship. The system's answer is that
no edge is written unless it can name the chunk it came from, and that chunk
has to exist as a node. So the live tests do not merely count edges -- they
join every relation back to a real `:Chunk` and fail if a single one dangles.
A relation that cannot be traced is dropped and *counted*, because a silently
filtered edge and an extractor that never produced one look identical in the
graph.

**Idempotence is what makes an ETL an ETL.** Re-ingesting a document must
replace its contribution, never add a second copy of it. The live tests
therefore run the whole write twice and assert the counts did not move -- the
only way to catch a MERGE keyed on something that varies between runs.

**Deletes reach the new layer too.** A document leaving the source takes its
relations and mentions with it, and an entity left with no remaining
provenance is swept. An entity another document still cites survives. Both
directions are asserted, because a sweep that is too eager is as wrong as one
that never fires.

Test isolation: this database is shared with the other live suites, so every
synthetic node this file creates is namespaced with `_PROBE_NS` (pid-scoped)
and removed in a `finally`. The one test that uses the real corpus writes the
real doc_ids on purpose -- leaving the graph correctly ingested is the
intended end state, not a side effect.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from rag.models import Chunk, DocumentMetadata, Entity, Relation
from tests.conftest import azure_configured, neo4j_configured

live_graph = pytest.mark.skipif(not neo4j_configured(), reason="Neo4j not configured")
live_llm = pytest.mark.skipif(
    not azure_configured("azure_openai_endpoint", "azure_openai_key",
                         "azure_openai_chat_deployment"),
    reason="Azure OpenAI chat deployment not configured",
)

# Every synthetic doc_id, entity name and department this file writes carries
# this prefix. Two runs of the suite (another agent, CI, a laptop) share one
# Neo4j instance, and a probe that is not namespaced is a probe that deletes
# somebody else's data half way through their assertion.
_PROBE_NS = f"gent{os.getpid()}"
_PROBE_DEPT = f"{_PROBE_NS}-dept"


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _meta(doc_id: str) -> DocumentMetadata:
    return DocumentMetadata(doc_id=doc_id, title=f"Probe {doc_id}",
                            department=_PROBE_DEPT, version="1.0")


def _chunks(doc_id: str) -> list[Chunk]:
    return [
        Chunk(doc_id=doc_id, section_path=["1 Leave"],
              display_text="Employees are entitled to Paid Time Off after 90 days.",
              embed_text="x", content_type="prose", page=1, chunk_index=0,
              section_number="1"),
        Chunk(doc_id=doc_id, section_path=["1 Leave"],
              display_text="Manager approval is required for Paid Time Off.",
              embed_text="y", content_type="prose", page=1, chunk_index=1,
              section_number="1"),
    ]


def _structure(doc_id: str):
    from rag.targets.graph import build_graph_elements

    return build_graph_elements(_chunks(doc_id), _meta(doc_id))


def _entity(name: str, entity_type: str = "Benefit") -> Entity:
    return Entity(name=name, type=entity_type, department=_PROBE_DEPT)


def _relation(doc_id: str, chunk_id: str, *, subject="Paid Time Off",
              predicate="REQUIRES", obj="Manager approval",
              subject_type="Benefit", object_type="Obligation") -> Relation:
    return Relation(
        subject=subject, predicate=predicate, object=obj,
        subject_type=subject_type, object_type=object_type,
        doc_id=doc_id, source_chunk_id=chunk_id, section_path="1 Leave",
        page=1, department=_PROBE_DEPT, confidence=0.9,
        evidence_span="Manager approval is required for Paid Time Off.",
    )


# ==========================================================================
# 1. build_semantic_elements -- the provenance gate, with no network at all
# ==========================================================================


def test_entity_nodes_are_keyed_on_the_models_canonical_id():
    from rag.targets.graph import build_semantic_elements

    structure = _structure(f"{_PROBE_NS}/a.pdf")
    entity = _entity("Paid Time Off")
    elements = build_semantic_elements(structure, [entity], [], [])

    assert [e.entity_id for e in elements.entities] == [entity.compute_id()]
    assert elements.entities[0].department == _PROBE_DEPT


def test_a_relation_whose_source_chunk_is_not_in_the_document_is_dropped_and_counted():
    """The whole answer to "an LLM can invent a relationship" is that an edge
    has to name a chunk that really exists. An unresolvable citation is not a
    warning, it is a rejection -- and it is counted, so a corpus that starts
    producing them is visible rather than merely smaller."""
    from rag.targets.graph import build_semantic_elements

    doc_id = f"{_PROBE_NS}/a.pdf"
    structure = _structure(doc_id)
    good = _relation(doc_id, structure.chunks[1].chunk_id)
    invented = _relation(doc_id, "0000000000000000000000000000000000000000",
                         subject="Ghost Policy")

    elements = build_semantic_elements(structure, [], [good, invented], [])

    assert [r.subject for r in elements.relations] == ["Paid Time Off"]
    assert elements.dropped_no_provenance == 1


def test_a_relation_with_no_source_chunk_at_all_is_dropped():
    from rag.targets.graph import build_semantic_elements

    doc_id = f"{_PROBE_NS}/a.pdf"
    structure = _structure(doc_id)
    elements = build_semantic_elements(structure, [], [_relation(doc_id, "")], [])

    assert elements.relations == []
    assert elements.dropped_no_provenance == 1


def test_relation_endpoints_become_entities_even_when_never_declared():
    """The extractor emits `entities` and `relations` as two independent
    lists and nothing forces them to agree. An edge whose endpoint has no node
    is an edge nothing can traverse, so the endpoints are minted here rather
    than trusted to have been declared."""
    from rag.targets.graph import build_semantic_elements

    doc_id = f"{_PROBE_NS}/a.pdf"
    structure = _structure(doc_id)
    elements = build_semantic_elements(
        structure, [], [_relation(doc_id, structure.chunks[0].chunk_id)], []
    )

    names = {e.name for e in elements.entities}
    assert names == {"Paid Time Off", "Manager approval"}
    ids = {e.entity_id for e in elements.entities}
    assert {elements.relations[0].subject_id, elements.relations[0].object_id} == ids


def test_an_off_ontology_predicate_never_becomes_a_relationship_type():
    """The predicate is interpolated into Cypher as a real relationship type,
    so "is it in the closed vocabulary" is a safety check, not only a schema
    one."""
    from rag.targets.graph import build_semantic_elements

    doc_id = f"{_PROBE_NS}/a.pdf"
    structure = _structure(doc_id)
    bad = _relation(doc_id, structure.chunks[0].chunk_id, predicate="DELETE]->() //")

    elements = build_semantic_elements(structure, [], [bad], [])

    assert elements.relations == []
    assert elements.dropped_unknown_predicate == 1


def test_mentions_are_scoped_to_this_documents_chunks_and_deduplicated():
    from rag.targets.graph import build_semantic_elements

    doc_id = f"{_PROBE_NS}/a.pdf"
    structure = _structure(doc_id)
    pto = _entity("Paid Time Off")
    mentions = [
        (structure.chunks[0].chunk_id, pto),
        (structure.chunks[0].chunk_id, pto),          # duplicate
        ("not-a-chunk-of-this-document", pto),        # foreign chunk
    ]

    elements = build_semantic_elements(structure, [pto], [], mentions)

    assert [(m.chunk_id, m.entity_id) for m in elements.mentions] == [
        (structure.chunks[0].chunk_id, pto.compute_id())
    ]


def test_every_relation_also_gets_a_mention_from_its_own_source_chunk():
    """MENTIONS is what an entity-anchored retrieval walks and what the
    orphan sweep reads as "this entity still has evidence". An edge whose
    endpoints were never mentioned by any chunk would be swept away the next
    time anything was deleted, so the source chunk always mentions both ends
    of the edge it justifies."""
    from rag.targets.graph import build_semantic_elements

    doc_id = f"{_PROBE_NS}/a.pdf"
    structure = _structure(doc_id)
    chunk_id = structure.chunks[1].chunk_id
    elements = build_semantic_elements(structure, [], [_relation(doc_id, chunk_id)], [])

    mentioned = {m.entity_id for m in elements.mentions if m.chunk_id == chunk_id}
    assert mentioned == {e.entity_id for e in elements.entities}


# ==========================================================================
# 1b. Routing -- tables must reach the deterministic path, not the model
# ==========================================================================


class _RecordingExtractor:
    """Records what was handed to the model and returns nothing."""

    def __init__(self) -> None:
        self.seen: list = []

    async def extract(self, units):
        from rag.models import ExtractionResult

        self.seen.extend(units)
        return [ExtractionResult(unit_id=u.unit_id) for u in units]


async def test_every_parsed_table_is_paired_with_its_extraction_unit():
    """The pairing between a parsed TABLE block and the unit built from it is
    a text match, which makes it quietly fragile: anything that decorates a
    table chunk's text -- a section caption added for the reranker's benefit,
    say -- breaks equality and every table in the corpus falls through to the
    LLM path, where triage skips it as a table and it produces nothing at all.

    That failure has no symptom other than a graph with fewer relations than
    it should have, which is why it is asserted against the real corpus
    document with the most tables rather than a fixture.
    """
    from rag.etl.app import _table_rows_by_unit, parse_and_chunk
    from rag.extraction.units import build_units

    for doc_id in ("sales/Discounts.xlsx", "finance/ExpensePolicy.pdf"):
        parsed, chunks, meta = await parse_and_chunk(
            open(f"source_data/{doc_id}", "rb").read(), doc_id
        )
        units = build_units(chunks, meta)
        table_units = [u for u in units if u.content_type == "table"]
        assert table_units, f"{doc_id} has no table units to route"
        assert len(_table_rows_by_unit(parsed, units)) == len(table_units), \
            f"{doc_id}: a parsed table failed to pair with its extraction unit"


async def test_tables_never_reach_the_model_and_still_produce_relations():
    """The owner's direction, asserted rather than assumed: a rate table is
    already relational, so it is read deterministically and the model never
    sees it. Asserted by counting the calls that do not happen -- the only way
    "this costs nothing" is observable."""
    from rag.etl.app import extract_semantics, parse_and_chunk

    doc_id = "sales/Discounts.xlsx"
    parsed, chunks, meta = await parse_and_chunk(
        open(f"source_data/{doc_id}", "rb").read(), doc_id
    )
    extractor = _RecordingExtractor()
    entities, relations, mentions = await extract_semantics(
        parsed, chunks, meta, extractor
    )

    assert not [u for u in extractor.seen if u.content_type == "table"], \
        "a table was sent to the model"
    deterministic = [r for r in relations if r.deterministic]
    assert deterministic, "the deterministic tabular path produced no relations"
    assert len(deterministic) == len(relations)
    assert entities and mentions


# ==========================================================================
# 2. The output-truncation cliff
#
# These exercise `rag.extraction.llm`, not the graph writer. They live here
# because the truncation guard was written as part of this task: a pack of
# ~35 units that hits the completion ceiling used to be discarded whole --
# paid for, and silently absent from the graph. That is a defect of the
# semantic layer, however far upstream the fix lands.
# ==========================================================================


def _completion(content: str, finish_reason: str, completion_tokens: int = 4096):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                 finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=completion_tokens,
                              total_tokens=1200 + completion_tokens,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=1024)),
    )


class _TruncatingClient:
    """Truncates any request carrying more than `tolerable` units, the way a
    real completion ceiling does: deterministically, and by size."""

    def __init__(self, tolerable: int = 1) -> None:
        self._tolerable = tolerable
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        labels = _labels_in(kwargs)
        if len(labels) > self._tolerable:
            return _completion('{"units": [{"unit_id": "u1", "rela', "length")
        payload = {"units": [{"unit_id": label, "entities": [], "relations": []}
                             for label in labels]}
        return _completion(json.dumps(payload), "stop", completion_tokens=40)


def _labels_in(request: dict) -> list[str]:
    import re

    return re.findall(r"^\[(u\d+)\]", request["messages"][-1]["content"], re.MULTILINE)


def _extraction_units(count: int) -> list:
    from rag.models import ExtractionUnit

    body = (
        "Employees who have completed 90 days of continuous service are entitled "
        "to 20 days of paid time off per calendar year, and must obtain written "
        "approval from their manager at least 10 business days in advance. "
    )
    return [
        ExtractionUnit(
            unit_id=f"unit-{i}", doc_id=f"{_PROBE_NS}/a.pdf", department=_PROBE_DEPT,
            section_path=f"{i} Leave", text=f"Clause {i}. {body}", page=1,
            chunk_ids=[f"chunk-{i}"],
        )
        for i in range(count)
    ]


def test_the_request_carries_an_explicit_completion_budget(tmp_path):
    """Without one the deployment's own default decides where the cliff is,
    and it moves under us when the deployment changes."""
    from rag.extraction.llm import MAX_COMPLETION_TOKENS, RelationExtractor

    client = _TruncatingClient(tolerable=99)
    extractor = RelationExtractor(client=client,
                                  cache=_temp_cache(tmp_path / "budget.db"))
    _run(extractor.extract(_extraction_units(2)))

    assert client.requests
    assert client.requests[0]["max_completion_tokens"] == MAX_COMPLETION_TOKENS


def test_a_truncated_pack_is_split_and_retried_instead_of_being_thrown_away(tmp_path):
    """A 3000-token pack can hold ~35 units and produce ~10k completion
    tokens. Discarding the whole pack on truncation loses 35 sections and
    still pays for them; halving until it fits loses nothing."""
    from rag.extraction.llm import RelationExtractor

    client = _TruncatingClient(tolerable=1)
    extractor = RelationExtractor(client=client,
                                  cache=_temp_cache(tmp_path / "split.db"))
    units = _extraction_units(4)

    results = _run(extractor.extract(units))

    assert [r.unit_id for r in results] == [u.unit_id for u in units]
    assert all(not r.skipped_reason for r in results), \
        "a truncated pack must be recovered by splitting, not reported as skipped"
    assert extractor.stats.packs_truncated >= 1
    assert extractor.stats.packs_split >= 1
    assert extractor.stats.units_extracted == len(units)


def test_a_single_unit_that_still_truncates_is_reported_not_silently_lost(tmp_path):
    """Splitting bottoms out at one unit. That case cannot be recovered by
    halving, so it is recorded as a failed pack -- and deliberately not
    cached, so raising the budget retries it instead of serving the emptiness
    forever."""
    from rag.extraction.llm import RelationExtractor

    client = _TruncatingClient(tolerable=0)
    extractor = RelationExtractor(client=client,
                                  cache=_temp_cache(tmp_path / "single.db"))

    results = _run(extractor.extract(_extraction_units(1)))

    assert results[0].relations == []
    assert extractor.stats.packs_failed == 1
    assert "Truncated" in results[0].skipped_reason


def _temp_cache(path):
    from rag.extraction.cache import ExtractionCache

    return ExtractionCache(path)


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# ==========================================================================
# 3. Live Neo4j
# ==========================================================================


async def _scalar(writer, cypher: str, **params):
    async with writer._driver.session(database=writer._database) as session:
        result = await session.run(cypher, **params)
        record = await result.single()
        return record[0] if record else None


async def _semantic_counts(writer, doc_id: str) -> dict:
    """Entities reachable from this document, plus its relations and mentions."""
    async with writer._driver.session(database=writer._database) as session:
        result = await session.run(
            "OPTIONAL MATCH (:Document {doc_id:$doc_id})-[:CONTAINS*1..2]->(:Chunk)"
            "-[m:MENTIONS]->(e:Entity) "
            "WITH count(m) AS mentions, count(DISTINCT e) AS entities "
            "OPTIONAL MATCH (:Entity)-[r]->(:Entity) WHERE r.doc_id = $doc_id "
            "RETURN mentions, entities, count(r) AS relations",
            doc_id=doc_id,
        )
        return dict(await result.single())


async def _dangling_relations(writer, doc_id: str) -> int:
    """Relations of `doc_id` whose `source_chunk_id` names no real Chunk.

    This is the assertion the whole provenance rule exists for; it must be
    zero, always, for every document in the graph.
    """
    return await _scalar(
        writer,
        "MATCH (:Entity)-[r]->(:Entity) WHERE r.doc_id = $doc_id "
        "AND NOT EXISTS { MATCH (c:Chunk {chunk_id: r.source_chunk_id}) } "
        "RETURN count(r)",
        doc_id=doc_id,
    )


async def _purge_probe(writer) -> None:
    """Remove everything this file wrote, by namespace. Runs in a `finally`
    so a failing assertion never leaves debris in a shared database."""
    async with writer._driver.session(database=writer._database) as session:
        await session.run(
            "MATCH (d:Document) WHERE d.doc_id STARTS WITH $ns "
            "OPTIONAL MATCH (d)-[:CONTAINS*1..2]->(n) DETACH DELETE d, n",
            ns=_PROBE_NS,
        )
        await session.run(
            "MATCH (e:Entity {department:$dept}) DETACH DELETE e", dept=_PROBE_DEPT
        )
        await session.run(
            "MATCH (dept:Department {name:$dept}) DETACH DELETE dept", dept=_PROBE_DEPT
        )


@pytest.fixture
async def writer():
    from rag.targets.graph import Neo4jGraphWriter

    w = Neo4jGraphWriter()
    try:
        yield w
    finally:
        await _purge_probe(w)
        await w.close()


def _semantics(doc_id: str, structure):
    """Structure + a small, fully-provenanced semantic layer over it."""
    from rag.targets.graph import build_semantic_elements

    pto = _entity("Paid Time Off")
    relations = [
        _relation(doc_id, structure.chunks[1].chunk_id),
        _relation(doc_id, structure.chunks[0].chunk_id, predicate="ELIGIBLE_FOR",
                  obj="Employees", object_type="Role"),
    ]
    mentions = [(structure.chunks[0].chunk_id, pto)]
    return build_semantic_elements(structure, [pto], relations, mentions)


@live_graph
async def test_schema_declares_the_keys_every_merge_looks_up(writer):
    """Without these, every `MERGE (e:Entity {entity_id: ...})` is a label
    scan and ingest time grows with the square of the corpus. The constraint
    is also the thing that makes the entity key actually unique rather than
    merely intended to be."""
    await writer.ensure_schema()

    async with writer._driver.session(database=writer._database) as session:
        result = await session.run("SHOW CONSTRAINTS YIELD labelsOrTypes, properties")
        constrained = {(tuple(r["labelsOrTypes"] or ()), tuple(r["properties"] or ()))
                       async for r in result}
        result = await session.run("SHOW INDEXES YIELD labelsOrTypes, properties")
        indexed = {(tuple(r["labelsOrTypes"] or ()), tuple(r["properties"] or ()))
                   async for r in result}

    assert (("Entity",), ("entity_id",)) in constrained
    assert (("Chunk",), ("chunk_id",)) in constrained
    assert (("Document",), ("doc_id",)) in constrained
    assert (("Entity",), ("department", "type")) in indexed


@live_graph
async def test_write_semantics_creates_entities_relations_and_mentions(writer):
    doc_id = f"{_PROBE_NS}/write.pdf"
    structure = _structure(doc_id)
    await writer.ensure_schema()
    await writer.write(structure)
    await writer.write_semantics(_semantics(doc_id, structure))

    counts = await _semantic_counts(writer, doc_id)
    assert counts["relations"] == 2
    assert counts["entities"] >= 3       # PTO, Manager approval, Employees
    assert counts["mentions"] >= 4
    assert await _dangling_relations(writer, doc_id) == 0


@live_graph
async def test_every_relation_carries_its_full_provenance(writer):
    doc_id = f"{_PROBE_NS}/prov.pdf"
    structure = _structure(doc_id)
    await writer.ensure_schema()
    await writer.write(structure)
    await writer.write_semantics(_semantics(doc_id, structure))

    async with writer._driver.session(database=writer._database) as session:
        result = await session.run(
            "MATCH (a:Entity)-[r:REQUIRES]->(b:Entity) WHERE r.doc_id = $doc_id "
            "RETURN properties(r) AS props, a.name AS subject, b.name AS object",
            doc_id=doc_id,
        )
        record = await result.single()

    props = record["props"]
    assert record["subject"] == "Paid Time Off"
    assert record["object"] == "Manager approval"
    for field in ("doc_id", "source_chunk_id", "section_path", "page",
                  "department", "confidence", "evidence_span", "deterministic"):
        assert field in props, f"relation is missing provenance field {field!r}"
    assert props["source_chunk_id"] == structure.chunks[1].chunk_id
    assert props["department"] == _PROBE_DEPT
    assert props["deterministic"] is False


@live_graph
async def test_rewriting_the_same_document_does_not_duplicate_anything(writer):
    doc_id = f"{_PROBE_NS}/idem.pdf"
    await writer.ensure_schema()

    structure = _structure(doc_id)
    await writer.write(structure)
    await writer.write_semantics(_semantics(doc_id, structure))
    first = await _semantic_counts(writer, doc_id)

    structure = _structure(doc_id)
    await writer.write(structure)
    await writer.write_semantics(_semantics(doc_id, structure))
    second = await _semantic_counts(writer, doc_id)

    assert first == second
    assert first["relations"] == 2


@live_graph
async def test_a_relation_that_lost_its_evidence_is_replaced_not_accumulated(writer):
    """A re-ingest that reshapes sections gives every chunk a new id, so the
    previous run's relations cite chunks that no longer exist. They have to
    go with the run that produced them -- keeping them would leave the graph
    citing evidence the corpus no longer contains."""
    doc_id = f"{_PROBE_NS}/restate.pdf"
    await writer.ensure_schema()

    structure = _structure(doc_id)
    await writer.write(structure)
    await writer.write_semantics(_semantics(doc_id, structure))

    # Second pass with different text: new chunk ids, one relation instead of two.
    from rag.targets.graph import build_graph_elements, build_semantic_elements

    chunks = [Chunk(doc_id=doc_id, section_path=["1 Leave"],
                    display_text="Entirely different wording for the same section.",
                    embed_text="z", content_type="prose", page=1, chunk_index=0,
                    section_number="1")]
    restated = build_graph_elements(chunks, _meta(doc_id))
    await writer.write(restated)
    await writer.write_semantics(build_semantic_elements(
        restated, [], [_relation(doc_id, restated.chunks[0].chunk_id)], []
    ))

    counts = await _semantic_counts(writer, doc_id)
    assert counts["relations"] == 1
    assert await _dangling_relations(writer, doc_id) == 0


@live_graph
async def test_a_reingest_sweeps_the_entities_its_previous_pass_left_behind(writer):
    """Re-extraction is not deterministic across model versions or prompt
    changes, so pass two names entities pass one did not. Pass one's entity
    lost its mentions when `write()` replaced the chunks and its relations
    when `write_semantics` cleared them by doc_id -- which leaves it in the
    graph with no citable source at all, exactly the unfalsifiable claim the
    delete path refuses to keep. A re-ingest has to sweep for the same reason
    a delete does; only the trigger differs.

    An entity another document still cites must survive the sweep, which is
    what makes this narrower than "delete every entity this document wrote".
    """
    doc_id, other = f"{_PROBE_NS}/resweep.pdf", f"{_PROBE_NS}/other-doc.pdf"
    await writer.ensure_schema()
    from rag.targets.graph import build_semantic_elements

    shared = _entity("Paid Time Off")
    dropped = _entity("Sabbatical Programme", "Benefit")

    # Another document independently cites the shared entity.
    other_structure = _structure(other)
    await writer.write(other_structure)
    await writer.write_semantics(build_semantic_elements(
        other_structure, [shared], [], [(other_structure.chunks[0].chunk_id, shared)]
    ))

    structure = _structure(doc_id)
    candidates = await writer.write(structure)
    await writer.write_semantics(build_semantic_elements(
        structure, [shared, dropped], [],
        [(structure.chunks[0].chunk_id, shared),
         (structure.chunks[0].chunk_id, dropped)],
    ), orphan_candidates=candidates)

    # Second pass names only the shared entity.
    structure = _structure(doc_id)
    candidates = await writer.write(structure)
    await writer.write_semantics(build_semantic_elements(
        structure, [shared], [], [(structure.chunks[0].chunk_id, shared)]
    ), orphan_candidates=candidates)

    names = await _scalar(
        writer,
        "MATCH (e:Entity {department:$dept}) RETURN collect(e.name) AS names",
        dept=_PROBE_DEPT,
    )
    assert dropped.name not in names, \
        "an entity nothing cites any more survived a re-ingest"
    assert shared.name in names, "an entity another document still cites was swept"


@live_graph
async def test_deleting_a_document_takes_its_relations_mentions_and_orphans(writer):
    """Two documents share one entity. Deleting the first must remove its
    edges and the entity only it cited, and must leave the shared one alone.
    A sweep that fires on the shared entity is a data-loss bug; one that never
    fires leaves unfalsifiable claims in the graph forever."""
    doomed, keeper = f"{_PROBE_NS}/doomed.pdf", f"{_PROBE_NS}/keeper.pdf"
    await writer.ensure_schema()

    shared = _entity("Paid Time Off")
    lonely = _entity("Manager approval", "Obligation")

    for doc_id in (doomed, keeper):
        structure = _structure(doc_id)
        await writer.write(structure)
        relations = [_relation(doc_id, structure.chunks[1].chunk_id)] if doc_id == doomed else []
        from rag.targets.graph import build_semantic_elements

        await writer.write_semantics(build_semantic_elements(
            structure, [shared], relations, [(structure.chunks[0].chunk_id, shared)]
        ))

    before = await _semantic_counts(writer, doomed)
    assert before["relations"] == 1

    await writer.delete_document(doomed)

    after = await _semantic_counts(writer, doomed)
    assert after == {"mentions": 0, "entities": 0, "relations": 0}

    surviving = await _scalar(
        writer,
        "MATCH (e:Entity {department:$dept}) RETURN collect(e.name) AS names",
        dept=_PROBE_DEPT,
    )
    assert shared.name in surviving, "an entity another document still cites was swept"
    assert lonely.name not in surviving, "an entity with no remaining provenance survived"


@live_graph
async def test_an_entity_graph_supersedes_edge_survives_document_relinking(writer):
    """`SUPERSEDES` is both a Document->Document structural edge and a member
    of the extraction ontology. `_relink_supersedes` rebuilds the structural
    ones on every write; if it deletes them untyped it silently destroys the
    extracted layer's SUPERSEDES relations too."""
    doc_id = f"{_PROBE_NS}/supersedes.pdf"
    await writer.ensure_schema()

    structure = _structure(doc_id)
    await writer.write(structure)
    from rag.targets.graph import build_semantic_elements

    await writer.write_semantics(build_semantic_elements(
        structure, [],
        [_relation(doc_id, structure.chunks[0].chunk_id, predicate="SUPERSEDES",
                   subject="2026 Rate Card", obj="2025 Rate Card",
                   subject_type="Document", object_type="Document")],
        [],
    ))

    # Any other document's write triggers a full SUPERSEDES relink.
    other = _structure(f"{_PROBE_NS}/other.pdf")
    await writer.write(other)

    survived = await _scalar(
        writer,
        "MATCH (:Entity)-[r:SUPERSEDES]->(:Entity) WHERE r.doc_id = $doc_id "
        "RETURN count(r)",
        doc_id=doc_id,
    )
    assert survived == 1


# ==========================================================================
# 4. The real corpus, end to end
# ==========================================================================


@live_graph
@live_llm
async def test_the_real_corpus_gets_a_semantic_layer_and_a_second_pass_changes_nothing():
    """The acceptance test for this task, run against the real documents and
    the real stores.

    It drives `process_document` itself -- decorator, error handling and all
    -- with only the vector-store collaborators substituted, so what is proven
    is the ETL wiring rather than a re-implementation of it. The second pass
    is served entirely from the extraction cache and must leave every count
    exactly where it was.
    """
    from importlib import import_module
    from pathlib import Path

    app = import_module("rag.etl.app")
    from rag.extraction.llm import RelationExtractor
    from rag.targets.graph import Neo4jGraphWriter

    class _NullSink:
        async def delete_document(self, doc_id: str) -> int:
            return 0

        async def upsert(self, chunks, vectors, meta) -> int:
            return len(chunks)

    class _NullEmbedder:
        async def embed(self, texts):
            return [[0.0] for _ in texts]

    graph_writer = Neo4jGraphWriter()
    extractor = RelationExtractor()
    original_collaborators = app._collaborators
    original_extractor = app._extractor
    app._collaborators = lambda: (_NullSink(), _NullEmbedder(), graph_writer)
    app._extractor = lambda: extractor

    class _File:
        def __init__(self, path: Path) -> None:
            self.file_path = SimpleNamespace(
                path=Path(path.relative_to("source_data").as_posix())
            )
            self._path = path

        async def read(self) -> bytes:
            return self._path.read_bytes()

    files = sorted(p for p in Path("source_data").rglob("*") if p.is_file())
    assert files, "no corpus to ingest"

    try:
        await graph_writer.ensure_schema()
        await graph_writer.ensure_departments()

        app.INGEST_STATS.reset()
        for path in files:
            assert await app.process_document(_File(path)) >= 0
        assert app.INGEST_STATS.documents_failed == 0, app.INGEST_STATS.errors

        totals = await _corpus_totals(graph_writer)
        assert totals["entities"] > 0, "no entities were extracted from the real corpus"
        assert totals["relations"] > 0, "no relations were extracted from the real corpus"
        assert totals["mentions"] > 0
        assert totals["dangling"] == 0, \
            "a relation cites a chunk that does not exist -- provenance is broken"

        # Second pass: same bytes, same sections, everything served from the
        # content-hash cache. Nothing may move.
        first_pass_docs = app.INGEST_STATS.documents_succeeded
        cache_hits_before = extractor.stats.cache_hits
        for path in files:
            await app.process_document(_File(path))

        assert app.INGEST_STATS.documents_succeeded > first_pass_docs, \
            "the second pass was short-circuited -- it proves nothing about idempotence"
        assert extractor.stats.cache_hits > cache_hits_before, \
            "a re-ingest of unchanged text must be served by the extraction cache"
        assert await _corpus_totals(graph_writer) == totals

        print("\ncorpus semantic layer:", totals)
        print("extraction:", extractor.stats.summary())
        print("cost:", extractor.cost.summary())
    finally:
        app._collaborators = original_collaborators
        app._extractor = original_extractor
        await graph_writer.close()


async def _corpus_totals(writer) -> dict:
    async with writer._driver.session(database=writer._database) as session:
        result = await session.run(
            "MATCH (e:Entity) WITH count(e) AS entities "
            "MATCH ()-[m:MENTIONS]->() WITH entities, count(m) AS mentions "
            "OPTIONAL MATCH (:Entity)-[r]->(:Entity) "
            "RETURN entities, mentions, count(r) AS relations, "
            "count(CASE WHEN NOT EXISTS { MATCH (c:Chunk {chunk_id: r.source_chunk_id}) } "
            "THEN 1 END) AS dangling"
        )
        return dict(await result.single())
