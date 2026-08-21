"""Neo4j property-graph shapes: the structural layer and the semantic one.

Two layers over the same documents, written by the same driver, kept
deliberately distinct because they have opposite failure modes.

**Layer A -- structure (deterministic, free).**

    Nodes:  Department -> Document -> Section -> Chunk
    Edges:  HAS_DOCUMENT (Department->Document)
            CONTAINS (Document->Section, Section->Chunk)
            NEXT     (Chunk->Chunk, reading order)
            SUPERSEDES (Document->Document, parsed from the header)

Every node and edge here is derived deterministically from parsing, so it
needs no LLM extraction and cannot hallucinate a relationship. It is always
present, even for a document nothing could be extracted from.

**Layer B -- semantics (LLM-extracted or read off a table, triaged).**

    Nodes:  Entity (keyed on department + type + normalized name)
    Edges:  MENTIONS (Chunk->Entity)
            the closed relation vocabulary (Entity->Entity), each edge
            carrying the provenance that justifies it

Layer B *can* hallucinate, and the design's answer is not to hope it does
not: it is that **no edge is written without a resolvable source chunk**.
`build_semantic_elements` refuses any relation whose `source_chunk_id` is not
one of the chunks this very document just wrote, and counts the refusals --
because a silently filtered edge and an extractor that never produced one
leave an identical graph behind. What survives can be joined back to a
`:Chunk`, a page, and a quoted span, which is what makes a graph-derived
answer as auditable as a vector-derived one.

The two layers also delete differently. Structure hangs off the Document, so
`DETACH DELETE` on the document's subtree takes it all. A relation edge hangs
between two `Entity` nodes that outlive any single document, so it has to be
found by its `doc_id` property and removed explicitly, and the entities it
left behind have to be swept if nothing else cites them. Both halves are in
`delete_document`; missing either one is an invisible leak.

`Department` is the one node type that comes from configuration rather than
from a document. Its whole purpose is that the graph's top level mirrors the
`DEPARTMENTS` env list exactly -- see `rag.departments` -- so adding or
removing a department reshapes the graph without a code change, and a
department with no documents yet is still visibly part of the corpus rather
than absent.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from rag.departments import get_registry
from rag.extraction.ontology import RELATION_TYPES
from rag.models import Chunk, DocumentMetadata, Entity, Relation

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_WORD_RE = re.compile(r"[A-Za-z]+")

# Minimum token length for title-token overlap. Filters out noise like "v",
# "a" while still catching real stems ("Pricing", "Rate").
_MIN_TOKEN_LEN = 3

# The predicate becomes a real Neo4j relationship type, which Cypher will not
# accept as a parameter -- it has to be interpolated into the statement text.
# That makes "is this predicate in the closed vocabulary?" a safety check and
# not merely a schema one, so the membership test is done against this frozen
# set immediately before any interpolation happens, never against whatever
# string an extractor happened to return.
_RELATION_TYPE_SET = frozenset(RELATION_TYPES)


def _section_id(doc_id: str, path: list[str]) -> str:
    return hashlib.sha1(f"{doc_id}\x1f{'/'.join(path)}".encode()).hexdigest()


def _title_tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) >= _MIN_TOKEN_LEN}


@dataclass
class DocumentNode:
    doc_id: str
    title: str
    department: str
    version: str
    is_current: bool
    supersedes_hint: str = ""


@dataclass
class SectionNode:
    section_id: str
    doc_id: str
    section_path: str
    section_number: str


@dataclass
class ChunkNode:
    chunk_id: str
    doc_id: str
    content_type: str
    page: int
    text: str


@dataclass
class ContainsRel:
    from_id: str
    to_id: str


@dataclass
class NextRel:
    from_id: str
    to_id: str


@dataclass
class SupersedesRel:
    from_id: str
    to_hint: str


@dataclass
class GraphElements:
    document: DocumentNode
    sections: list[SectionNode] = field(default_factory=list)
    chunks: list[ChunkNode] = field(default_factory=list)
    contains: list[ContainsRel] = field(default_factory=list)
    next_edges: list[NextRel] = field(default_factory=list)
    supersedes: list[SupersedesRel] = field(default_factory=list)


def build_graph_elements(chunks: list[Chunk], meta: DocumentMetadata) -> GraphElements:
    graph = GraphElements(document=DocumentNode(
        doc_id=meta.doc_id, title=meta.title, department=meta.department,
        version=meta.version, is_current=meta.is_current,
        supersedes_hint=meta.supersedes,
    ))

    seen_sections: dict[str, SectionNode] = {}
    for chunk in chunks:
        sid = _section_id(meta.doc_id, chunk.section_path)
        if sid not in seen_sections:
            node = SectionNode(section_id=sid, doc_id=meta.doc_id,
                               section_path=" > ".join(chunk.section_path),
                               section_number=chunk.section_number)
            seen_sections[sid] = node
            graph.sections.append(node)
            graph.contains.append(ContainsRel(from_id=meta.doc_id, to_id=sid))

        cid = chunk.compute_id()
        graph.chunks.append(ChunkNode(chunk_id=cid, doc_id=meta.doc_id,
                                      content_type=chunk.content_type,
                                      page=chunk.page, text=chunk.display_text))
        graph.contains.append(ContainsRel(from_id=sid, to_id=cid))

    ids = [c.chunk_id for c in graph.chunks]
    graph.next_edges = [NextRel(from_id=a, to_id=b) for a, b in zip(ids, ids[1:])]

    if meta.supersedes:
        graph.supersedes.append(
            SupersedesRel(from_id=meta.doc_id, to_hint=meta.supersedes)
        )
    return graph


# --------------------------------------------------------------------------
# Layer B -- the semantic elements
# --------------------------------------------------------------------------


@dataclass
class EntityNode:
    """A canonical thing the corpus talks about, as the graph stores it.

    `entity_id` is `Entity.compute_id()` and nothing else: department + type +
    normalized name. That is what makes the same benefit named in four
    documents of one department converge on one node without any
    cross-document coordination at write time -- each document computes the
    same key independently and MERGEs onto it.
    """
    entity_id: str
    name: str
    type: str
    department: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RelationEdge:
    """One typed edge between two entities, with every field an auditor needs.

    The provenance is not metadata *about* the edge, it is the edge's licence
    to exist. `source_chunk_id` names a `:Chunk` node that is written in the
    same transaction as the structure; `evidence_span` is the substring of
    that chunk the extractor quoted; `page` and `section_path` are where a
    human looks. `deterministic` separates "a model read this" from "this was
    read off a spreadsheet", which retrieval weighs differently and a reader
    deserves to be told.
    """
    relation_id: str
    predicate: str
    subject_id: str
    object_id: str
    subject: str
    object: str
    doc_id: str
    source_chunk_id: str
    section_path: str
    page: int
    department: str
    confidence: float
    evidence_span: str
    deterministic: bool


@dataclass
class MentionEdge:
    """(:Chunk)-[:MENTIONS]->(:Entity).

    Two jobs. It is the anchor an entity-anchored retrieval walks backwards
    to reach text, and it is the record the orphan sweep reads as "this
    entity still has evidence somewhere". An entity with no mention and no
    relation is an unfalsifiable claim, and this system does not keep those.
    """
    chunk_id: str
    entity_id: str


@dataclass
class SemanticElements:
    """One document's contribution to Layer B, already filtered.

    The two `dropped_*` counters are part of the output, not a log line. A
    corpus whose `dropped_no_provenance` starts climbing has an extractor
    that changed underneath it, and a run that reports zero relations is
    indistinguishable from a run that silently rejected all of them unless
    the rejections are carried out alongside the acceptances.
    """
    doc_id: str
    entities: list[EntityNode] = field(default_factory=list)
    relations: list[RelationEdge] = field(default_factory=list)
    mentions: list[MentionEdge] = field(default_factory=list)
    dropped_no_provenance: int = 0
    dropped_unknown_predicate: int = 0


def _entity_node(entity: Entity) -> EntityNode:
    return EntityNode(
        entity_id=entity.compute_id(), name=entity.name, type=entity.type,
        department=entity.department, aliases=list(entity.aliases),
        description=entity.description,
    )


def _endpoint(name: str, entity_type: str, department: str) -> Entity:
    """The `Entity` an edge's endpoint refers to, minted from the edge itself."""
    return Entity(name=name, type=entity_type, department=department)


def build_semantic_elements(
    structure: GraphElements,
    entities: Iterable[Entity],
    relations: Iterable[Relation],
    mentions: Iterable[tuple[str, Entity]],
) -> SemanticElements:
    """Turn extraction output into exactly what may be written, and count the rest.

    `structure` is the document's own just-built Layer A, and it is the
    authority on which chunk ids exist. Taking it as the argument rather than
    a bare doc_id is deliberate: the set a relation's citation is checked
    against is then *the same object* that will create those `:Chunk` nodes,
    so there is no window in which the check passes against a stale chunk set.

    Three rules, in order:

    1. **No edge without a resolvable source chunk.** A relation citing a
       chunk this document did not produce -- a hallucinated id, a stale id
       from a previous chunking, an empty string -- is dropped and counted.
       This is the design's whole answer to "an LLM can invent a
       relationship": the invention is not filtered by plausibility, it is
       filtered by whether a reader could go and check.
    2. **No edge outside the closed vocabulary.** The predicate becomes a
       Neo4j relationship type, so an unrecognised one is both an ontology
       violation and an injection vector.
    3. **Every endpoint gets a node.** The extractor emits entities and
       relations as two independent lists and nothing makes them agree; an
       edge whose endpoint has no node is an edge nothing can traverse. The
       endpoints are therefore minted from the edge, and the source chunk is
       recorded as mentioning both of them -- otherwise the first delete
       anywhere in the corpus would sweep away entities that are still cited
       by a live edge.
    """
    valid_chunks = {c.chunk_id for c in structure.chunks}
    doc_id = structure.document.doc_id

    nodes: dict[str, EntityNode] = {}

    def _register(entity: Entity) -> str:
        node = _entity_node(entity)
        existing = nodes.get(node.entity_id)
        if existing is None:
            nodes[node.entity_id] = node
        elif not existing.description and node.description:
            existing.description = node.description
        return node.entity_id

    for entity in entities:
        _register(entity)

    edges: list[RelationEdge] = []
    mention_pairs: dict[tuple[str, str], MentionEdge] = {}
    dropped_provenance = 0
    dropped_predicate = 0

    for relation in relations:
        if relation.source_chunk_id not in valid_chunks:
            dropped_provenance += 1
            logger.debug(
                "dropped relation %r-[%s]->%r in %s: source chunk %r resolves to nothing",
                relation.subject, relation.predicate, relation.object,
                doc_id, relation.source_chunk_id,
            )
            continue
        if relation.predicate not in _RELATION_TYPE_SET:
            dropped_predicate += 1
            logger.debug("dropped relation with off-ontology predicate %r in %s",
                         relation.predicate, doc_id)
            continue

        department = relation.department or structure.document.department
        subject_id = _register(_endpoint(relation.subject, relation.subject_type, department))
        object_id = _register(_endpoint(relation.object, relation.object_type, department))
        edges.append(RelationEdge(
            relation_id=relation.compute_id(),
            predicate=relation.predicate,
            subject_id=subject_id, object_id=object_id,
            subject=relation.subject, object=relation.object,
            doc_id=doc_id, source_chunk_id=relation.source_chunk_id,
            section_path=relation.section_path, page=relation.page,
            department=department, confidence=relation.confidence,
            evidence_span=relation.evidence_span,
            deterministic=relation.deterministic,
        ))
        for entity_id in (subject_id, object_id):
            key = (relation.source_chunk_id, entity_id)
            mention_pairs.setdefault(key, MentionEdge(*key))

    for chunk_id, entity in mentions:
        if chunk_id not in valid_chunks:
            continue
        key = (chunk_id, _register(entity))
        mention_pairs.setdefault(key, MentionEdge(*key))

    return SemanticElements(
        doc_id=doc_id,
        entities=list(nodes.values()),
        relations=edges,
        mentions=list(mention_pairs.values()),
        dropped_no_provenance=dropped_provenance,
        dropped_unknown_predicate=dropped_predicate,
    )


def resolve_supersedes(
    source: DocumentNode, hint: str, candidates: list[DocumentNode]
) -> DocumentNode | None:
    """Match a free-text `Supersedes:` hint (e.g. "2025 Rate Card (v1.4)") to
    a document node already known to the graph.

    The naive approach -- `toLower(hint) CONTAINS toLower(candidate.version)`
    -- is fragile in two ways: a version string like "1.4" can appear as a
    substring inside an unrelated department's document, and a version like
    "1.0" can coincidentally appear as a substring of descriptive hint text
    that has nothing to do with that document. Both are real risks once the
    corpus has more than a couple of documents sharing small version numbers.

    This requires ALL of the following, so no single coincidental substring
    match is ever sufficient on its own:
      - same `department` as `source` (a document can only supersede a
        document from its own department's series);
      - a literal 4-digit year parsed from the hint text appears in the
        candidate's doc_id, title, or version;
      - at least one letter-token (len >= 3) is shared between `source`'s
        title and the candidate's title -- e.g. "Pricing2026" and
        "Pricing2025" both tokenize to include "pricing", but a document
        titled "Roadmap2025" would not overlap even though its year matches.

    Returns the first candidate satisfying all three, or None if the hint
    carries no year or no candidate qualifies.
    """
    year_match = _YEAR_RE.search(hint)
    if not year_match:
        return None
    year = year_match.group(1)

    source_tokens = _title_tokens(source.title)
    if not source_tokens:
        return None

    for candidate in candidates:
        if candidate.doc_id == source.doc_id:
            continue
        if candidate.department != source.department:
            continue
        haystack = f"{candidate.doc_id} {candidate.title} {candidate.version}"
        if year not in haystack:
            continue
        if source_tokens & _title_tokens(candidate.title):
            return candidate
    return None


def resolve_all_supersedes(nodes: list[DocumentNode]) -> list[tuple[str, str]]:
    """Recompute every SUPERSEDES pair across the whole graph's known documents.

    Cross-document resolution needs both sides of a pair to exist as nodes,
    which a single per-document write cannot guarantee (processing order is
    not fixed). Recomputing the full set from every Document node's stored
    `supersedes_hint` on each write is cheap at this corpus's scale and
    means the edge appears as soon as both documents have been ingested at
    least once, regardless of which was processed first.
    """
    pairs: list[tuple[str, str]] = []
    for node in nodes:
        if not node.supersedes_hint:
            continue
        target = resolve_supersedes(node, node.supersedes_hint, nodes)
        if target is not None:
            pairs.append((node.doc_id, target.doc_id))
    return pairs


class Neo4jGraphWriter:
    """Idempotent MERGE-based writer. Re-running a document replaces its
    subgraph rather than duplicating it.

    Follows the same lifecycle pattern as `AzureSearchSink`: the driver is
    opened lazily once and reused for every `write()` call, then closed once
    at shutdown via `close()` -- never opened/closed per call.
    """

    def __init__(self) -> None:
        from neo4j import AsyncGraphDatabase

        from rag.config import get_settings

        settings = get_settings()
        self._database = settings.neo4j_database
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def write(self, elements: GraphElements) -> list[str]:
        """Write the document's structural layer, replacing the previous one.

        Returns the element ids of the `Entity` nodes that were mentioned by
        the chunks this call is about to destroy -- the *only* moment those
        entities can still be identified. Pass them to `write_semantics` as
        `orphan_candidates` so the ones the new pass no longer cites are
        swept; see that method for why a re-ingest needs a sweep at all.
        """
        async with self._driver.session(database=self._database) as session:
            # Deleting this document's chunks takes their MENTIONS with them,
            # so any entity whose only evidence was this document becomes
            # unprovenanced at this instant and unfindable a moment later.
            # Noted by internal element id rather than by property, for the
            # same reason `delete_document` does it that way.
            result = await session.run(
                "MATCH (:Document {doc_id:$doc_id})-[:CONTAINS*1..2]->()"
                "-[:MENTIONS]->(e:Entity) "
                "RETURN collect(DISTINCT elementId(e)) AS entity_ids",
                doc_id=elements.document.doc_id,
            )
            record = await result.single()
            orphan_candidates = list(record["entity_ids"]) if record else []

            # Drop this document's existing sections/chunks so a re-ingest
            # (or a re-chunk that changes chunk_id/section_id) does not leave
            # orphans behind. The Document node itself is preserved so its
            # incoming SUPERSEDES edges (from a doc it supersedes) survive.
            await session.run(
                "MATCH (d:Document {doc_id:$doc_id})-[:CONTAINS*1..2]->(n) "
                "DETACH DELETE n",
                doc_id=elements.document.doc_id,
            )
            await session.run(
                "MERGE (d:Document {doc_id:$doc_id}) "
                "SET d.title=$title, d.department=$department, "
                "    d.version=$version, d.is_current=$is_current, "
                "    d.supersedes_hint=$supersedes_hint",
                doc_id=elements.document.doc_id,
                title=elements.document.title,
                department=elements.document.department,
                version=elements.document.version,
                is_current=elements.document.is_current,
                supersedes_hint=elements.document.supersedes_hint,
            )
            if elements.sections:
                await session.run(
                    "UNWIND $rows AS r MERGE (s:Section {section_id:r.section_id}) "
                    "SET s.doc_id=r.doc_id, s.section_path=r.section_path, "
                    "    s.section_number=r.section_number",
                    rows=[vars(s) for s in elements.sections],
                )
            if elements.chunks:
                await session.run(
                    "UNWIND $rows AS r MERGE (c:Chunk {chunk_id:r.chunk_id}) "
                    "SET c.doc_id=r.doc_id, c.content_type=r.content_type, "
                    "    c.page=r.page, c.text=r.text",
                    rows=[vars(c) for c in elements.chunks],
                )
            if elements.contains:
                # Split by relationship shape and match each side by its own
                # label instead of an unlabeled `MATCH (a) WHERE a.doc_id=...
                # OR a.section_id=...`. Section and Chunk nodes ALSO carry a
                # `doc_id` property (set just above), so for a Document->
                # Section row that unlabeled OR would match the Document node
                # PLUS every Section/Chunk node sharing that doc_id --
                # producing a combinatorial blowup of CONTAINS edges instead
                # of one per row. Labeled matches make each row bind exactly
                # the two nodes it names.
                doc_id = elements.document.doc_id
                doc_section_rows = [vars(e) for e in elements.contains if e.from_id == doc_id]
                section_chunk_rows = [vars(e) for e in elements.contains if e.from_id != doc_id]
                if doc_section_rows:
                    await session.run(
                        "UNWIND $rows AS r "
                        "MATCH (a:Document {doc_id:r.from_id}) "
                        "MATCH (b:Section {section_id:r.to_id}) "
                        "MERGE (a)-[:CONTAINS]->(b)",
                        rows=doc_section_rows,
                    )
                if section_chunk_rows:
                    await session.run(
                        "UNWIND $rows AS r "
                        "MATCH (a:Section {section_id:r.from_id}) "
                        "MATCH (b:Chunk {chunk_id:r.to_id}) "
                        "MERGE (a)-[:CONTAINS]->(b)",
                        rows=section_chunk_rows,
                    )
            if elements.next_edges:
                await session.run(
                    "UNWIND $rows AS r "
                    "MATCH (a:Chunk {chunk_id:r.from_id}), (b:Chunk {chunk_id:r.to_id}) "
                    "MERGE (a)-[:NEXT]->(b)",
                    rows=[vars(e) for e in elements.next_edges],
                )

            # The department label already went through the registry in
            # `extract_metadata`, so it is the configured canonical name, not
            # a raw folder segment. The existing edge is dropped first: a
            # document that moves between departments must not end up owned
            # by both.
            await session.run(
                "MATCH (:Department)-[r:HAS_DOCUMENT]->(:Document {doc_id:$doc_id}) "
                "DELETE r",
                doc_id=elements.document.doc_id,
            )
            await session.run(
                "MERGE (dept:Department {name:$name}) "
                "WITH dept MATCH (d:Document {doc_id:$doc_id}) "
                "MERGE (dept)-[:HAS_DOCUMENT]->(d)",
                name=elements.document.department,
                doc_id=elements.document.doc_id,
            )

            await self._relink_supersedes(session)

        return orphan_candidates

    async def write_semantics(
        self, elements: SemanticElements, orphan_candidates: Iterable[str] = ()
    ) -> None:
        """Write one document's `Entity` nodes, relation edges and mentions.

        Call after `write()`, never before: the relation MERGE matches the
        `:Chunk` its provenance names, so the structure has to exist for the
        edge to be allowed to.

        **How re-ingest is made idempotent without losing shared entities.**
        `write()` has already DETACH DELETEd this document's chunks, which
        took its `MENTIONS` with them -- so those need no separate cleanup.
        Relation edges are the opposite case: they hang between two `Entity`
        nodes that outlive any single document, so nothing about deleting the
        document's chunks touches them, and a re-ingest that reworded a
        section would otherwise leave the previous run's edges pointing at
        chunk ids that no longer exist. They are therefore deleted by
        `doc_id` first and rewritten from scratch. What is deliberately NOT
        deleted is the `Entity` nodes themselves: another department's
        document may be the only remaining thing citing one, and dropping it
        because *this* document stopped mentioning it would silently break
        that document's edges. Entities are removed only by the orphan sweep,
        which asks whether anything at all still cites them.

        The two writes are then MERGEs on keys that are pure functions of
        content (`entity_id`, `relation_id`), so running this twice over
        identical input converges rather than accumulating.

        **Why a re-ingest still needs an orphan sweep.** Extraction is not
        stable across model versions, prompt changes, or a cache that was
        cleared: pass two names entities pass one did not. Such an entity has
        just lost its mentions (with the chunks) and its relations (cleared
        above), which leaves it in the graph citing nothing -- the same
        unfalsifiable claim `delete_document` refuses to keep, arriving by a
        different route. `orphan_candidates` is what `write()` returned, and
        the sweep is deliberately restricted to it plus the endpoints of the
        relations just cleared: those are the only entities this document
        could possibly have orphaned, so the cost is bounded by the document
        rather than by the corpus. Entities another document still cites are
        untouched, which is why this is not simply "delete what this document
        wrote last time".
        """
        entity_rows = [vars(e) for e in elements.entities]
        by_predicate: dict[str, list[dict]] = {}
        for edge in elements.relations:
            # Belt and braces around the interpolation below: `build_semantic_
            # elements` already rejected anything off-vocabulary, and this
            # refuses to be the place where that stops being true.
            if edge.predicate not in _RELATION_TYPE_SET:
                raise ValueError(f"refusing to write off-ontology predicate {edge.predicate!r}")
            by_predicate.setdefault(edge.predicate, []).append({
                "relation_id": edge.relation_id,
                "subject_id": edge.subject_id,
                "object_id": edge.object_id,
                "props": {
                    "subject": edge.subject, "object": edge.object,
                    "doc_id": edge.doc_id, "source_chunk_id": edge.source_chunk_id,
                    "section_path": edge.section_path, "page": edge.page,
                    "department": edge.department, "confidence": edge.confidence,
                    "evidence_span": edge.evidence_span,
                    "deterministic": edge.deterministic,
                },
            })

        async with self._driver.session(database=self._database) as session:
            at_risk = set(orphan_candidates) | await self._relation_endpoints(
                session, elements.doc_id
            )
            await self._clear_relations(session, elements.doc_id)

            if entity_rows:
                await session.run(
                    "UNWIND $rows AS r MERGE (e:Entity {entity_id:r.entity_id}) "
                    "SET e.name=r.name, e.type=r.type, e.department=r.department, "
                    "    e.aliases=r.aliases, e.description=r.description",
                    rows=entity_rows,
                )

            written = 0
            for predicate, rows in by_predicate.items():
                # One statement per predicate rather than one dynamic-type
                # statement for all of them. The vocabulary is closed and
                # small (14 types), so this is at most 14 round trips per
                # document, and it keeps the relationship type a literal that
                # was checked against the frozen set two lines above rather
                # than a value flowing in from an extractor.
                result = await session.run(
                    "UNWIND $rows AS r "
                    "MATCH (a:Entity {entity_id:r.subject_id}) "
                    "MATCH (b:Entity {entity_id:r.object_id}) "
                    "MATCH (c:Chunk {chunk_id:r.props.source_chunk_id}) "
                    f"MERGE (a)-[e:{predicate} {{relation_id:r.relation_id}}]->(b) "
                    "SET e += r.props "
                    "RETURN count(e) AS written",
                    rows=rows,
                )
                record = await result.single()
                written += record["written"] if record else 0

            if written != len(elements.relations):
                # The `MATCH (c:Chunk ...)` above silently drops any row whose
                # citation does not resolve. Silently is exactly what this
                # system must not do about provenance, so the shortfall is
                # reported rather than left to be inferred from a count.
                logger.warning(
                    "%s: %d of %d relations were not written -- provenance did not resolve",
                    elements.doc_id, len(elements.relations) - written,
                    len(elements.relations),
                )

            if elements.mentions:
                await session.run(
                    "UNWIND $rows AS r "
                    "MATCH (c:Chunk {chunk_id:r.chunk_id}) "
                    "MATCH (e:Entity {entity_id:r.entity_id}) "
                    "MERGE (c)-[:MENTIONS]->(e)",
                    rows=[vars(m) for m in elements.mentions],
                )

            # After the writes, never before: an entity this pass re-cites has
            # its new mentions by now and must not be swept on the strength of
            # having lost its old ones.
            swept = await self._sweep_orphans(session, at_risk)
            if swept:
                logger.info("%s: swept %d entity node(s) left with no provenance",
                            elements.doc_id, swept)

        if elements.dropped_no_provenance or elements.dropped_unknown_predicate:
            logger.info(
                "%s: dropped %d relation(s) with unresolvable provenance and "
                "%d with an off-ontology predicate",
                elements.doc_id, elements.dropped_no_provenance,
                elements.dropped_unknown_predicate,
            )

    async def ensure_schema(self) -> list[str]:
        """Create the constraints and indexes every lookup in this module needs.

        Not an optimisation. `MERGE (e:Entity {entity_id: ...})` without an
        index on `entity_id` is a full label scan, and a document's write does
        one per entity -- so ingest time grows with (documents x entities),
        which is quadratic in the corpus. On eleven documents that is
        invisible; at five million it is the difference between a run that
        finishes and one that does not. The uniqueness constraints do double
        duty: they provide the index *and* make the content-derived keys
        actually unique rather than merely intended to be.

        The relationship-property indexes on `doc_id` exist for one query:
        deleting or replacing a document's relation edges. Those edges hang
        between `Entity` nodes rather than off the Document, so they can only
        be found by their `doc_id` property -- and without an index that is a
        scan of every relation in the corpus, on every re-ingest of every
        document. Same quadratic, different edge.

        Idempotent (`IF NOT EXISTS`), so it is safe to call on every startup,
        which is the only way a schema change ever actually reaches a
        long-lived database.
        """
        statements = [
            "CREATE CONSTRAINT document_doc_id IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT section_section_id IF NOT EXISTS "
            "FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_entity_id IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT department_name IF NOT EXISTS "
            "FOR (d:Department) REQUIRE d.name IS UNIQUE",
            # Entity-anchored retrieval is always department-scoped (security
            # is deny-by-default and the scope is applied in the Cypher, not
            # after it), so the composite is the shape that is actually
            # queried.
            "CREATE INDEX entity_department_type IF NOT EXISTS "
            "FOR (e:Entity) ON (e.department, e.type)",
            "CREATE INDEX chunk_doc_id IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
        ]
        statements += [
            f"CREATE INDEX relation_doc_id_{predicate.lower()} IF NOT EXISTS "
            f"FOR ()-[r:{predicate}]-() ON (r.doc_id)"
            for predicate in RELATION_TYPES
        ]
        async with self._driver.session(database=self._database) as session:
            for statement in statements:
                await session.run(statement)
        return statements

    async def ensure_departments(self) -> list[str]:
        """Make the graph's Department nodes match the configured list.

        Called once at ETL startup rather than being left to `write()`, so a
        newly configured department is present in the graph from the moment
        it is configured -- before it has any documents, and even if it never
        gets any. That is the visible proof that departments are configuration
        rather than a side effect of whatever happened to be ingested.

        The reverse direction is pruned too, but only for departments that
        hold no documents. A department dropped from `DEPARTMENTS` whose
        documents are still in the graph keeps its node until delete
        reconciliation has swept those documents out; the next run then
        removes the now-empty node. Deleting the node first would orphan
        documents that are still there, which is worse than converging one
        run later.
        """
        names = get_registry().names()
        async with self._driver.session(database=self._database) as session:
            await session.run(
                "UNWIND $names AS name MERGE (:Department {name:name})", names=names
            )
            await session.run(
                "MATCH (d:Department) WHERE NOT d.name IN $names "
                "AND NOT (d)-[:HAS_DOCUMENT]->() DELETE d",
                names=names,
            )
        return names

    async def delete_document(self, doc_id: str) -> int:
        """Remove a document's entire subgraph. Called when the source item
        disappears, alongside `AzureSearchSink.delete_document`.

        ETL means deletes propagate: a document removed from the source has to
        leave the graph as well as the vector store, or every graph-anchored
        answer keeps citing a document that no longer exists. Deleting only
        from the vector store -- which is what happened before this method
        existed -- leaves the subgraph orphaned permanently, because nothing
        else ever visits that doc_id again.

        Four steps, in this order for a reason:

        1. Note which `Entity` nodes this document was the evidence for --
           both the ones its chunks MENTION and the ones its relation edges
           connect. This has to happen *before* the deletion, since the very
           edges that identify them are about to be destroyed. Entities are
           captured by internal element id rather than by a property, so the
           note survives whatever the entity key happens to be. `write()`
           takes the same note for the same reason; see `write_semantics`.
        2. Delete this document's relation edges. They are the half that
           `DETACH DELETE` cannot reach: a relation hangs between two
           `Entity` nodes, not off the Document, so removing the document's
           subtree leaves every one of its extracted edges behind, still
           citing a `source_chunk_id` that no longer resolves. They are found
           by their `doc_id` property, which is what the relationship-property
           indexes in `ensure_schema` exist for.
        3. Delete the Document, its Sections and its Chunks. `DETACH DELETE`
           takes their edges with them: CONTAINS, NEXT, MENTIONS, the
           incoming HAS_DOCUMENT, and any SUPERSEDES touching this document.
        4. Sweep any noted entity that has just lost its last piece of
           provenance -- no remaining mention, no remaining relation. An
           entity with no citable source is exactly what this system refuses
           to keep, an unfalsifiable claim, so it goes with its evidence.
           Entities another document still cites survive untouched: sweeping
           one of those would silently break that document's edges, which is
           a worse failure than keeping a node one run too long.

        SUPERSEDES is then recomputed for the same reason `write()` does it:
        removing a document changes which candidates the remaining documents'
        `Supersedes:` hints can resolve against, and one consistent rule for
        that edge is cheaper to reason about than two.

        Returns the number of nodes removed.
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (:Document {doc_id:$doc_id})-[:CONTAINS*1..2]->()"
                "-[:MENTIONS]->(e:Entity) "
                "RETURN collect(DISTINCT elementId(e)) AS entity_ids",
                doc_id=doc_id,
            )
            record = await result.single()
            entity_ids = set(record["entity_ids"]) if record else set()
            entity_ids |= await self._relation_endpoints(session, doc_id)

            await self._clear_relations(session, doc_id)

            result = await session.run(
                "MATCH (d:Document {doc_id:$doc_id}) "
                "OPTIONAL MATCH (d)-[:CONTAINS*1..2]->(n) "
                "WITH collect(DISTINCT d) + collect(DISTINCT n) AS doomed "
                "WITH doomed, size(doomed) AS removed "
                "FOREACH (node IN doomed | DETACH DELETE node) "
                "RETURN removed",
                doc_id=doc_id,
            )
            record = await result.single()
            removed = record["removed"] if record else 0

            swept = await self._sweep_orphans(session, entity_ids)

            await self._relink_supersedes(session)

        logger.info(
            "Deleted %s from the graph: %d structure nodes, %d orphaned entities",
            doc_id, removed, swept,
        )
        return removed + swept

    @staticmethod
    async def _relation_endpoints(session: Any, doc_id: str) -> set[str]:
        """Element ids of the entities this document's relations connect.

        The other half of the orphan-candidate set: `write()` can only see the
        entities its chunks MENTION, and an entity can be an edge endpoint
        without ever having been mentioned by a chunk that survived.
        """
        result = await session.run(
            "MATCH (a:Entity)-[r]->(b:Entity) WHERE r.doc_id = $doc_id "
            "RETURN collect(DISTINCT elementId(a)) + collect(DISTINCT elementId(b)) "
            "AS entity_ids",
            doc_id=doc_id,
        )
        record = await result.single()
        return set(record["entity_ids"]) if record else set()

    @staticmethod
    async def _sweep_orphans(session: Any, entity_ids: Iterable[str]) -> int:
        """Delete the named entities that no longer have any provenance.

        "No provenance" is no incoming `MENTIONS` and no incident relation --
        both, not either. An entity reachable only through an edge some other
        document wrote is still citable through that edge's `source_chunk_id`,
        and an entity mentioned by a chunk is citable even with no edges at
        all. Requiring both to be absent is what keeps the sweep from taking
        nodes that are still someone's evidence.
        """
        ids = list(entity_ids)
        if not ids:
            return 0
        result = await session.run(
            "MATCH (e:Entity) WHERE elementId(e) IN $entity_ids "
            "AND NOT (e)<-[:MENTIONS]-() AND NOT (e)-[]-(:Entity) "
            "DETACH DELETE e RETURN count(*) AS swept",
            entity_ids=ids,
        )
        record = await result.single()
        return record["swept"] if record else 0

    @staticmethod
    async def _clear_relations(session: Any, doc_id: str) -> int:
        """Delete every extracted relation this document is the evidence for.

        Matched as `(:Entity)-[r]->(:Entity)` with `r.doc_id` rather than by
        listing the fourteen relationship types: the vocabulary can grow, and
        an edge left behind because its type was added after this line was
        written is exactly the kind of leak nothing else in the system would
        ever notice. `MENTIONS` runs Chunk->Entity and `SUPERSEDES` between
        Documents, so neither is reachable by this pattern.
        """
        result = await session.run(
            "MATCH (:Entity)-[r]->(:Entity) WHERE r.doc_id = $doc_id "
            "DELETE r RETURN count(*) AS removed",
            doc_id=doc_id,
        )
        record = await result.single()
        return record["removed"] if record else 0

    async def _relink_supersedes(self, session: Any) -> None:
        """Recompute SUPERSEDES edges across every Document node currently
        in the graph. See `resolve_all_supersedes` for why this is a
        full-graph pass rather than a single-document one."""
        result = await session.run(
            "MATCH (d:Document) RETURN d.doc_id AS doc_id, d.title AS title, "
            "d.department AS department, d.version AS version, "
            "coalesce(d.is_current, true) AS is_current, "
            "coalesce(d.supersedes_hint, '') AS supersedes_hint"
        )
        nodes = [
            DocumentNode(
                doc_id=r["doc_id"], title=r["title"] or "",
                department=r["department"] or "", version=r["version"] or "",
                is_current=r["is_current"], supersedes_hint=r["supersedes_hint"],
            )
            async for r in result
        ]
        pairs = resolve_all_supersedes(nodes)

        # Labelled on both ends on purpose. `SUPERSEDES` is *also* a member
        # of the extraction ontology, so an unlabelled delete here would wipe
        # every extracted Entity->Entity SUPERSEDES edge in the graph every
        # time any document was written -- a data-loss bug with no symptom
        # except a semantic layer that keeps coming up short.
        await session.run("MATCH (:Document)-[r:SUPERSEDES]->(:Document) DELETE r")
        if pairs:
            await session.run(
                "UNWIND $rows AS r "
                "MATCH (a:Document {doc_id:r.from_id}), (b:Document {doc_id:r.to_id}) "
                "MERGE (a)-[:SUPERSEDES]->(b)",
                rows=[{"from_id": a, "to_id": b} for a, b in pairs],
            )

    async def close(self) -> None:
        await self._driver.close()
