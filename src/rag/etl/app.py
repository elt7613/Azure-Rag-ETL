"""CocoIndex ETL application.

Wires parsing, metadata extraction, chunking, embedding, Azure AI Search
indexing, relationship extraction, and the Neo4j graph into one incremental
dataflow over the department source folders. A document therefore leaves this
module in three shapes at once: chunks with vectors in the search index, a
Document/Section/Chunk subgraph, and -- when `graph_extraction_enabled` -- a
layer of entities and typed, provenance-carrying relations over that subgraph.
Reprocessing is content-fingerprinted by CocoIndex's `@coco.fn(memo=True)`
machinery, not mtime-based: touching a file without changing its bytes is a
no-op, while a genuine content change, a new file, or a deletion triggers
exactly the affected work.

Live monitoring: `localfs.walk_dir(live=True)` returns a LiveMapView backed by
watchdog (inotify on Linux), so `cocoindex --app-dir . update rag.etl.app --live`
reacts to file creates/modifies/deletes across all five department folders in
real time, with `rescan_interval` as a safety net for events the OS watcher drops.

CocoIndex's own incremental state (what was processed, at what fingerprint) is
kept in an embedded LMDB store at the filesystem path named by the `COCOINDEX_DB`
environment variable -- read directly by the cocoindex library itself, not by
this module. `rag.config.Settings.cocoindex_db` documents the same path for the
rest of the app; nothing here needs to touch it. CocoIndex v0's
`COCOINDEX_DATABASE_URL` Postgres setting is a deprecated leftover with no
effect in v1 and is not part of this configuration.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cocoindex as coco
from cocoindex.connectors import localfs
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

from rag.config import get_settings
from rag.embedding.azure_openai import AzureOpenAIEmbedder
from rag.enrich.chunker import chunk_document
from rag.enrich.metadata import extract_metadata
from rag.extraction.llm import RelationExtractor
from rag.extraction.resolve import remap_relations, resolve_entities
from rag.extraction.tabular import extract_from_table
from rag.extraction.units import build_units
from rag.models import (
    BlockType,
    Chunk,
    DocumentMetadata,
    Entity,
    ExtractionUnit,
    IngestError,
    ParsedDocument,
    Relation,
)
from rag.parsing.base import select_parser
from rag.targets.azure_search import AzureSearchSink, ensure_index
from rag.targets.graph import Neo4jGraphWriter, SemanticElements
from rag.targets.version_sync import DocumentRegistry, reconcile_versions

logger = logging.getLogger(__name__)

EMBEDDER = coco.ContextKey[AzureOpenAIEmbedder]("embedder")
SEARCH_SINK = coco.ContextKey[AzureSearchSink]("search_sink")

# Only provided (non-None) when `settings.graph_enabled` is true. This is a
# document-structure graph mirror of the same chunks already written to
# Azure AI Search -- see `rag.targets.graph`'s module docstring -- kept
# entirely additive to the existing vector-store pipeline below.
GRAPH_WRITER = coco.ContextKey[Neo4jGraphWriter | None]("graph_writer")

# Passed to `localfs.walk_dir()` as its `path` argument (a ContextKey, not a
# plain `Path`) so every yielded `FilePath` is tracked relative to THIS base
# directory. Passing `settings.local_source_dir` (a bare `Path`) directly
# would leave `file.file_path.path` relative to the process's cwd instead --
# e.g. "source_data/HR/Benefits.pdf" rather than "HR/Benefits.pdf" -- which
# breaks `extract_metadata`'s department-from-folder inference downstream
# (it reads the first path segment). See `FilePath`'s own docstring example
# in `cocoindex.resources.file` for this exact base_dir pattern.
SOURCE_DIR = coco.ContextKey[Path]("source_dir")

# The relationship extractor, or None when `graph_extraction_enabled` is off
# (or the graph itself is). Provided once for the whole run rather than built
# per document on purpose: it owns the SQLite content-hash cache, the HTTP
# client whose connections are worth reusing, and the `CostTracker` whose
# whole value is that it totals a *run* rather than a document.
EXTRACTOR = coco.ContextKey[RelationExtractor | None]("extractor")

# Holds the (opened) Azure Blob container client for the lifetime of the app
# when DOC_SOURCE=blob, or None for DOC_SOURCE=local. Blob keys never carry
# the container name, so `AzureBlobFile.file_path.path` -- and thus doc_id --
# is already correctly relative with no base-dir juggling needed (unlike
# localfs, which requires SOURCE_DIR above).
CONTAINER = coco.ContextKey[Any]("blob_container")

# One row per ingested document, holding the metadata cross-document
# supersession resolution needs. Written per document while that metadata is
# already in hand; read in full by the reconciliation pass at the end of the
# run, which is the only point at which both sides of a supersession are
# visible at once. See `rag.targets.version_sync` for why this is a sidecar
# rather than a field on the search index or a read back from the graph.
REGISTRY = coco.ContextKey[DocumentRegistry]("document_registry")

# Keeping every failure of a five-million-document run in memory is itself a
# way to fail the run. The count stays exact; the retained sample is bounded,
# because the first hundred failures are enough to diagnose a systemic problem
# and a hundred-thousandth adds nothing a counter does not already say.
_MAX_RETAINED_ERRORS = 100


@dataclass
class IngestStats:
    """Live counters for the current process's ingest, plus the errors behind
    them.

    Deliberately a plain object rather than a scatter of module-level ints:
    the ingest report and the `/stats` endpoint both need to read this, and a
    value nobody can get a reference to is a value nobody can report. Counts
    are per-process and reset only when asked -- CocoIndex's memo layer means
    an unchanged document is not re-processed, so these describe work actually
    done, not the size of the corpus.
    """

    documents_succeeded: int = 0
    documents_failed: int = 0
    chunks_written: int = 0
    entities_written: int = 0
    relations_written: int = 0
    mentions_written: int = 0
    # Relations the extractor produced that could not name a chunk of the
    # document they came from. Counted rather than merely refused: a run that
    # reports few relations and a run that rejected most of the ones it found
    # leave an identical graph, and only one of them is a bug.
    relations_dropped_no_provenance: int = 0
    errors: list[IngestError] = field(default_factory=list)

    def record_success(self, chunks: int) -> None:
        self.documents_succeeded += 1
        self.chunks_written += chunks

    def record_semantics(self, elements: SemanticElements) -> None:
        self.entities_written += len(elements.entities)
        self.relations_written += len(elements.relations)
        self.mentions_written += len(elements.mentions)
        self.relations_dropped_no_provenance += elements.dropped_no_provenance

    def record_failure(self, error: IngestError) -> None:
        self.documents_failed += 1
        if len(self.errors) < _MAX_RETAINED_ERRORS:
            self.errors.append(error)

    def reset(self) -> None:
        self.documents_succeeded = 0
        self.documents_failed = 0
        self.chunks_written = 0
        self.entities_written = 0
        self.relations_written = 0
        self.mentions_written = 0
        self.relations_dropped_no_provenance = 0
        self.errors = []

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents_succeeded": self.documents_succeeded,
            "documents_failed": self.documents_failed,
            "chunks_written": self.chunks_written,
            "entities_written": self.entities_written,
            "relations_written": self.relations_written,
            "mentions_written": self.mentions_written,
            "relations_dropped_no_provenance": self.relations_dropped_no_provenance,
            "errors_retained": len(self.errors),
            "errors": [vars(e) for e in self.errors],
        }


INGEST_STATS = IngestStats()


@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    ensure_index()
    sink = AzureSearchSink()
    builder.provide(EMBEDDER, AzureOpenAIEmbedder())
    builder.provide(SEARCH_SINK, sink)
    settings = get_settings()
    builder.provide(SOURCE_DIR, settings.local_source_dir.resolve())
    # Same lazily-opened-once / closed-once-at-shutdown pattern as
    # `AzureSearchSink` above -- see `Neo4jGraphWriter`'s docstring.
    graph_writer = Neo4jGraphWriter() if settings.graph_enabled else None
    if graph_writer is not None:
        # Both before any document is read. The schema first, because every
        # MERGE below looks up a key it declares and doing that unindexed is
        # what turns ingest time quadratic in the corpus; then the department
        # layer, so the graph's top level reflects DEPARTMENTS itself rather
        # than whatever happened to be in the source folders this run.
        await graph_writer.ensure_schema()
        await graph_writer.ensure_departments()
    builder.provide(GRAPH_WRITER, graph_writer)
    builder.provide(REGISTRY, DocumentRegistry(settings.document_registry_path))

    # The gate for the whole semantic layer lives here and only here: an
    # extractor that is never built cannot be called, cannot open a cache
    # file, and cannot cost anything. `process_document` then reads "is there
    # an extractor?" instead of re-deciding policy per document.
    extractor = (
        RelationExtractor()
        if graph_writer is not None and settings.graph_extraction_enabled
        else None
    )
    builder.provide(EXTRACTOR, extractor)
    try:
        if settings.doc_source == "blob":
            from rag.sources import build_container_client

            async with build_container_client() as container:
                builder.provide(CONTAINER, container)
                yield
        else:
            builder.provide(CONTAINER, None)
            yield
    finally:
        if extractor is not None:
            # The run's own numbers, from the provider's usage blocks. Six of
            # the seven cost controls are claims about money and none of them
            # is worth anything unprinted.
            logger.info("Extraction: %s", extractor.stats.summary())
            logger.info("Extraction cost: %s", extractor.cost.summary())
        logger.info("Ingest: %s", INGEST_STATS.as_dict())
        await sink.aclose()
        if graph_writer is not None:
            await graph_writer.close()


async def parse_and_chunk(
    data: bytes, doc_id: str
) -> tuple[ParsedDocument, list[Chunk], DocumentMetadata]:
    """Everything the transform core produces, including the parse itself.

    `ingest_one` throws the `ParsedDocument` away, which is right for the
    vector-store path -- chunks and metadata are all it needs. The graph's
    semantic layer needs one thing more: a table's *cells*. Chunking renders a
    table to markdown, and the deterministic tabular extractor works off the
    row/column structure that rendering flattens. Re-deriving the cells by
    parsing the markdown back would be inventing a second, worse parser, so
    the real one's output is kept and handed on.
    """
    parsed = await select_parser().parse(data, doc_id)
    meta = extract_metadata(parsed, doc_id)
    return parsed, chunk_document(parsed, meta), meta


async def ingest_one(data: bytes, doc_id: str) -> tuple[list[Chunk], DocumentMetadata]:
    """Pure transform core: bytes -> chunks + metadata. No I/O to targets.

    Cross-document supersession (`Supersedes:` resolving to `is_current` /
    `superseded_by` on the *target* document) cannot be resolved here: CocoIndex
    processes each document independently with no barrier across documents, so
    a per-document call has only one `DocumentMetadata` in hand and nothing to
    link against. `extract_metadata` still parses the `Supersedes:` header for
    this document in isolation, which is correct and sufficient at this scope.
    The reverse links are resolved by a separate whole-corpus reconciliation
    pass (Task 13), not here.
    """
    _, chunks, meta = await parse_and_chunk(data, doc_id)
    return chunks, meta


def _collaborators() -> tuple[AzureSearchSink, AzureOpenAIEmbedder, Neo4jGraphWriter | None]:
    """The three targets `process_document` writes through, pulled out of the
    CocoIndex environment in one place.

    Isolated into its own function purely so it can be substituted: the
    environment only exists inside a running app, and a test that wants to
    exercise `process_document`'s own behaviour should not have to stand up
    an app to do it.
    """
    return (
        coco.use_context(SEARCH_SINK),
        coco.use_context(EMBEDDER),
        coco.use_context(GRAPH_WRITER),
    )


def _extractor() -> RelationExtractor | None:
    """The run's relationship extractor, or None when extraction is off.

    Separate from `_collaborators` rather than a fourth element of it. The
    three targets above are what every document is written to; this is an
    optional stage that most of the pipeline has no opinion about, and keeping
    the two apart means a test that only cares about failure handling can
    substitute one without knowing the other exists.
    """
    return coco.use_context(EXTRACTOR)


# --------------------------------------------------------------------------
# The semantic layer
# --------------------------------------------------------------------------


def _table_rows_by_unit(
    parsed: ParsedDocument, units: list[ExtractionUnit]
) -> dict[str, list[list[str]]]:
    """Pair each table unit back to the raw cells the parser read.

    `build_units` builds from `Chunk.display_text`, and for a table chunk that
    is the section caption followed by `Block.to_markdown()` -- so a block and
    a unit are the same table when the unit's text *contains* that block's
    rendering. Containment rather than equality on purpose: the caption exists
    for the reranker's benefit and is not this module's business, and a chunker
    that adds or changes decoration around the grid must not silently unroute
    every table in the corpus back to the LLM. Matching on the rendering rather
    than on position is what makes this survive a chunker that reorders,
    merges, or drops blocks.

    A table that finds no unit is one that never became a unit; it produces
    nothing here rather than being sent to the model, because the owner's
    direction is that structured data is read, not inferred.
    """
    remaining = [u for u in units if u.content_type == "table"]
    rows_by_unit: dict[str, list[list[str]]] = {}
    unmatched = 0
    for block in parsed.blocks:
        if block.type is not BlockType.TABLE or not block.rows:
            continue
        markdown = block.to_markdown()
        match = next((u for u in remaining if markdown in u.text), None)
        if match is None:
            unmatched += 1
            continue
        remaining.remove(match)
        rows_by_unit[match.unit_id] = block.rows
    if unmatched or remaining:
        # Loud, because the symptom otherwise is silent: the deterministic
        # path produces nothing, the table units fall through to triage, get
        # skipped as tables, and the document simply has fewer relations than
        # it should with nothing anywhere saying why.
        logger.warning(
            "%s: %d parsed table(s) matched no extraction unit and %d table unit(s) "
            "matched no parsed table -- their relations will not be extracted",
            parsed.doc_id, unmatched, len(remaining),
        )
    return rows_by_unit


def _mention_chunks(unit: ExtractionUnit, name: str,
                    text_by_chunk: dict[str, str]) -> list[str]:
    """Which of a unit's chunks should be recorded as mentioning `name`.

    A unit is a whole section and may span several chunks, so "the section
    mentions this entity" is true at a granularity `MENTIONS` does not have.
    Claiming every chunk in the section mentions it would make the edge say
    something false at chunk granularity -- and MENTIONS is what an
    entity-anchored retrieval follows back to text, so a false one sends a
    reader to a passage that does not contain what they searched for.

    So the chunks are filtered by whether they actually contain the surface
    form, falling back to the unit's first chunk when none does. The fallback
    is not a guess: the extractor is free to normalise a name ("PTO" for "paid
    time off (PTO)"), and an entity with no mention at all would be swept as
    unprovenanced the first time anything nearby was deleted.
    """
    needle = name.casefold()
    hits = [cid for cid in unit.chunk_ids if needle in text_by_chunk.get(cid, "").casefold()]
    return hits or unit.chunk_ids[:1]


async def extract_semantics(
    parsed: ParsedDocument,
    chunks: list[Chunk],
    meta: DocumentMetadata,
    extractor: RelationExtractor,
) -> tuple[list[Entity], list[Relation], list[tuple[str, Entity]]]:
    """One document's entities, relations and mentions, ready for the graph.

    The routing is the owner's direction made literal: **tables never reach
    the LLM.** A rate table is already relational, a model reading one can
    transpose a digit and a parser cannot, so table units go to
    `extraction.tabular` for free and exactly, and only prose is paid for.
    (`extraction.triage` would skip tables anyway; routing them explicitly
    means the deterministic path actually runs rather than the table simply
    being dropped.)

    Entity resolution runs per document and over the *declared* entities
    only, then rewrites the relations onto the canonical names it chose.
    Relation endpoints are deliberately not fed into it: on the tabular path
    an endpoint is a cell value ("$61.75", "20 days"), and merging near-
    identical values would fuse two rows of a rate card into one -- a silent
    corruption of exactly the data that path exists to keep exact. Endpoints
    still find their canonical node, because the name map is keyed on every
    surface form of every merged member, so a relation that said "PTO"
    resolves to the "Paid Time Off" node the declared entities produced.

    Per document is also all a per-document pipeline can do -- CocoIndex
    processes each file independently with no barrier -- but it is not the
    whole story and does not need to be: `Entity.compute_id()` is a pure
    function of (department, type, normalized name), so two documents that
    arrive at the same normalized name converge on one node with no
    coordination at all. What resolution adds on top is the variants
    normalization alone cannot see, inside the document where the evidence
    for that equivalence actually is.
    """
    units = build_units(chunks, meta)
    text_by_chunk = {c.compute_id(): c.display_text for c in chunks}
    unit_by_id = {u.unit_id: u for u in units}

    entities: list[Entity] = []
    relations: list[Relation] = []
    mentions: list[tuple[str, Entity]] = []

    def _record(unit: ExtractionUnit, found: list[Entity], edges: list[Relation]) -> None:
        entities.extend(found)
        relations.extend(edges)
        for entity in found:
            for chunk_id in _mention_chunks(unit, entity.name, text_by_chunk):
                mentions.append((chunk_id, entity))

    rows_by_unit = _table_rows_by_unit(parsed, units)
    for unit_id, rows in rows_by_unit.items():
        unit = unit_by_id[unit_id]
        _record(unit, *extract_from_table(rows, unit))

    prose_units = [u for u in units if u.unit_id not in rows_by_unit]
    for result in await extractor.extract(prose_units):
        unit = unit_by_id[result.unit_id]
        _record(unit, result.entities, result.relations)

    resolution = resolve_entities(entities)
    canonical = {(e.department, e.type, e.name): e for e in resolution.entities}

    def _canonical(entity: Entity) -> Entity:
        name = resolution.canonical_name(entity.department, entity.type, entity.name)
        return canonical.get((entity.department, entity.type, name), entity)

    return (
        resolution.entities,
        remap_relations(relations, resolution),
        [(chunk_id, _canonical(entity)) for chunk_id, entity in mentions],
    )


@coco.fn(memo=True)
async def process_document(file: FileLike) -> int:
    """Memoized per (content, code): re-parses and re-embeds only on real change.

    `file` is typed as `FileLike` rather than `localfs.File` so this same
    function serves any source connector whose file objects derive from
    `FileLike` and expose `file_path.path` + `await read()` -- e.g. Task 11's
    `AzureBlobFile`.

    Because this function only runs at all on a genuine content change (memo
    skips it otherwise), an in-place content change means the previous run's
    chunks for this `doc_id` may no longer match: `Chunk.compute_id()` hashes
    section path/text/position, so if the new content reshapes sections the
    new chunk set has different keys than the old one. `AzureSearchSink.upsert`
    is a keyed merge_or_upload, not a doc-scoped replace, so it would leave
    those old, now-stale chunks behind forever. Deleting this doc_id's chunks
    before writing the fresh set keeps the index doc-consistent; it costs one
    extra search+delete round trip per *changed* document, not per document.

    **Nothing escapes this function.** A corrupt file, a password-protected
    PDF, a format nobody anticipated, a transient 429 -- any of them raising
    out of here would take down the whole run, and at five million documents
    some fraction of the corpus is guaranteed to be one of those. The failure
    is recorded as an `IngestError` naming the stage it happened in, counted
    on `INGEST_STATS`, logged at ERROR, and the document is reported as zero
    chunks written so the rest of the run continues. The stage matters: a
    corpus failing at `parse` is a format-coverage problem, one failing at
    `embed` or `index` is a service problem, and the counter alone cannot
    tell those apart.

    **Three stores, one document, one failure boundary.** The graph's
    structural layer mirrors what went to the search index; on top of it, the
    semantic layer (entities and typed relations) is extracted only when an
    extractor was provided, which is the single place `graph_extraction_
    enabled` is consulted. All three writes sit inside the same handler, so a
    Neo4j hiccup or an extraction failure degrades one document rather than
    stopping the run -- and the `stage` recorded says which of them it was,
    because "extract" (a model or quota problem) and "graph" (a database
    problem) call for completely different responses.

    Because the result is memoized, a document that fails is not retried
    until its bytes change or the memo is invalidated -- which is the right
    default (retrying an unparseable file every run costs the same and fails
    the same way) and the reason the error list, not just the count, has to
    survive the run.
    """
    doc_id = file.file_path.path.as_posix()
    stage = "context"
    try:
        sink, embedder, graph_writer = _collaborators()

        stage = "read"
        data = await file.read()

        stage = "parse"
        parsed, chunks, meta = await parse_and_chunk(data, doc_id)

        stage = "index"
        await sink.delete_document(doc_id)
        if not chunks:
            INGEST_STATS.record_success(0)
            return 0

        stage = "embed"
        vectors = await embedder.embed([c.embed_text for c in chunks])

        stage = "index"
        count = await sink.upsert(chunks, vectors, meta)

        if graph_writer is not None:
            from rag.targets.graph import build_graph_elements, build_semantic_elements

            stage = "graph"
            structure = build_graph_elements(chunks, meta)
            # The entities the previous pass's chunks were the evidence for.
            # They can only be identified before those chunks are replaced,
            # which is why `write` hands them back rather than sweeping them
            # itself -- whether they are orphaned depends on what the
            # extraction below produces, which has not happened yet.
            orphan_candidates = await graph_writer.write(structure)

            extractor = _extractor()
            if extractor is not None:
                # Inside the same handler as everything else, and deliberately
                # after the structural write. Extraction is the one stage that
                # depends on an external model's judgement, so it is also the
                # one most likely to fail in a way nobody anticipated -- and
                # when it does, the right outcome is a document that is
                # searchable and structurally present in the graph but missing
                # its semantic layer, not a document that is missing entirely.
                # Degrading a document beats failing a run.
                stage = "extract"
                entities, relations, mentions = await extract_semantics(
                    parsed, chunks, meta, extractor
                )

                stage = "graph"
                semantic = build_semantic_elements(structure, entities, relations, mentions)
                await graph_writer.write_semantics(
                    semantic, orphan_candidates=orphan_candidates
                )
                INGEST_STATS.record_semantics(semantic)

        # Recorded after a successful write, never before: a document that
        # failed to index must not sit in the registry as though it had, or
        # the version pass would resolve supersession against a document that
        # is not actually retrievable.
        #
        # Guarded, and deliberately outside the stage handler below. By this
        # point the document IS indexed and IS in the graph; letting a registry
        # write fail the call would report a successful ingest as a failure and
        # invite a retry of work already done. The cost of losing this row is
        # that the document does not participate in version resolution until
        # its next ingest -- a degraded ranking signal, not a missing document.
        try:
            coco.use_context(REGISTRY).upsert(meta)
        except Exception:
            logger.warning(
                "Document registry unavailable for %s; version resolution will "
                "skip it until the next ingest", doc_id, exc_info=True,
            )

        INGEST_STATS.record_success(count)
        return count
    except Exception as exc:
        error = IngestError(doc_id=doc_id, stage=stage,
                            error_type=type(exc).__name__, message=str(exc))
        logger.error(
            "Ingest failed: doc_id=%s stage=%s %s: %s",
            error.doc_id, error.stage, error.error_type, error.message,
            exc_info=True,
        )
        INGEST_STATS.record_failure(error)
        return 0


_KNOWN_DOC_IDS_STATE_KEY = "known_doc_ids"


async def _current_doc_ids(matcher: PatternFilePathMatcher) -> set[str]:
    """One-shot (non-live) listing of doc_ids currently present in the configured source."""
    settings = get_settings()
    if settings.doc_source == "blob":
        from cocoindex.connectors import azure_blob

        scan = azure_blob.list_blobs(
            coco.use_context(CONTAINER),
            prefix=settings.azure_blob_prefix,
            path_matcher=matcher,
        )
    else:
        scan = localfs.walk_dir(SOURCE_DIR, live=False, recursive=True, path_matcher=matcher)
    return {key async for key, _ in scan.items()}


async def apply_deletions(
    doc_ids: Iterable[str],
    *,
    sink: Any,
    graph_writer: Any | None,
    registry: Any | None = None,
) -> int:
    """Remove each doc_id from every store that holds it.

    Split out from `_reconcile_deletions` because the two halves fail
    differently: working out *what* disappeared needs a live source scan and
    CocoIndex's persisted state, while removing it needs nothing but the
    targets. Keeping the removal addressable on its own is what lets the
    delete path be proven against the real Azure AI Search index and the real
    Neo4j instance without standing up a CocoIndex app.

    Both stores are always visited. Deleting from the vector store alone --
    the behaviour this replaces -- leaves the document's Document/Section/
    Chunk subgraph in Neo4j forever, so a graph-anchored answer keeps citing a
    document that no longer exists anywhere else. A partial delete is worse
    than no delete, because it is invisible.
    """
    removed = 0
    for doc_id in doc_ids:
        await sink.delete_document(doc_id)
        if graph_writer is not None:
            await graph_writer.delete_document(doc_id)
        if registry is not None:
            # Otherwise a deleted document keeps voting in the next version
            # reconciliation, and can supersede a document that still exists.
            registry.delete(doc_id)
        removed += 1
    return removed


async def _reconcile_deletions(matcher: PatternFilePathMatcher) -> None:
    """Delete a removed document's chunks from Azure AI Search AND its
    subgraph from Neo4j.

    `AzureSearchSink` has no built-in target reconciliation (see its module
    docstring: CocoIndex has no public custom-target SDK, so writes are
    memoized but not auto-reconciled). `mount_each`'s own "N deleted" stat is
    just mount-lifecycle bookkeeping -- it never calls `process_document` for a
    removed key, so nothing upstream ever calls `delete_document` on its own.
    This does that explicitly: it takes a fresh one-shot (non-live) scan of
    the configured source (local directory or Blob container), diffs it
    against the set of doc_ids persisted from the previous run via
    `coco.use_state`, and deletes whatever dropped out.

    Runs once per `app_main` invocation -- i.e. once per catch-up `update`,
    and once at the start of a `--live` session. A deletion that happens
    *during* a long-running `--live` session is picked up on the next
    invocation, not immediately: for localfs the live watcher does surface
    per-key delete events directly to `LiveBlobItems`/`_LiveDirItems`'s
    subscriber (see `rag.sources.azure_blob_live.LiveBlobItems.watch`, which
    calls `subscriber.delete(key)` on a dropped ETag), but that only detaches
    the mounted `process_document` component for that key -- it does not
    call `AzureSearchSink.delete_document`. Only this explicit reconciliation
    does, and it is not re-run mid-session.
    """
    current_ids = await _current_doc_ids(matcher)

    state = coco.use_state(_KNOWN_DOC_IDS_STATE_KEY, initial_value=[])
    previously_known = set(state.value or [])

    removed = await apply_deletions(
        sorted(previously_known - current_ids),
        sink=coco.use_context(SEARCH_SINK),
        graph_writer=coco.use_context(GRAPH_WRITER),
        registry=coco.use_context(REGISTRY),
    )
    if removed:
        logger.info("Reconciled %d source deletion(s) out of both stores", removed)

    state.value = sorted(current_ids)


@coco.fn
async def app_main() -> None:
    settings = get_settings()
    matcher = PatternFilePathMatcher(included_patterns=settings.included_patterns)
    if settings.doc_source == "blob":
        from rag.sources import build_source

        blob_items = build_source(coco.use_context(CONTAINER), path_matcher=matcher)
        await coco.mount_each(process_document, blob_items)
    else:
        files = localfs.walk_dir(
            SOURCE_DIR,
            live=True,
            recursive=True,
            rescan_interval=settings.rescan_interval,
            path_matcher=matcher,
        )
        await coco.mount_each(process_document, files.items())
    await _reconcile_deletions(matcher)
    await _reconcile_versions()


async def _reconcile_versions() -> None:
    """Resolve supersession across the corpus and stamp it onto the index.

    Runs after deletions, so a document removed in this pass cannot be
    resolved as the successor of anything. Cheap in the steady state: it
    computes the delta against what was last written and touches nothing when
    nothing changed, which is what makes it safe to run after every ingest.

    This is the pass that gives the vector store a version signal at all.
    Without it every chunk carries `is_current = true`, including a superseded
    rate card, and a pricing question becomes a coin flip between two answers
    that are each correct for a different year.
    """
    try:
        summary = await reconcile_versions(
            coco.use_context(REGISTRY), coco.use_context(SEARCH_SINK)
        )
    except Exception:
        # A failure here leaves the index without a version signal, which
        # degrades ranking on version-sensitive questions but breaks nothing
        # else -- so it must not take down an otherwise successful ingest.
        # It is logged loudly because a silent loss of the signal is exactly
        # the failure this pass exists to prevent.
        logger.exception("Version reconciliation failed; index version flags are stale")
        return
    logger.info("Version reconciliation: %s", summary)


app = coco.App(coco.AppConfig(name="RagIngestion"), app_main)
