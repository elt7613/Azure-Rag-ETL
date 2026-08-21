"""Smart LLM relationship extraction -- the machinery that makes it affordable.

The owner's requirement is that the semantic graph is built by LLM extraction
for *any* document type. At 5M documents of 100-200 pages, the naive reading of
that requirement -- one call per chunk -- is not expensive, it is impossible.
This package is everything that decides what never needs sending, and what
never needs sending twice:

    ontology  closed entity/relation vocabularies + the strict JSON schema
              that constrains the model to them
    units     section-level extraction units (one call per section, not per
              chunk) and token-budget packing (many small sections per call)
    triage    the gate: tables, too-short, boilerplate, and low-signal units
              never reach the model at all
    cache     content-hash memo store, so a re-ingest, an unchanged section
              inside a changed document, and the same clause appearing in 400
              documents are all free after the first

On top of those four sit the three stages that actually produce the graph:

    llm       the extractor -- triage -> cache -> pack -> one call per pack,
              strict structured outputs, evidence spans checked against the
              source, and a truncated pack halved and retried rather than
              thrown away
    tabular   the deterministic path tables take instead, per the owner's
              direction: a rate card is already relational, and a parser
              cannot transpose a digit the way a model can
    resolve   entity resolution -- normalize, block, score, merge -- so "PTO"
              and "Paid Time Off" end up one node without an LLM call per pair

The Batch API path (`batch.py`) reuses the same pieces for bulk backfill.
"""
from __future__ import annotations

from rag.extraction.cache import CacheStats, ExtractionCache
from rag.extraction.llm import (
    MAX_COMPLETION_TOKENS,
    PROMPT_CACHE_MIN_TOKENS,
    SYSTEM_PROMPT,
    ExtractionStats,
    RelationExtractor,
    TruncatedResponse,
    build_messages,
    evidence_supported,
)
from rag.extraction.ontology import (
    ENTITY_TYPES,
    RELATION_TYPES,
    build_extraction_schema,
    coerce_entity_type,
    coerce_relation_type,
    is_valid_entity_type,
    is_valid_relation_type,
    response_format,
)
from rag.extraction.triage import (
    BoilerplateIndex,
    SignalCounts,
    SkipReason,
    TriageDecision,
    signal_components,
    signal_score,
    triage_unit,
    triage_units,
)
from rag.extraction.resolve import (
    AmbiguousPair,
    ResolutionResult,
    fuzzy_score,
    generate_candidate_pairs,
    normalize,
    remap_relations,
    resolve_entities,
    surface_forms,
)
from rag.extraction.tabular import CellValue, classify_cell_value, extract_from_table
from rag.extraction.units import (
    UNIT_FRAMING_TOKENS,
    UnitPack,
    build_units,
    count_tokens,
    pack_units,
)

__all__ = [
    "AmbiguousPair",
    "BoilerplateIndex",
    "CacheStats",
    "CellValue",
    "ENTITY_TYPES",
    "ExtractionCache",
    "ExtractionStats",
    "MAX_COMPLETION_TOKENS",
    "PROMPT_CACHE_MIN_TOKENS",
    "RELATION_TYPES",
    "RelationExtractor",
    "ResolutionResult",
    "SYSTEM_PROMPT",
    "SignalCounts",
    "SkipReason",
    "TriageDecision",
    "TruncatedResponse",
    "UNIT_FRAMING_TOKENS",
    "UnitPack",
    "build_extraction_schema",
    "build_messages",
    "build_units",
    "classify_cell_value",
    "coerce_entity_type",
    "coerce_relation_type",
    "count_tokens",
    "evidence_supported",
    "extract_from_table",
    "fuzzy_score",
    "generate_candidate_pairs",
    "is_valid_entity_type",
    "is_valid_relation_type",
    "normalize",
    "pack_units",
    "remap_relations",
    "resolve_entities",
    "response_format",
    "signal_components",
    "signal_score",
    "surface_forms",
    "triage_unit",
    "triage_units",
]
