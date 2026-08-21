"""Extraction units: what the LLM is actually shown, and how many at a time.

Two of the seven cost controls live here.

**Section-level units.** Chunking optimises for retrieval -- ~700 tokens with
overlap, so a matching chunk is small enough to be precise and big enough to
stand alone. Extraction wants the opposite: the whole coherent thought, once.
A section that chunked into five overlapping windows would otherwise cost five
calls, four of them re-reading text they already saw, and would produce five
partial views of one obligation that entity resolution then has to sew back
together. Rebuilding the section from its chunks costs one call and the
extractor sees the whole rule.

**Packing.** The fixed prompt prefix (ontology, rules, examples) is >=1024
tokens by design so prompt caching engages. Even cached, that prefix is paid
per call, so sending a 60-token section on its own means the overhead dwarfs
the payload. Packing amortises it across as many small units as fit under
`GRAPH_EXTRACT_PACK_TOKENS`, with each unit's id carried into the response
schema so attribution survives the batching.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache

import tiktoken

from rag.config import get_settings
from rag.models import Chunk, DocumentMetadata, ExtractionUnit

# Rendering one unit into the user message costs more than its body: an id
# line, a section breadcrumb, and delimiters. Charged per unit so a pack of
# forty tiny units does not quietly overrun the budget by its own scaffolding.
UNIT_FRAMING_TOKENS = 16


@lru_cache(maxsize=4)
def _encoder(name: str):
    return tiktoken.get_encoding(name)


def count_tokens(text: str) -> int:
    """Token count under the project's configured encoding.

    The same encoding the chunker uses, so a "700-token chunk" and a
    "3000-token pack" are measured on one ruler.
    """
    return len(_encoder(get_settings().tokenizer_encoding).encode(text))


def _unit_id(doc_id: str, section_path: str, content_type: str, ordinal: int) -> str:
    payload = "\x1f".join([doc_id, section_path, content_type, str(ordinal)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_units(chunks: list[Chunk], meta: DocumentMetadata) -> list[ExtractionUnit]:
    """Group a document's chunks into section-level extraction units.

    Prose and tables in the same section become *separate* units, and each
    table becomes a unit of its own: tables are handled deterministically
    (`extraction.tabular`), so folding a pricing grid into the prose unit
    would both pay LLM prices for structured data and dilute the prose's
    signal score. Keeping them apart lets triage route each to the right path.

    Identical text repeated inside one document collapses to a single unit
    that credits every chunk it came from -- a running footer that the parser
    could not strip is extracted once, not once per page.
    """
    grouped: dict[tuple[str, str, int], list[Chunk]] = {}
    for chunk in chunks:
        section_path = " > ".join(chunk.section_path)
        # Tables get a unique discriminator (the chunk index) so two tables in
        # one section stay two units; prose shares a slot and is concatenated.
        discriminator = chunk.chunk_index if chunk.content_type == "table" else -1
        grouped.setdefault((section_path, chunk.content_type, discriminator), []).append(chunk)

    units: list[ExtractionUnit] = []
    by_hash: dict[str, ExtractionUnit] = {}
    for ordinal, ((section_path, content_type, _), members) in enumerate(grouped.items()):
        text = "\n".join(c.display_text for c in members if c.display_text.strip()).strip()
        if not text:
            continue
        unit = ExtractionUnit(
            unit_id=_unit_id(meta.doc_id, section_path, content_type, ordinal),
            doc_id=meta.doc_id,
            department=meta.department,
            section_path=section_path,
            text=text,
            page=min(c.page for c in members),
            chunk_ids=[c.compute_id() for c in members],
            content_type=content_type,
        )
        existing = by_hash.get(unit.content_hash())
        if existing is not None:
            existing.chunk_ids.extend(unit.chunk_ids)
            continue
        by_hash[unit.content_hash()] = unit
        units.append(unit)
    return units


@dataclass
class UnitPack:
    """One extraction call's worth of units.

    `oversized` marks the one case the budget cannot hold: a single unit whose
    own body exceeds it. Such a unit is sent alone rather than split, because
    splitting a section reintroduces exactly the partial-view problem
    section-level units exist to avoid. The caller can see the flag and decide
    (truncate, downgrade the model, or accept the cost) instead of discovering
    it as a context-length error mid-run.
    """
    units: list[ExtractionUnit] = field(default_factory=list)
    token_count: int = 0
    oversized: bool = False

    @property
    def unit_ids(self) -> list[str]:
        return [u.unit_id for u in self.units]


def pack_units(units: list[ExtractionUnit], budget: int | None = None) -> list[UnitPack]:
    """Group `units` into packs of at most `budget` tokens, order preserved.

    The guarantee is "never merge past the budget": every pack holding two or
    more units is within budget. A lone oversized unit is the sole exception
    and is flagged as such.
    """
    limit = budget if budget is not None else get_settings().graph_extract_pack_tokens
    if limit <= 0:
        raise ValueError(f"pack budget must be positive, got {limit}")

    packs: list[UnitPack] = []
    current = UnitPack()
    for unit in units:
        cost = count_tokens(unit.text) + UNIT_FRAMING_TOKENS
        if cost > limit:
            # Flush what we have, then emit the giant on its own so it never
            # drags a small unit over the line with it.
            if current.units:
                packs.append(current)
                current = UnitPack()
            packs.append(UnitPack(units=[unit], token_count=cost, oversized=True))
            continue
        if current.units and current.token_count + cost > limit:
            packs.append(current)
            current = UnitPack()
        current.units.append(unit)
        current.token_count += cost
    if current.units:
        packs.append(current)
    return packs
