"""Domain types shared by every ETL stage."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"


@dataclass
class Block:
    """Format-agnostic unit emitted by every parser."""
    type: BlockType
    text: str
    page: int
    level: int = 0
    rows: list[list[str]] | None = None

    def to_markdown(self) -> str:
        if self.type is not BlockType.TABLE or not self.rows:
            return self.text
        header, *body = self.rows
        cells = lambda r: "| " + " | ".join(c or "" for c in r) + " |"
        sep = "| " + " | ".join("---" for _ in header) + " |"
        return "\n".join([cells(header), sep, *(cells(r) for r in body)])


@dataclass
class DocumentMetadata:
    doc_id: str
    title: str
    department: str
    version: str = ""
    owner: str = ""
    effective_from: date | None = None
    effective_to: date | None = None
    supersedes: str = ""
    superseded_by: str = ""
    is_current: bool = True


@dataclass
class ParsedDocument:
    doc_id: str
    file_format: str
    blocks: list[Block]
    raw_header: str = ""


@dataclass
class Section:
    number: str
    title: str
    level: int
    page: int
    blocks: list[Block] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)

    def path(self, prefix: list[str] | None = None) -> list[str]:
        return (prefix or []) + [f"{self.number} {self.title}".strip()]


@dataclass
class Chunk:
    doc_id: str
    section_path: list[str]
    display_text: str
    embed_text: str
    content_type: str
    page: int
    chunk_index: int
    section_number: str = ""
    prev_chunk_id: str = ""
    next_chunk_id: str = ""

    def compute_id(self) -> str:
        payload = "\x1f".join([
            self.doc_id, "/".join(self.section_path),
            self.display_text, self.content_type,
            str(self.page), str(self.chunk_index),
        ])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Semantic graph layer
#
# Everything below describes the *extracted* graph -- entities and typed
# relations -- as opposed to the structural graph (Document/Section/Chunk)
# built deterministically from parsing. The defining rule here is provenance:
# a relation that cannot name the chunk it came from is not written. An LLM
# can invent a relationship; a relation carrying a resolvable
# `source_chunk_id`, page, and evidence span can be checked against the
# source, which is what makes the extracted layer auditable rather than
# merely plausible.
# --------------------------------------------------------------------------


@dataclass
class Entity:
    """A canonical thing the corpus talks about (a policy, role, benefit, rate).

    `entity_id` is derived from (department, type, normalized name) so the
    same entity mentioned in two documents of the same department resolves to
    one node. `aliases` accumulates the surface forms that resolved into it.
    """
    name: str
    type: str
    department: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""

    def normalized(self) -> str:
        return " ".join(self.name.lower().replace("-", " ").split())

    def compute_id(self) -> str:
        payload = "\x1f".join([self.department, self.type, self.normalized()])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class Relation:
    """A typed edge between two entities, with the evidence that produced it."""
    subject: str          # entity name as written in the source
    predicate: str        # must be a member of the closed relation vocabulary
    object: str
    subject_type: str = ""
    object_type: str = ""
    doc_id: str = ""
    source_chunk_id: str = ""
    section_path: str = ""
    page: int = 0
    department: str = ""
    confidence: float = 0.0
    evidence_span: str = ""
    # False for LLM-extracted relations, True for ones derived deterministically
    # from a table. Lets retrieval and the UI distinguish "the model read this"
    # from "this was read off a spreadsheet".
    deterministic: bool = False

    def compute_id(self) -> str:
        payload = "\x1f".join([
            self.doc_id, self.source_chunk_id,
            self.subject.lower(), self.predicate, self.object.lower(),
        ])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class ExtractionUnit:
    """One coherent span of a document offered to the extractor.

    A unit is a *section*, not a chunk: a section spanning five chunks costs
    one extraction call rather than five, and the extractor sees the whole
    thought instead of a window into it.
    """
    unit_id: str
    doc_id: str
    department: str
    section_path: str
    text: str
    page: int
    chunk_ids: list[str] = field(default_factory=list)
    content_type: str = "prose"

    def content_hash(self) -> str:
        normalized = " ".join(self.text.split()).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class ExtractionResult:
    """What one extraction call produced, plus what it cost."""
    unit_id: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    from_cache: bool = False
    skipped_reason: str = ""


@dataclass
class IngestError:
    """A per-document failure that must not stop the run.

    At 5M documents, some fraction will be corrupt, password-protected, or a
    format nobody anticipated. The pipeline records those and keeps going;
    aborting the batch on one bad file is the wrong failure mode.
    """
    doc_id: str
    stage: str
    error_type: str
    message: str
