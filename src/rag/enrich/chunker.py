"""Chunk by walking the section tree.

Two text fields per chunk, deliberately:
  embed_text   — breadcrumb-prefixed, what gets vectorized
  display_text — clean, what reaches the LLM context and citations
Conflating them degrades both retrieval and answer quality.
"""
from __future__ import annotations

import tiktoken

from rag.config import get_settings
from rag.enrich.structure import build_section_tree, iter_leaf_sections
from rag.models import Block, BlockType, Chunk, DocumentMetadata, ParsedDocument


def _encoder():
    return tiktoken.get_encoding(get_settings().tokenizer_encoding)


def _breadcrumb(meta: DocumentMetadata, path: list[str]) -> str:
    version = f", v{meta.version}" if meta.version else ""
    head = f"{meta.title} ({meta.department}{version})"
    return " > ".join([head, *path])


def _split_prose(text: str, target: int, overlap: int, enc) -> list[str]:
    tokens = enc.encode(text)
    if len(tokens) <= target:
        return [text]
    out: list[str] = []
    start = 0
    step = max(target - overlap, 1)
    while start < len(tokens):
        out.append(enc.decode(tokens[start : start + target]))
        start += step
    return out


def chunk_document(doc: ParsedDocument, meta: DocumentMetadata) -> list[Chunk]:
    settings = get_settings()
    enc = _encoder()
    chunks: list[Chunk] = []
    index = 0

    leaves = list(iter_leaf_sections(build_section_tree(doc)))
    carry_over = ""  # prose from a non-citable "Preamble" leaf, folded into the next section

    for i, (path, section) in enumerate(leaves):
        crumb = _breadcrumb(meta, path)
        tables = [b for b in section.blocks if b.type is BlockType.TABLE]
        prose = [b for b in section.blocks if b.type is not BlockType.TABLE]

        # Tables are never split: one chunk each, full markdown grid preserved.
        #
        # The grid is captioned with its section title, and that caption is part
        # of `display_text` rather than only of `embed_text`. The reason is the
        # reranker: Azure's cross-encoder scores the *stored content*, and a
        # bare grid of "| Expense Amount | Required Approver |" reads as
        # structureless next to a prose paragraph that says the words
        # "approval" and "expense" in a sentence. Measured on this corpus, the
        # Approval Matrix table ranked below three prose sections of its own
        # document for "who approves an expense of $3,000" -- the table holding
        # the literal answer, outranked by the document's Purpose section.
        # Captioning also helps the answer prompt and the reader, since a
        # quoted grid with no title is ambiguous on its own.
        for table in tables:
            markdown = table.to_markdown()
            caption = f"{section.number} {section.title}".strip() or meta.title
            captioned = f"{caption}\n{markdown}"
            chunks.append(Chunk(
                doc_id=meta.doc_id, section_path=path,
                display_text=captioned,
                embed_text=f"{crumb}\n{captioned}",
                content_type="table", page=table.page,
                chunk_index=index, section_number=section.number,
            ))
            index += 1

        body = "\n".join(b.text for b in prose if b.text).strip()

        # A "Preamble" leaf (structure.py's fallback for content before any
        # heading) has no section number, so it's never independently
        # citable -- typically just the bare document title. Emitting it as
        # its own chunk produces a near-empty orphan that only duplicates
        # what the breadcrumb already carries. Fold it into the following
        # section instead, unless it's the only content in the document.
        is_uncitable_preamble = section.number == "" and section.title == "Preamble"
        if is_uncitable_preamble and i + 1 < len(leaves):
            carry_over = body
            continue

        if carry_over:
            body = f"{carry_over}\n{body}".strip() if body else carry_over
            carry_over = ""

        if not body:
            continue
        for piece in _split_prose(body, settings.chunk_target_tokens,
                                  settings.chunk_overlap_tokens, enc):
            chunks.append(Chunk(
                doc_id=meta.doc_id, section_path=path,
                display_text=piece,
                embed_text=f"{crumb}\n{piece}",
                content_type="prose", page=section.page,
                chunk_index=index, section_number=section.number,
            ))
            index += 1

    # Link neighbours so retrieval can expand context without a graph hop.
    ids = [c.compute_id() for c in chunks]
    for i, chunk in enumerate(chunks):
        chunk.prev_chunk_id = ids[i - 1] if i > 0 else ""
        chunk.next_chunk_id = ids[i + 1] if i + 1 < len(ids) else ""
    return chunks
