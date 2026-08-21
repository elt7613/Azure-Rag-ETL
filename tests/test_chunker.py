from rag.enrich.chunker import chunk_document
from rag.models import Block, BlockType, DocumentMetadata, ParsedDocument

META = DocumentMetadata(doc_id="sales/Pricing2026.pdf", title="Pricing 2026",
                        department="sales", version="1.0")


def test_embed_text_carries_breadcrumb_display_text_does_not():
    doc = ParsedDocument(doc_id=META.doc_id, file_format="pdf", blocks=[
        Block(type=BlockType.HEADING, text="3 Subscription Tiers", page=1, level=1),
        Block(type=BlockType.PARAGRAPH, text="Starter is $32 per seat.", page=1),
    ])
    chunk = chunk_document(doc, META)[0]
    assert chunk.embed_text.startswith("Pricing 2026 (sales, v1.0)")
    assert "3 Subscription Tiers" in chunk.embed_text
    assert chunk.display_text == "Starter is $32 per seat."


def test_table_is_never_split_and_kept_as_markdown():
    rows = [["Tier", "Price"]] + [[f"T{i}", f"${i}"] for i in range(200)]
    doc = ParsedDocument(doc_id=META.doc_id, file_format="pdf", blocks=[
        Block(type=BlockType.HEADING, text="3 Tiers", page=1, level=1),
        Block(type=BlockType.TABLE, text="", page=1, rows=rows),
    ])
    tables = [c for c in chunk_document(doc, META) if c.content_type == "table"]
    assert len(tables) == 1
    assert "| Tier | Price |" in tables[0].display_text


def test_oversized_prose_splits_with_neighbour_links():
    long_para = " ".join(f"Sentence number {i} about policy." for i in range(400))
    doc = ParsedDocument(doc_id=META.doc_id, file_format="pdf", blocks=[
        Block(type=BlockType.HEADING, text="1 Overview", page=1, level=1),
        Block(type=BlockType.PARAGRAPH, text=long_para, page=1),
    ])
    chunks = chunk_document(doc, META)
    assert len(chunks) > 1
    assert chunks[0].next_chunk_id == chunks[1].compute_id()
    assert chunks[1].prev_chunk_id == chunks[0].compute_id()
