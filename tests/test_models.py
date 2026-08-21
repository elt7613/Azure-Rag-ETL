from rag.models import Block, BlockType, Chunk, DocumentMetadata


def test_chunk_id_is_deterministic_content_hash():
    meta = DocumentMetadata(doc_id="sales/Pricing2026.pdf", title="Pricing 2026",
                            department="sales", version="1.0")
    a = Chunk(doc_id=meta.doc_id, section_path=["3. Tiers"], display_text="Starter $32",
              embed_text="x", content_type="table", page=1, chunk_index=0)
    b = Chunk(doc_id=meta.doc_id, section_path=["3. Tiers"], display_text="Starter $32",
              embed_text="x", content_type="table", page=1, chunk_index=0)
    assert a.compute_id() == b.compute_id()
    assert len(a.compute_id()) == 40


def test_chunk_id_changes_with_content():
    base = dict(doc_id="d", section_path=["1"], embed_text="x",
                content_type="prose", page=1, chunk_index=0)
    a = Chunk(display_text="one", **base)
    b = Chunk(display_text="two", **base)
    assert a.compute_id() != b.compute_id()


def test_block_table_carries_rows_and_markdown():
    b = Block(type=BlockType.TABLE, text="", page=1,
              rows=[["Tier", "Price"], ["Starter", "$32"]])
    assert b.to_markdown().startswith("| Tier | Price |")
