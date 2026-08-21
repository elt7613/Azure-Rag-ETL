async def test_ingest_one_produces_chunks_with_metadata():
    from rag.etl.app import ingest_one

    data = open("source_data/sales/Pricing2026.pdf", "rb").read()
    chunks, meta = await ingest_one(data, "sales/Pricing2026.pdf")
    assert chunks
    assert meta.department == "sales"
    assert meta.version == "1.0"
    assert all(c.embed_text.startswith(meta.title) for c in chunks)


async def test_ingest_is_idempotent():
    from rag.etl.app import ingest_one

    data = open("source_data/HR/LeavePolicy.pdf", "rb").read()
    first, _ = await ingest_one(data, "HR/LeavePolicy.pdf")
    second, _ = await ingest_one(data, "HR/LeavePolicy.pdf")
    assert [c.compute_id() for c in first] == [c.compute_id() for c in second]


async def test_ingest_one_does_not_link_versions_across_documents():
    """ingest_one is per-document and pure: CocoIndex processes each document
    independently with no barrier across documents, so calling `link_versions`
    on a single document inside `ingest_one` would be a no-op. Real corpus
    documents where Pricing2026 supersedes Pricing2025 (see test_metadata.py)
    must NOT come back cross-linked when ingested one at a time here -- that
    reconciliation is Task 13's job, run once over the whole corpus."""
    from rag.etl.app import ingest_one

    data_2025 = open("source_data/sales/Pricing2025.pdf", "rb").read()
    data_2026 = open("source_data/sales/Pricing2026.pdf", "rb").read()
    _, meta_2025 = await ingest_one(data_2025, "sales/Pricing2025.pdf")
    _, meta_2026 = await ingest_one(data_2026, "sales/Pricing2026.pdf")

    assert meta_2025.is_current is True
    assert meta_2025.superseded_by == ""
    assert meta_2026.supersedes  # still parsed per-document, just not linked
