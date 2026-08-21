"""Corpus-wide supersession resolution.

The failure this closes: every chunk in the index carried `is_current = true`,
including the superseded 2025 rate card, because supersession is a fact about a
*pair* of documents and the per-document ingest path only ever holds one.
"""
from __future__ import annotations

from datetime import date

import pytest

from tests.conftest import azure_configured

from rag.models import DocumentMetadata
from rag.targets.version_sync import (
    DocumentRegistry,
    compute_changes,
    reconcile_versions,
)


def pricing(year: int) -> DocumentMetadata:
    return DocumentMetadata(
        doc_id=f"sales/Pricing{year}.pdf",
        title=f"Pricing{year}",
        department="sales",
        version="1.4" if year == 2025 else "1.0",
        effective_from=date(year, 1, 1),
        supersedes="2025 Rate Card (v1.4)" if year == 2026 else "",
    )


@pytest.fixture
def registry(tmp_path):
    return DocumentRegistry(tmp_path / "registry.db")


# ---------------- the registry ----------------


def test_registry_round_trips_metadata(registry):
    registry.upsert(pricing(2026))
    stored = registry.all()
    assert len(stored) == 1
    assert stored[0].doc_id == "sales/Pricing2026.pdf"
    assert stored[0].supersedes == "2025 Rate Card (v1.4)"
    assert stored[0].effective_from == date(2026, 1, 1)


def test_upsert_is_idempotent(registry):
    registry.upsert(pricing(2026))
    registry.upsert(pricing(2026))
    assert registry.count() == 1


def test_reingest_resets_resolved_status_so_it_is_restamped(registry):
    """A re-ingest must reset the flag, because the index write resets it too.

    This looks backwards and is the whole point. The ingest that re-upserts
    here also rewrites the document's chunks in the search index with
    `is_current` at its per-document default. If the registry kept the
    resolved value, reconciliation would compare against its own stale record,
    conclude nothing changed, write nothing -- and leave the index claiming a
    superseded rate card is current. Silently.
    """
    registry.upsert(pricing(2025))
    registry.upsert(pricing(2026))
    metas = registry.all()
    compute_changes(metas, registry.stored_state())
    registry.record_resolution(metas)
    assert any(not m.is_current for m in registry.all())

    registry.upsert(pricing(2025))  # re-ingest rewrites the index too
    assert all(m.is_current for m in registry.all())

    # …and the next reconciliation therefore re-stamps it.
    again = compute_changes(registry.all(), registry.stored_state())
    assert [(m.doc_id, m.is_current) for m in again] == [
        ("sales/Pricing2025.pdf", False)
    ]


def test_delete_removes_the_row(registry):
    registry.upsert(pricing(2026))
    registry.delete("sales/Pricing2026.pdf")
    assert registry.count() == 0


def test_registry_survives_reopening(tmp_path):
    path = tmp_path / "registry.db"
    DocumentRegistry(path).upsert(pricing(2026))
    assert DocumentRegistry(path).count() == 1


# ---------------- resolution ----------------


def test_supersession_is_resolved_across_documents(registry):
    registry.upsert(pricing(2025))
    registry.upsert(pricing(2026))

    changed = compute_changes(registry.all(), registry.stored_state())

    superseded = [m for m in changed if not m.is_current]
    assert [m.doc_id for m in superseded] == ["sales/Pricing2025.pdf"]
    assert superseded[0].superseded_by == "sales/Pricing2026.pdf"


def test_a_steady_state_corpus_produces_no_writes(registry):
    """Reconciliation must be cheap enough to run after every ingest."""
    registry.upsert(pricing(2025))
    registry.upsert(pricing(2026))
    metas = registry.all()
    registry.record_resolution(compute_changes(metas, registry.stored_state()) and metas)

    assert compute_changes(registry.all(), registry.stored_state()) == []


def test_a_document_with_no_successor_stays_current(registry):
    registry.upsert(pricing(2026))
    registry.upsert(
        DocumentMetadata(doc_id="HR/LeavePolicy.pdf", title="LeavePolicy",
                         department="HR", version="4.2")
    )
    changed = compute_changes(registry.all(), registry.stored_state())
    assert all(m.is_current for m in changed)


def test_supersession_does_not_cross_departments(registry):
    """A sales rate card cannot supersede an HR policy that shares a year."""
    registry.upsert(pricing(2026))
    registry.upsert(
        DocumentMetadata(doc_id="HR/Handbook2025.pdf", title="Handbook2025",
                         department="HR", effective_from=date(2025, 1, 1))
    )
    changed = compute_changes(registry.all(), registry.stored_state())
    assert all(m.is_current for m in changed)


# ---------------- the write-back ----------------


class FakeSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, str]] = []

    async def set_version_flags(self, doc_id, *, is_current, superseded_by):
        self.calls.append((doc_id, is_current, superseded_by))
        return 3


async def test_only_changed_documents_are_written(registry):
    registry.upsert(pricing(2025))
    registry.upsert(pricing(2026))
    sink = FakeSink()

    first = await reconcile_versions(registry, sink)
    # Only the 2025 card actually moves: the 2026 card resolves to exactly the
    # status it was stored with, so writing it back would be a no-op write.
    assert first["changed"] == 1
    assert sink.calls == [("sales/Pricing2025.pdf", False, "sales/Pricing2026.pdf")]

    sink.calls.clear()
    second = await reconcile_versions(registry, sink)
    assert second["changed"] == 0
    assert sink.calls == []


async def test_reconciling_an_empty_corpus_is_a_no_op(registry):
    summary = await reconcile_versions(registry, FakeSink())
    assert summary == {"documents": 0, "changed": 0, "chunks_updated": 0}


async def test_summary_names_the_superseded_documents(registry):
    registry.upsert(pricing(2025))
    registry.upsert(pricing(2026))
    summary = await reconcile_versions(registry, FakeSink())
    assert summary["superseded"] == ["sales/Pricing2025.pdf"]
    assert summary["chunks_updated"] == 3  # one document changed, three chunks


# ---------------- live ----------------


@pytest.mark.skipif(not azure_configured(), reason="Azure credentials not configured")
async def test_version_flags_reach_the_live_index(tmp_path, facts):
    """End to end: the 2025 rate card's chunks really do become non-current.

    This is the assertion that was impossible before this pass existed -- and
    the reason `current_only=True` could not exclude the superseded document.
    """
    from rag.targets.azure_search import AzureSearchSink
    from rag.retrieval.searcher import HybridSearcher

    registry = DocumentRegistry(tmp_path / "live.db")
    registry.upsert(pricing(2025))
    registry.upsert(pricing(2026))
    assert facts.pricing_superseded == "sales/Pricing2025.pdf"

    sink = AzureSearchSink()
    try:
        summary = await reconcile_versions(registry, sink)
        assert summary["chunks_updated"] > 0
    finally:
        await sink.aclose()

    searcher = HybridSearcher()
    try:
        hits = await searcher.search(
            f"{facts.top_tier} price per seat", departments=["sales"], top=10,
            doc_ids=[facts.pricing_superseded, facts.pricing_current],
        )
        by_doc = {h.doc_id: h for h in hits}
        assert "sales/Pricing2025.pdf" in by_doc, "superseded doc must stay retrievable"
        assert by_doc["sales/Pricing2025.pdf"].is_current is False
        assert by_doc["sales/Pricing2025.pdf"].superseded_by == "sales/Pricing2026.pdf"
        assert by_doc["sales/Pricing2026.pdf"].is_current is True

        current = await searcher.search(
            f"{facts.top_tier} price per seat",
            departments=["sales"], top=10, current_only=True,
        )
        assert current
        assert all(h.doc_id != "sales/Pricing2025.pdf" for h in current)
    finally:
        await searcher.aclose()
