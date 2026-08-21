"""The content-hash memo store: the control that makes re-ingest free.

Keyed on normalized content rather than on (doc_id, section), so the same
paragraph appearing in 400 documents is extracted once, and a document that
changed in one section does not pay for the other forty.
"""
from __future__ import annotations

import sqlite3
import threading

import pytest

from rag.extraction.cache import ExtractionCache
from rag.models import Entity, ExtractionResult, ExtractionUnit, Relation


def _result(unit_id="u1") -> ExtractionResult:
    return ExtractionResult(
        unit_id=unit_id,
        entities=[Entity(name="Parental Leave", type="Benefit", department="HR",
                         aliases=["parental leave"], description="12 weeks paid")],
        relations=[Relation(subject="Parental Leave", predicate="GRANTS",
                            object="12 weeks", subject_type="Benefit",
                            object_type="Period", doc_id="HR/LeavePolicy.pdf",
                            source_chunk_id="c1", section_path="2.3",
                            page=1, department="HR", confidence=0.9,
                            evidence_span="receive 12 weeks of paid parental leave")],
        prompt_tokens=1200, completion_tokens=180, cached_tokens=1024,
    )


def _cache(tmp_path, name="extraction_cache.db") -> ExtractionCache:
    return ExtractionCache(tmp_path / "nested" / "dirs" / name)


def test_creates_its_parent_directory(tmp_path):
    path = tmp_path / "data" / "deeper" / "extraction_cache.db"
    with ExtractionCache(path):
        pass
    assert path.exists()


def test_miss_then_hit(tmp_path):
    with _cache(tmp_path) as cache:
        assert cache.get("deadbeef") is None
        cache.put("deadbeef", _result())
        hit = cache.get("deadbeef")
        assert hit is not None
        assert hit.entities[0].name == "Parental Leave"
        assert hit.relations[0].predicate == "GRANTS"
        assert hit.relations[0].evidence_span.startswith("receive 12 weeks")


def test_a_hit_is_marked_as_such_so_cost_accounting_stays_honest(tmp_path):
    with _cache(tmp_path) as cache:
        cache.put("h", _result())
        hit = cache.get("h")
        assert hit.from_cache is True
        # The tokens the *original* call cost must not be re-billed.
        assert hit.prompt_tokens == 0
        assert hit.completion_tokens == 0


def test_hit_is_rebound_to_the_asking_units_provenance(tmp_path):
    """Boilerplate shared by 400 documents must cite the document it is read
    in, not the one that happened to be extracted first."""
    asking = ExtractionUnit(
        unit_id="unit-from-doc-400", doc_id="legal/Contract400.pdf",
        department="legal", section_path="4 Confidentiality",
        text="...", page=7, chunk_ids=["chunk-400", "chunk-401"])
    with _cache(tmp_path) as cache:
        cache.put("h", _result(unit_id="unit-from-doc-1"))
        hit = cache.get("h", asking)
        assert hit.unit_id == "unit-from-doc-400"
        assert hit.relations[0].doc_id == "legal/Contract400.pdf"
        assert hit.relations[0].source_chunk_id == "chunk-400"
        assert hit.relations[0].section_path == "4 Confidentiality"
        assert hit.relations[0].page == 7
        assert hit.relations[0].department == "legal"
        assert hit.entities[0].department == "legal"
        # The evidence itself is content, so it survives the rebinding intact.
        assert hit.relations[0].evidence_span.startswith("receive 12 weeks")


def test_get_without_a_unit_returns_the_result_unbound(tmp_path):
    with _cache(tmp_path) as cache:
        cache.put("h", _result())
        hit = cache.get("h")
        assert hit.unit_id == ""
        assert hit.from_cache is True


def test_survives_being_reopened(tmp_path):
    path = tmp_path / "data" / "cache.db"
    with ExtractionCache(path) as cache:
        cache.put("persisted", _result())
    with ExtractionCache(path) as reopened:
        assert reopened.get("persisted") is not None
        assert reopened.stats().entries == 1


def test_reingest_of_an_unchanged_unit_is_a_genuine_no_op(tmp_path):
    unit = ExtractionUnit(unit_id="u", doc_id="HR/LeavePolicy.pdf", department="HR",
                          section_path="2.3 Parental Leave",
                          text="Eligible employees receive 12 weeks of paid parental leave.",
                          page=1, chunk_ids=["c1"])
    # Same content, different document, different whitespace and case.
    resurfaced = ExtractionUnit(
        unit_id="u2", doc_id="finance/Handbook.pdf", department="finance",
        section_path="7 Leave",
        text="  Eligible  employees receive 12 WEEKS of paid parental leave.  ",
        page=9, chunk_ids=["c9"])
    calls = 0

    def extract(u):
        nonlocal calls
        calls += 1
        return _result(u.unit_id)

    with _cache(tmp_path) as cache:
        for u in (unit, unit, resurfaced):
            key = u.content_hash()
            if cache.get(key, u) is None:
                cache.put(key, extract(u))
    assert calls == 1


def test_stats_report_entries_hits_and_misses(tmp_path):
    with _cache(tmp_path) as cache:
        cache.get("a")
        cache.put("a", _result())
        cache.get("a")
        cache.get("a")
        cache.get("b")
        stats = cache.stats()
        assert stats.entries == 1
        assert stats.hits == 2
        assert stats.misses == 2
        assert stats.hit_rate == pytest.approx(0.5)


def test_put_is_idempotent(tmp_path):
    with _cache(tmp_path) as cache:
        cache.put("a", _result())
        cache.put("a", _result())
        assert cache.stats().entries == 1


def test_uses_wal_so_readers_do_not_block_the_writer(tmp_path):
    path = tmp_path / "data" / "cache.db"
    with ExtractionCache(path) as cache:
        cache.put("a", _result())
        mode = sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_concurrent_writers_do_not_corrupt_or_raise(tmp_path):
    """The extractor runs GRAPH_EXTRACT_CONCURRENCY units at once and they all
    write through this one store."""
    path = tmp_path / "data" / "cache.db"
    errors: list[BaseException] = []
    with ExtractionCache(path) as cache:
        def worker(start: int) -> None:
            try:
                for i in range(start, start + 40):
                    cache.put(f"hash-{i}", _result(f"u{i}"))
                    cache.get(f"hash-{i}")
            except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n * 40,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert cache.stats().entries == 320


def test_defaults_to_the_configured_path(tmp_path, monkeypatch):
    from rag.config import Settings, get_settings

    target = tmp_path / "data" / "extraction_cache.db"
    settings = get_settings().model_copy(update={"extraction_cache_path": target})
    monkeypatch.setattr("rag.extraction.cache.get_settings", lambda: settings)
    assert isinstance(settings, Settings)
    with ExtractionCache() as cache:
        cache.put("a", _result())
    assert target.exists()
