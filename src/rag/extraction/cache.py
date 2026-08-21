"""Content-hash memo store for extraction results.

The key is a SHA-256 of the *normalized unit text* -- not the document id, not
the chunk id, not a section path. That choice is the whole point of the module,
and it buys three savings at once:

* a re-ingest of an unchanged corpus costs nothing;
* a document that changed in one section pays for that section only, because
  the other forty hash to what they hashed last time (this composes with
  CocoIndex's document-level memo, which skips unchanged *documents* -- this
  one skips unchanged *sections inside changed documents*);
* the standard confidentiality paragraph that appears in 400 contracts is
  extracted once and served 399 times.

Because the key is content and not location, a cached result carries the
*first* unit's provenance, which is wrong for every subsequent caller. `get()`
therefore rebinds the hit to the asking unit -- its doc_id, chunk id, section
path, page, and department -- before returning it. Skipping that step would
give document 400's confidentiality clause a `source_chunk_id` pointing into
document 1, and an edge whose citation resolves to the wrong document is worse
than no edge: it looks auditable and isn't. Token counters are zeroed for the
same honesty reason -- the cost report should bill the one call that happened,
not the 400 that didn't.

Concurrency: the online extractor runs `GRAPH_EXTRACT_CONCURRENCY` units at a
time, all sharing one store. WAL keeps a reader from blocking the writer,
`busy_timeout` absorbs the writer-vs-writer contention that remains, and a
process-local lock serialises use of the single connection (SQLite connection
objects are not safe to use concurrently even when the database is).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.config import get_settings
from rag.models import Entity, ExtractionResult, ExtractionUnit, Relation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS extractions (
    content_hash TEXT PRIMARY KEY,
    payload      TEXT NOT NULL,
    created_at   REAL NOT NULL
)
"""

# Long enough to ride out a burst of concurrent writers, short enough that a
# genuinely stuck lock surfaces as an error instead of hanging the ingest.
_BUSY_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class CacheStats:
    entries: int
    hits: int
    misses: int
    puts: int

    @property
    def hit_rate(self) -> float:
        looked_up = self.hits + self.misses
        return self.hits / looked_up if looked_up else 0.0


class ExtractionCache:
    """SQLite-backed memo of `ExtractionResult` keyed by content hash."""

    def __init__(self, path: Path | str | None = None,
                 timeout: float = _BUSY_TIMEOUT_SECONDS) -> None:
        self.path = Path(path) if path is not None else get_settings().extraction_cache_path
        # The configured default lives under data/, which is gitignored and may
        # not exist on a fresh clone; failing an 8-hour backfill on a missing
        # directory is not a reasonable failure mode.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path, timeout=timeout, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        # WAL already survives process crashes; NORMAL trades an fsync per
        # commit for throughput on a store that is a cache, not a system of
        # record -- a lost tail costs a re-extraction, not data.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_SCHEMA)
        self._hits = 0
        self._misses = 0
        self._puts = 0

    # ---- lookup ----

    def get(self, content_hash: str,
            unit: ExtractionUnit | None = None) -> ExtractionResult | None:
        """The memoized result for `content_hash`, rebound to `unit`.

        Returns None on a miss. Pass the unit that is asking so the relations
        come back citing *its* document and chunk; omit it only when the
        caller does not intend to write the result to the graph. The result
        carries `from_cache=True` and zeroed token counts.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM extractions WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            if row is None:
                self._misses += 1
                return None
            self._hits += 1
        return rebind(_decode(row[0]), unit)

    def put(self, content_hash: str, result: ExtractionResult) -> None:
        """Memoize `result`. Idempotent -- a repeat write replaces in place."""
        payload = json.dumps(_encode(result), separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                "INSERT INTO extractions (content_hash, payload, created_at) "
                "VALUES (?, ?, ?) ON CONFLICT(content_hash) DO UPDATE SET "
                "payload = excluded.payload, created_at = excluded.created_at",
                (content_hash, payload, time.time()),
            )
            self._puts += 1

    def stats(self) -> CacheStats:
        """Store size plus hit/miss counters since this handle was opened."""
        with self._lock:
            entries = self._conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
            return CacheStats(entries=entries, hits=self._hits,
                              misses=self._misses, puts=self._puts)

    # ---- lifecycle ----

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ExtractionCache":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _encode(result: ExtractionResult) -> dict:
    return {
        "entities": [asdict(e) for e in result.entities],
        "relations": [asdict(r) for r in result.relations],
    }


def _decode(payload: str) -> ExtractionResult:
    data = json.loads(payload)
    return ExtractionResult(
        unit_id="",
        entities=[Entity(**e) for e in data["entities"]],
        relations=[Relation(**r) for r in data["relations"]],
        from_cache=True,
    )


def rebind(result: ExtractionResult, unit: ExtractionUnit | None) -> ExtractionResult:
    """Re-point a cached result at the unit that asked for it.

    Entity `department` moves too: the same clause in an HR document and a
    finance document names two entities that resolve within their own
    department, which is what keeps department-scoped retrieval sound.
    `source_chunk_id` takes the unit's first chunk -- the unit is the whole
    section, so any of its chunks is a valid citation, and the first is the
    one a reader lands on.
    """
    if unit is None:
        return result
    chunk_id = unit.chunk_ids[0] if unit.chunk_ids else ""
    for entity in result.entities:
        entity.department = unit.department
    for relation in result.relations:
        relation.doc_id = unit.doc_id
        relation.source_chunk_id = chunk_id
        relation.section_path = unit.section_path
        relation.page = unit.page
        relation.department = unit.department
    result.unit_id = unit.unit_id
    return result
