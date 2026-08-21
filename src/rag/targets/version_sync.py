"""Resolving document supersession across the whole corpus.

A gap this closes: `extract_metadata` parses a document's own
`Supersedes: 2025 Rate Card (v1.4)` header, but the *consequence* of that
header lands on a **different** document -- Pricing2025 becomes `is_current
= false` with `superseded_by = sales/Pricing2026.pdf`. CocoIndex processes
each document independently with no barrier between them, so the per-document
path has one `DocumentMetadata` in hand and nothing to link against. The result
was that every chunk in the search index carried `is_current = true`,
including the superseded 2025 rate card: the vector store had no version signal
at all, and "what is the Enterprise price?" was a coin flip between $99 and
$109.

The fix is a whole-corpus reconciliation pass, and the interesting question is
where it gets its inputs. Three options, and why this one:

- *Re-parse every document.* Correct and unaffordable -- at five million
  documents it re-does the entire ingest to answer a question about headers.
- *Read them back from Neo4j*, where `supersedes_hint` is already stored.
  Cheap, but it makes versioning in the vector store depend on the graph being
  enabled, which is an invisible coupling between two independent targets.
- *Keep a registry.* A small SQLite sidecar recording one row per document --
  the handful of metadata fields supersession needs. Written during ingest,
  when the metadata is already in hand and costs nothing extra; read in full
  during reconciliation, because it is tiny even for millions of documents
  (a few hundred bytes each). This is what is implemented here.

The reconciliation only writes back where something actually changed, so a
steady-state run touches nothing.

One operational note for anyone verifying this by hand: Azure AI Search is
eventually consistent, so a merge is not immediately visible to a query. A
check run in the same second as the ingest will still see the old flags and
look like a failure. It is not.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from rag.enrich.metadata import link_versions
from rag.models import DocumentMetadata

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT '',
    department      TEXT NOT NULL DEFAULT '',
    version         TEXT NOT NULL DEFAULT '',
    owner           TEXT NOT NULL DEFAULT '',
    effective_from  TEXT,
    effective_to    TEXT,
    supersedes      TEXT NOT NULL DEFAULT '',
    superseded_by   TEXT NOT NULL DEFAULT '',
    is_current      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS documents_department ON documents(department);
"""


def _as_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


class DocumentRegistry:
    """One row per ingested document: what supersession needs, nothing more.

    SQLite rather than the search index itself because the index has no
    `supersedes` field and adding one would mean recreating the index -- and
    because reconciliation wants a cheap full scan of metadata, which is a
    different access pattern from what a search index is for.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30)
        # WAL so the ingest's concurrent writers do not serialise behind each
        # other or trip over a reader mid-reconciliation.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def upsert(self, meta: DocumentMetadata) -> None:
        """Record a document's metadata as parsed, before cross-document resolution.

        Re-ingesting a document **resets** its resolved `is_current` /
        `superseded_by` here, which looks wrong and is not: the same ingest
        that calls this also rewrites the document's chunks in the search
        index with those fields at their per-document defaults. The registry's
        job is to mirror what the index actually holds, so that the
        reconciliation pass -- which only writes where the two differ -- can
        see that the document needs re-stamping.

        Keeping the resolved value here instead is the bug this comment
        exists to prevent, and it is silent: reconciliation compares against
        its own stale record, concludes nothing changed, writes nothing, and
        the index is left claiming a superseded rate card is current.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (doc_id, title, department, version, owner,
                     effective_from, effective_to, supersedes,
                     superseded_by, is_current)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title,
                    department=excluded.department,
                    version=excluded.version,
                    owner=excluded.owner,
                    effective_from=excluded.effective_from,
                    effective_to=excluded.effective_to,
                    supersedes=excluded.supersedes,
                    is_current=excluded.is_current,
                    superseded_by=excluded.superseded_by
                """,
                (
                    meta.doc_id, meta.title, meta.department, meta.version,
                    meta.owner,
                    meta.effective_from.isoformat() if meta.effective_from else None,
                    meta.effective_to.isoformat() if meta.effective_to else None,
                    meta.supersedes, meta.superseded_by, int(meta.is_current),
                ),
            )
            conn.commit()

    def delete(self, doc_id: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()

    def all(self) -> list[DocumentMetadata]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY doc_id").fetchall()
        return [
            DocumentMetadata(
                doc_id=row["doc_id"],
                title=row["title"],
                department=row["department"],
                version=row["version"],
                owner=row["owner"],
                effective_from=_as_date(row["effective_from"]),
                effective_to=_as_date(row["effective_to"]),
                supersedes=row["supersedes"],
                superseded_by=row["superseded_by"],
                is_current=bool(row["is_current"]),
            )
            for row in rows
        ]

    def stored_state(self) -> dict[str, tuple[bool, str]]:
        """`doc_id -> (is_current, superseded_by)` as last reconciled."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT doc_id, is_current, superseded_by FROM documents"
            ).fetchall()
        return {r["doc_id"]: (bool(r["is_current"]), r["superseded_by"]) for r in rows}

    def record_resolution(self, metas: list[DocumentMetadata]) -> None:
        with closing(self._connect()) as conn:
            conn.executemany(
                "UPDATE documents SET is_current = ?, superseded_by = ? "
                "WHERE doc_id = ?",
                [(int(m.is_current), m.superseded_by, m.doc_id) for m in metas],
            )
            conn.commit()

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


def compute_changes(
    metas: list[DocumentMetadata], previous: dict[str, tuple[bool, str]]
) -> list[DocumentMetadata]:
    """Documents whose supersession status differs from what is stored.

    Returning only the delta is what keeps this pass cheap: a steady-state
    corpus produces an empty list and no writes at all, so reconciliation can
    run after every ingest without being a cost.
    """
    resolved = link_versions(metas)
    changed: list[DocumentMetadata] = []
    for meta in resolved:
        before = previous.get(meta.doc_id)
        after = (meta.is_current, meta.superseded_by)
        if before is None or before != after:
            changed.append(meta)
    return changed


async def reconcile_versions(registry: DocumentRegistry, sink) -> dict:
    """Resolve supersession across the corpus and push the result to the index.

    `sink` is an `AzureSearchSink`. Returns a summary so the ingest report can
    say what moved, rather than the caller having to infer it from logs.
    """
    metas = registry.all()
    if not metas:
        return {"documents": 0, "changed": 0, "chunks_updated": 0}

    previous = registry.stored_state()
    changed = compute_changes(metas, previous)

    chunks_updated = 0
    for meta in changed:
        chunks_updated += await sink.set_version_flags(
            meta.doc_id,
            is_current=meta.is_current,
            superseded_by=meta.superseded_by,
        )

    if changed:
        registry.record_resolution(metas)
        logger.info(
            "version reconciliation updated %d document(s), %d chunk(s): %s",
            len(changed), chunks_updated,
            ", ".join(
                f"{m.doc_id}{'' if m.is_current else ' -> superseded by ' + m.superseded_by}"
                for m in changed
            ),
        )

    return {
        "documents": len(metas),
        "changed": len(changed),
        "chunks_updated": chunks_updated,
        "superseded": [m.doc_id for m in metas if not m.is_current],
    }
