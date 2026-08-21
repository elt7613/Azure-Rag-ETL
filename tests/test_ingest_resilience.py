"""One poisoned document must not stop the run.

At the eleven-document scale of the sample corpus a crash on a bad file is an
inconvenience; at the five-million-document scale the system is designed for
it is the difference between an ingest that finishes and one that never does.
These tests drive the real `process_document` -- decorator and all -- with its
collaborators substituted, so what is under test is the actual error handling
in the ETL entry point rather than a re-implementation of it.
"""
from importlib import import_module
from pathlib import PurePosixPath

import pytest


class _StubFilePath:
    def __init__(self, path: str) -> None:
        self.path = PurePosixPath(path)


class _StubFile:
    """The narrow slice of `FileLike` that `process_document` actually uses:
    a source-relative path and an awaitable read."""

    def __init__(self, path: str, data: bytes) -> None:
        self.file_path = _StubFilePath(path)
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _StubSink:
    def __init__(self) -> None:
        self.upserted: list[str] = []
        self.deleted: list[str] = []

    async def delete_document(self, doc_id: str) -> int:
        self.deleted.append(doc_id)
        return 0

    async def upsert(self, chunks, vectors, meta) -> int:
        self.upserted.append(meta.doc_id)
        return len(chunks)


class _StubEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


@pytest.fixture
def pipeline(monkeypatch):
    """`process_document` reads its collaborators out of the CocoIndex
    environment, which only exists inside a running app. Substituting the one
    resolver keeps the function itself -- including the failure handling being
    tested -- completely untouched.

    `import_module` rather than `from rag.etl import app`: the package
    re-exports the CocoIndex `App` object under that same name, so the plain
    import hands back the app instead of the module."""
    app = import_module("rag.etl.app")

    sink, embedder = _StubSink(), _StubEmbedder()
    monkeypatch.setattr(app, "_collaborators", lambda: (sink, embedder, None))
    app.INGEST_STATS.reset()
    yield sink
    app.INGEST_STATS.reset()


_GOOD = "source_data/HR/LeavePolicy.pdf"
_ALSO_GOOD = "source_data/sales/Pricing2026.pdf"
# A .docx is routed to the local parser under every DOC_PARSER setting, so
# this stays a genuinely local, deterministic failure rather than an
# accidental live call to Document Intelligence.
_CORRUPT = b"this is not an office open xml package"


async def test_corrupt_document_is_recorded_and_does_not_raise(pipeline):
    from rag.etl.app import INGEST_STATS, process_document

    written = await process_document(_StubFile("HR/Corrupt.docx", _CORRUPT))

    assert written == 0
    assert INGEST_STATS.documents_failed == 1
    error = INGEST_STATS.errors[0]
    assert error.doc_id == "HR/Corrupt.docx"
    assert error.stage == "parse"
    assert error.error_type == "BadZipFile"
    assert error.message


async def test_a_bad_document_does_not_stop_the_ones_around_it(pipeline):
    from rag.etl.app import INGEST_STATS, process_document

    sink = pipeline
    files = [
        _StubFile("HR/LeavePolicy.pdf", open(_GOOD, "rb").read()),
        _StubFile("HR/Corrupt.docx", _CORRUPT),
        _StubFile("sales/Pricing2026.pdf", open(_ALSO_GOOD, "rb").read()),
    ]
    written = [await process_document(f) for f in files]

    assert written[0] > 0
    assert written[1] == 0
    assert written[2] > 0
    assert sink.upserted == ["HR/LeavePolicy.pdf", "sales/Pricing2026.pdf"]
    assert INGEST_STATS.documents_succeeded == 2
    assert INGEST_STATS.documents_failed == 1
    assert INGEST_STATS.chunks_written == written[0] + written[2]


async def test_failure_is_logged_with_doc_id_and_stage(pipeline, caplog):
    from rag.etl.app import process_document

    with caplog.at_level("ERROR", logger="rag.etl.app"):
        await process_document(_StubFile("HR/Corrupt.docx", _CORRUPT))

    assert any(
        record.levelname == "ERROR"
        and "HR/Corrupt.docx" in record.getMessage()
        and "parse" in record.getMessage()
        for record in caplog.records
    )


async def test_stats_are_inspectable_and_resettable(pipeline):
    from rag.etl.app import INGEST_STATS, process_document

    await process_document(_StubFile("HR/Corrupt.docx", _CORRUPT))
    snapshot = INGEST_STATS.as_dict()

    assert snapshot["documents_failed"] == 1
    assert snapshot["errors"][0]["doc_id"] == "HR/Corrupt.docx"

    INGEST_STATS.reset()
    assert INGEST_STATS.documents_failed == 0
    assert INGEST_STATS.errors == []


async def test_retained_errors_are_capped_but_the_count_is_not(pipeline):
    """At scale the failure *list* has to be bounded or a bad batch becomes a
    memory leak; the failure *count* must stay exact so the ingest report and
    /stats never understate the damage."""
    from rag.etl.app import _MAX_RETAINED_ERRORS, INGEST_STATS
    from rag.models import IngestError

    for i in range(_MAX_RETAINED_ERRORS + 5):
        INGEST_STATS.record_failure(
            IngestError(doc_id=f"HR/{i}.docx", stage="parse",
                        error_type="BadZipFile", message="boom")
        )

    assert INGEST_STATS.documents_failed == _MAX_RETAINED_ERRORS + 5
    assert len(INGEST_STATS.errors) == _MAX_RETAINED_ERRORS
