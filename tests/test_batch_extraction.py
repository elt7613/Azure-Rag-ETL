"""The Azure OpenAI Batch API extraction path: `RelationExtractor`'s bulk sibling.

Same prompt prefix, same strict JSON schema, same validation -- half the price
and a `completion_window="24h"` instead of a network round trip. This module
tests four things, and the split mirrors `test_extractor.py`:

*Structural* tests pin the JSONL shape the Batch API requires and that the
request `body` it carries is byte-for-byte what the online extractor would
have sent -- a divergence here would mean batch-extracted and online-extracted
relations come from two different prompts without anyone deciding that.

*Fake-client* tests prove `submit` -> `poll` -> `apply` without a network,
by recording every call the fake client receives and by handing `apply` a
synthetic download that stands in for the real output file.

*Failure-path* tests are the point of item 4 in the task: a partially-errored
output file, a job that ends `failed`/`expired`, and a truncated
(`finish_reason == "length"`) line must each become a counted, explicit
result -- never a silent drop and never an unhandled exception.

*Live* exercises submission and polling against the real Azure OpenAI
resource, then cancels and cleans up. It does not wait for completion (that
would mean waiting up to 24h), so `apply()` is covered by the fake-client
tests above, not by a live round trip -- see the docstring on the live test
for what was actually observed.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag.config import get_settings
from rag.extraction.batch import (
    BATCH_ENDPOINT,
    COMPLETION_WINDOW,
    MAX_BATCH_LINES,
    BatchCostTracker,
    BatchExtractor,
    BatchLimitExceeded,
    BatchNotReadyError,
)
from rag.extraction.cache import ExtractionCache
from rag.extraction.llm import _TEMPERATURE, build_messages
from rag.extraction.ontology import response_format
from rag.extraction.units import UnitPack, pack_units
from rag.models import ExtractionUnit

from tests.conftest import azure_configured

# Evidence text mirrors test_extractor.py's KEPT_TEXT: money, a duration, a
# deontic verb, proper nouns -- the shape triage would keep in the online path.
# batch.py does not re-run triage (its input is packs the caller already
# triaged), but the same evidence-span guard applies, so the text needs a real
# quotable clause.
KEPT_TEXT = (
    "Employees who have completed twelve months of continuous service are "
    "eligible for tuition reimbursement of up to $5,250 per calendar year. "
    "Reimbursement requires prior written approval from the employee's manager."
)


def _unit(unit_id: str, text: str = KEPT_TEXT, doc_id: str = "HR/Benefits.pdf") -> ExtractionUnit:
    return ExtractionUnit(
        unit_id=unit_id,
        doc_id=doc_id,
        department="HR",
        section_path="6 Tuition Reimbursement",
        text=text,
        page=3,
        chunk_ids=[f"{unit_id}-chunk-a", f"{unit_id}-chunk-b"],
        content_type="prose",
    )


def _one_pack_per_unit(units: list[ExtractionUnit]) -> list[UnitPack]:
    """Each unit its own pack -- a tiny budget forces the split without relying
    on `pack_units`'s token-counting internals matching a hand-picked number."""
    return pack_units(units, budget=1)


def _line(custom_id: str, *, entities=None, relations=None, usage=(1200, 90, 1024),
          finish_reason: str = "stop") -> dict:
    """One well-formed Batch API output line: same envelope OpenAI/Azure emit,
    holding the same per-unit JSON body the online extractor parses."""
    payload = {"units": [{
        "unit_id": "u1",
        "entities": entities or [],
        "relations": relations or [],
    }]}
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": f"req_{custom_id}",
            "body": {
                "id": "chatcmpl-1",
                "choices": [{
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": json.dumps(payload)},
                }],
                "usage": {
                    "prompt_tokens": usage[0],
                    "completion_tokens": usage[1],
                    "total_tokens": usage[0] + usage[1],
                    "prompt_tokens_details": {"cached_tokens": usage[2]},
                },
            },
        },
        "error": None,
    }


def _relation(evidence: str) -> dict:
    return {
        "subject": "Tuition Reimbursement", "subject_type": "Benefit",
        "predicate": "REQUIRES", "object": "manager approval",
        "object_type": "Obligation", "confidence": 0.92,
        "evidence_span": evidence,
    }


def _error_line(custom_id: str, *, code: str = "server_error", message: str = "boom") -> dict:
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": None,
        "error": {"code": code, "message": message},
    }


class FakeBatchClient:
    """Stands in for `AsyncAzureOpenAI`'s `.files` and `.batches` surface.

    Records every upload/create/cancel/delete so composition ("submit uploads
    then creates", "a failed create does not orphan the upload") is provable
    without a network, the same discipline `test_extractor.py`'s `FakeClient`
    applies to chat completions.
    """

    def __init__(self, *, status: str = "completed", output_lines: list[dict] | None = None,
                 error_lines: list[dict] | None = None, create_error: BaseException | None = None,
                 counts: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.uploaded: list[tuple[str, bytes]] = []
        self.deleted: list[str] = []
        self.created: list[dict] = []
        self.cancelled: list[str] = []
        self._status = status
        self._output_lines = output_lines or []
        self._error_lines = error_lines or []
        self._create_error = create_error
        self._counts = counts
        self._contents: dict[str, str] = {}
        self._job: SimpleNamespace | None = None
        self.files = SimpleNamespace(
            create=self._files_create, content=self._files_content, delete=self._files_delete,
        )
        self.batches = SimpleNamespace(
            create=self._batches_create, retrieve=self._batches_retrieve, cancel=self._batches_cancel,
        )

    async def _files_create(self, *, file, purpose):
        assert purpose == "batch"
        name, data = file
        file_id = f"file-{len(self.uploaded) + 1}"
        self.uploaded.append((name, data))
        return SimpleNamespace(id=file_id)

    async def _files_content(self, file_id: str):
        return SimpleNamespace(text=self._contents.get(file_id, ""))

    async def _files_delete(self, file_id: str):
        self.deleted.append(file_id)
        return SimpleNamespace(id=file_id, deleted=True)

    async def _batches_create(self, *, input_file_id, endpoint, completion_window):
        self.created.append({
            "input_file_id": input_file_id, "endpoint": endpoint,
            "completion_window": completion_window,
        })
        if self._create_error is not None:
            raise self._create_error
        output_file_id = "file-out" if self._output_lines else None
        error_file_id = "file-err" if self._error_lines else None
        if output_file_id:
            self._contents[output_file_id] = "\n".join(json.dumps(l) for l in self._output_lines)
        if error_file_id:
            self._contents[error_file_id] = "\n".join(json.dumps(l) for l in self._error_lines)
        self._job = SimpleNamespace(
            id="batch-job-1", status=self._status, input_file_id=input_file_id,
            output_file_id=output_file_id, error_file_id=error_file_id,
            request_counts=SimpleNamespace(
                total=self._counts[0], completed=self._counts[1], failed=self._counts[2],
            ),
        )
        return self._job

    async def _batches_retrieve(self, job_id: str):
        assert self._job is not None and job_id == self._job.id
        return self._job

    async def _batches_cancel(self, job_id: str):
        self.cancelled.append(job_id)
        self._job.status = "cancelling"
        return self._job


def _extractor(client, tmp_path: Path, **kwargs) -> BatchExtractor:
    settings = get_settings().model_copy(update={
        "extraction_cache_path": tmp_path / "extraction_cache.db",
    })
    return BatchExtractor(
        client=client, cache=ExtractionCache(tmp_path / "c.db"), settings=settings, **kwargs,
    )


def _run(coro):
    return asyncio.run(coro)


async def _submit_and_apply(client, tmp_path: Path, units: list[ExtractionUnit], **kwargs):
    extractor = _extractor(client, tmp_path, **kwargs)
    packs = _one_pack_per_unit(units)
    jsonl_path = extractor.build_batch_file(packs, path=tmp_path / "batch.jsonl")
    job_id = await extractor.submit(jsonl_path)
    results = await extractor.apply(job_id)
    return extractor, results


# ==========================================================================
# 1. JSONL shape -- the Batch API's required request envelope
# ==========================================================================


def test_each_line_is_valid_json_with_the_required_keys(tmp_path):
    units = [_unit("u1"), _unit("u2"), _unit("u3")]
    extractor = _extractor(FakeBatchClient(), tmp_path)
    path = extractor.build_batch_file(_one_pack_per_unit(units), path=tmp_path / "b.jsonl")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)  # raises if not valid JSON
        assert set(obj) == {"custom_id", "method", "url", "body"}
        assert obj["method"] == "POST"
        assert obj["url"] == BATCH_ENDPOINT


def test_custom_ids_are_unique(tmp_path):
    units = [_unit(f"u{i}") for i in range(25)]
    extractor = _extractor(FakeBatchClient(), tmp_path)
    path = extractor.build_batch_file(_one_pack_per_unit(units), path=tmp_path / "b.jsonl")

    custom_ids = [json.loads(l)["custom_id"] for l in path.read_text().splitlines()]
    assert len(custom_ids) == len(set(custom_ids)) == 25


def test_body_matches_what_the_online_extractor_would_send(tmp_path):
    """Same model, same messages (system prefix + rendered pack), same strict
    schema, same temperature. A second copy of any of these would let batch
    and online extraction silently diverge."""
    pack = _one_pack_per_unit([_unit("u1")])[0]
    extractor = _extractor(FakeBatchClient(), tmp_path)
    path = extractor.build_batch_file([pack], path=tmp_path / "b.jsonl")

    body = json.loads(path.read_text().splitlines()[0])["body"]
    assert body["model"] == get_settings().azure_openai_chat_deployment
    assert body["messages"] == build_messages(pack)
    assert body["response_format"] == response_format()
    assert body["temperature"] == _TEMPERATURE


def test_a_large_synthetic_input_stays_under_the_documented_line_limit(tmp_path):
    """50,000 requests per file is the Batch API's own documented cap. A
    caller handing batch.py more than that must be told at build time, not
    discover a rejected upload hours later."""
    units = [_unit(f"u{i}", text=KEPT_TEXT) for i in range(5)]
    extractor = _extractor(FakeBatchClient(), tmp_path)
    with pytest.raises(BatchLimitExceeded):
        extractor.build_batch_file(_one_pack_per_unit(units), path=tmp_path / "b.jsonl",
                                    max_lines=3)


def test_a_large_synthetic_input_stays_under_the_documented_byte_limit(tmp_path):
    """200MB is the Batch API's documented file-size ceiling."""
    units = [_unit(f"u{i}", text=KEPT_TEXT * 50) for i in range(5)]
    extractor = _extractor(FakeBatchClient(), tmp_path)
    with pytest.raises(BatchLimitExceeded):
        extractor.build_batch_file(_one_pack_per_unit(units), path=tmp_path / "b.jsonl",
                                    max_bytes=500)


def test_a_realistic_input_fits_comfortably_under_both_limits(tmp_path):
    units = [_unit(f"u{i}", text=KEPT_TEXT) for i in range(200)]
    packs = pack_units(units, budget=3000)
    extractor = _extractor(FakeBatchClient(), tmp_path)
    path = extractor.build_batch_file(packs, path=tmp_path / "b.jsonl")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(packs) < MAX_BATCH_LINES
    assert path.stat().st_size < 200 * 1024 * 1024


# ==========================================================================
# 2. submit -> poll -> apply, composed against a fake client
# ==========================================================================


def test_submit_uploads_then_creates_the_job_with_a_24h_window(tmp_path):
    client = FakeBatchClient(output_lines=[_line("pack-000000")])
    extractor = _extractor(client, tmp_path)
    path = extractor.build_batch_file(_one_pack_per_unit([_unit("u1")]), path=tmp_path / "b.jsonl")

    job_id = _run(extractor.submit(path))

    assert len(client.uploaded) == 1
    assert client.uploaded[0][0] == path.name
    assert client.created == [{
        "input_file_id": "file-1", "endpoint": BATCH_ENDPOINT,
        "completion_window": COMPLETION_WINDOW,
    }]
    assert job_id == "batch-job-1"


def test_a_failed_job_creation_does_not_orphan_the_uploaded_file(tmp_path):
    client = FakeBatchClient(create_error=RuntimeError("quota"))
    extractor = _extractor(client, tmp_path)
    path = extractor.build_batch_file(_one_pack_per_unit([_unit("u1")]), path=tmp_path / "b.jsonl")

    with pytest.raises(RuntimeError):
        _run(extractor.submit(path))

    assert client.uploaded, "the file was uploaded"
    assert client.deleted == ["file-1"], "a failed submission must clean up its own upload"


def test_poll_reports_status_and_counts(tmp_path):
    client = FakeBatchClient(status="in_progress", counts=(3, 1, 0))
    extractor = _extractor(client, tmp_path)
    path = extractor.build_batch_file(_one_pack_per_unit([_unit("u1")]), path=tmp_path / "b.jsonl")
    job_id = _run(extractor.submit(path))

    status = _run(extractor.poll(job_id))
    assert status.job_id == job_id
    assert status.status == "in_progress"
    assert status.total == 3 and status.completed == 1
    assert status.is_terminal is False


def test_apply_before_the_job_is_terminal_raises(tmp_path):
    client = FakeBatchClient(status="validating")
    extractor = _extractor(client, tmp_path)
    path = extractor.build_batch_file(_one_pack_per_unit([_unit("u1")]), path=tmp_path / "b.jsonl")
    job_id = _run(extractor.submit(path))

    with pytest.raises(BatchNotReadyError):
        _run(extractor.apply(job_id))


def test_apply_distributes_relations_back_to_the_right_unit_and_writes_the_cache(tmp_path):
    unit = _unit("u1")
    evidence = "requires prior written approval"
    line = _line("pack-000000", entities=[{"name": "Tuition Reimbursement", "type": "Benefit",
                                            "description": ""}],
                  relations=[_relation(evidence)])
    client = FakeBatchClient(status="completed", output_lines=[line])

    extractor, results = _run(_submit_and_apply(client, tmp_path, [unit]))

    assert [r.unit_id for r in results] == ["u1"]
    relation = results[0].relations[0]
    assert relation.predicate == "REQUIRES"
    assert relation.doc_id == unit.doc_id
    assert relation.source_chunk_id == unit.chunk_ids[0]
    assert relation.department == "HR"
    assert results[0].entities[0].type == "Benefit"

    # Written to the same cache the online path reads, keyed by content hash.
    cached = extractor._cache.get(unit.content_hash(), unit)
    assert cached is not None and cached.relations


def test_apply_bills_at_the_batch_discounted_rate(tmp_path):
    """Batch is a flat 50% discount on input and output tokens (Azure OpenAI
    Batch API, `completion_window="24h"`) -- `BatchCostTracker` must reuse
    `CostTracker`'s own per-token arithmetic and only discount the total, so
    the pricing formula is never duplicated, only halved."""
    from rag.observability.cost import CostTracker

    unit = _unit("u1")
    line = _line("pack-000000", usage=(1200, 90, 1024))
    client = FakeBatchClient(status="completed", output_lines=[line])

    extractor, _ = _run(_submit_and_apply(client, tmp_path, [unit]))

    assert isinstance(extractor.cost, BatchCostTracker)
    assert extractor.cost.prompt_tokens == 1200
    assert extractor.cost.cached_tokens == 1024

    online = CostTracker(get_settings())
    for call in extractor.cost.calls:
        online.record(call.prompt_tokens, call.completion_tokens, call.cached_tokens)
    assert extractor.cost.chat_usd == pytest.approx(online.chat_usd * 0.5)


# ==========================================================================
# 3. Failure handling -- explicit and counted, never silently dropped
# ==========================================================================


def test_a_relation_with_unsupported_evidence_is_dropped_like_the_online_path(tmp_path):
    unit = _unit("u1")
    line = _line("pack-000000", relations=[_relation("this text is not in the source")])
    client = FakeBatchClient(status="completed", output_lines=[line])

    extractor, results = _run(_submit_and_apply(client, tmp_path, [unit]))
    assert results[0].relations == []
    assert extractor.stats.relations_dropped_evidence == 1


def test_partial_results_some_lines_errored_are_isolated_per_pack(tmp_path):
    good = _unit("u1", doc_id="HR/A.pdf")
    bad = _unit("u2", doc_id="HR/B.pdf")
    lines = [
        _line("pack-000000", relations=[_relation("requires prior written approval")]),
        _error_line("pack-000001", code="server_error", message="boom"),
    ]
    client = FakeBatchClient(status="completed", output_lines=lines)

    extractor, results = _run(_submit_and_apply(client, tmp_path, [good, bad]))
    by_id = {r.unit_id: r for r in results}
    assert by_id["u1"].relations, "the healthy line's pack must not be affected by the errored one"
    assert by_id["u2"].skipped_reason.startswith("error")
    assert extractor.stats.packs_failed == 1


def test_a_line_missing_from_the_output_entirely_is_still_a_counted_failure(tmp_path):
    """A pack the output file never mentions (dropped by the service, not
    merely errored) must not vanish -- it must come back as a failed result."""
    unit = _unit("u1")
    client = FakeBatchClient(status="completed", output_lines=[])  # no line at all

    extractor, results = _run(_submit_and_apply(client, tmp_path, [unit]))
    assert results[0].skipped_reason.startswith("error")
    assert extractor.stats.packs_failed == 1


def test_a_truncated_response_is_treated_as_a_failure_not_parsed(tmp_path):
    unit = _unit("u1")
    line = _line("pack-000000", finish_reason="length")
    client = FakeBatchClient(status="completed", output_lines=[line])

    extractor, results = _run(_submit_and_apply(client, tmp_path, [unit]))
    assert results[0].relations == []
    assert results[0].skipped_reason.startswith("error")
    assert "length" in results[0].skipped_reason or "truncat" in results[0].skipped_reason
    assert extractor.stats.packs_failed == 1


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
def test_a_job_that_fails_or_expires_fails_every_unit_explicitly(tmp_path, status):
    units = [_unit("u1"), _unit("u2", doc_id="HR/Other.pdf")]
    client = FakeBatchClient(status=status)

    extractor, results = _run(_submit_and_apply(client, tmp_path, units))
    assert len(results) == 2
    assert all(r.skipped_reason.startswith("error") for r in results)
    assert all(r.relations == [] for r in results)
    assert extractor.stats.packs_failed == 2


def test_a_failed_pack_is_not_written_to_the_cache(tmp_path):
    """Caching a failure would make it permanent and free on the next run --
    the same reasoning `llm.py._failed` documents for the online path."""
    unit = _unit("u1")
    client = FakeBatchClient(status="failed")
    extractor, results = _run(_submit_and_apply(client, tmp_path, [unit]))
    assert extractor._cache.get(unit.content_hash(), unit) is None


# ==========================================================================
# 4. Live: real submission and polling, no wait for completion
# ==========================================================================


live = pytest.mark.skipif(
    not azure_configured("azure_openai_endpoint", "azure_openai_key",
                         "azure_openai_chat_deployment"),
    reason="Azure OpenAI chat deployment not configured",
)


@live
def test_live_submit_and_poll_then_cancel(tmp_path, capsys):
    """Builds 2-3 real packs from `source_data/`, submits them for real, reads
    the job status back for real, then cancels and deletes the uploaded input
    file -- whatever the submission's outcome.

    This deployment (`gpt-4.1-mini` on the `GlobalStandard` SKU) was verified
    live to reject Batch API jobs outright: Azure's own 400 response is
    `invalid_deployment_type`, "not supported for batch jobs ... create a new
    deployment using one of the supported SKUs: [globalbatch,
    datazonebatch]". That is an account/deployment provisioning gap, not a
    defect in this module -- the request reaches Azure, is authenticated, and
    is rejected by a documented, structured error, which is itself proof the
    JSONL/upload/submit path is wired correctly end to end. If a batch-capable
    deployment is ever provisioned, this test also exercises the success path
    (poll reports a real status, then the job is cancelled and the input file
    deleted) without asserting a specific outcome either way.
    """
    from rag.enrich.chunker import chunk_document
    from rag.enrich.metadata import extract_metadata
    from rag.extraction.units import build_units
    from rag.parsing.local_parser import LocalParser

    source_dir = Path(__file__).resolve().parents[1] / "source_data"

    async def build_units_from_corpus() -> list[ExtractionUnit]:
        units: list[ExtractionUnit] = []
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            doc_id = path.relative_to(source_dir).as_posix()
            data = path.read_bytes()
            parsed = await LocalParser().parse(data, doc_id)
            meta = extract_metadata(parsed, doc_id)
            units.extend(build_units(chunk_document(parsed, meta), meta))
            if len(units) >= 3:
                break
        return units[:3]

    units = asyncio.run(build_units_from_corpus())
    assert len(units) >= 2, "source_data/ did not yield enough sections for a small live batch"

    extractor = BatchExtractor(cache=ExtractionCache(tmp_path / "live.db"))
    packs = _one_pack_per_unit(units)  # budget of 1 forces one pack per unit
    assert 2 <= len(packs) <= 3
    jsonl_path = extractor.build_batch_file(packs, path=tmp_path / "live_batch.jsonl")

    job_id: str | None = None
    input_file_id: str | None = None
    try:
        job_id = asyncio.run(extractor.submit(jsonl_path))
    except Exception as exc:  # noqa: BLE001 - the live outcome is asserted below, not assumed
        with capsys.disabled():
            print("\n" + "=" * 72)
            print("LIVE BATCH SUBMISSION was rejected by Azure (see test docstring):")
            print(f"  {type(exc).__name__}: {exc}")
            print("=" * 72)
        body = getattr(getattr(exc, "response", None), "text", "") or str(exc)

        # Two documented environment limits, in the order they were hit while
        # building this. Anything else is a real defect and must fail the test
        # rather than be waved through as "the known Azure thing".
        assert "invalid_deployment_type" not in body, (
            "the deployment is not batch-capable. Batch needs a `globalbatch` "
            "or `datazonebatch` deployment; this one is GlobalStandard. Point "
            "AZURE_OPENAI_BATCH_* at a batch-capable resource."
        )
        assert "token_limit_exceeded" in body, (
            "submission failed for an undocumented reason -- investigate "
            f"before assuming it is an environment limit: {body[:400]}"
        )
        pytest.skip(
            "batch deployment's enqueued-token quota is too small for one "
            "request. The fixed system prefix alone is ~2,263 tokens, so the "
            "minimum request is ~2,297 and the whole corpus needs ~11,700. "
            "Raise the deployment's batch quota in the Azure portal (it is "
            "separate from the online TPM allocation) and this test will "
            "exercise the real submit/poll/cancel round trip."
        )

    try:
        status = asyncio.run(extractor.poll(job_id))
        input_file_id = status.input_file_id
        with capsys.disabled():
            print("\n" + "=" * 72)
            print(f"LIVE BATCH JOB {job_id}: status={status.status} "
                  f"total={status.total} completed={status.completed} failed={status.failed}")
            print("=" * 72)
        assert status.job_id == job_id
        assert status.status in {
            "validating", "in_progress", "finalizing", "completed",
            "failed", "expired", "cancelling", "cancelled",
        }
    finally:
        if job_id is not None:
            try:
                asyncio.run(extractor.cancel(job_id))
            except Exception:
                pass
        if input_file_id is not None:
            try:
                asyncio.run(extractor.delete_input_file(input_file_id))
            except Exception:
                pass
