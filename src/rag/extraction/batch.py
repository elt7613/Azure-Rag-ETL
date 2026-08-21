"""The bulk sibling of `llm.RelationExtractor`: the Azure OpenAI Batch API path.

Verified against Microsoft Learn: the Batch API gives a flat 50% discount on
input and output tokens, requires `completion_window="24h"`, and draws on a
separate enqueued-token quota from online traffic. Output tokens are 96% of
the measured online bill, so batch is the single largest remaining lever for
backfilling millions of documents. `online` (`llm.py`) stays for the
incremental deltas live monitoring produces, where a day of latency is not
acceptable; `batch` (here) is for the initial load, where it costs nothing.
`GRAPH_EXTRACT_MODE` selects between them.

**Why this does not re-implement the prompt.** The whole cost argument for
batch collapses if it silently drifts from what online sends: a relation
extracted in bulk has to be validated exactly as a relation extracted inline,
or "the graph" is actually two graphs with different quality bars nobody
signed off on. So the request body is built from `llm.py`'s own
`build_messages`, `response_format`, and temperature -- imported, not
retyped -- and every response this module parses is handed to
`RelationExtractor._distribute` / `_entities` / `_relations`, the same code
that checks the evidence span and coerces the ontology online. This module
owns exactly one thing `llm.py` does not: the JSONL envelope and the
submit/poll/download lifecycle around a request that will not have an answer
for up to a day.

**The four-step shape.** `build_batch_file(packs)` turns already-packed units
(the caller has already run triage, checked the cache, and called
`units.pack_units` -- this module does not repeat that) into the JSONL the
Batch API requires, one line per pack, `custom_id` recoverable back to that
pack's units. `submit(jsonl_path)` uploads it and creates the job.
`poll(job_id)` reports where it is. `apply(job_id)` downloads whatever
finished, validates and caches every relation it produced, and returns
`ExtractionResult`s in the same shape `RelationExtractor.extract` returns --
including for the units a bad line, a truncated response, or a dead job never
answered for, because those outcomes must be counted, not disappear.

**Why a manifest, not just a job id.** A batch job answers up to a day after
it is submitted, possibly from a different process. `custom_id` alone cannot
carry a unit's full text, department, and provenance (Azure caps its length),
so `build_batch_file` also writes a manifest sidecar mapping each `custom_id`
to its units; `submit` re-keys that sidecar under the job id it receives so
`apply(job_id)` can reconstruct the original packs from the id alone, exactly
as the task's signature promises.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai import AsyncAzureOpenAI

from rag.config import Settings, get_settings
from rag.extraction.cache import ExtractionCache
from rag.extraction.llm import _TEMPERATURE, ExtractionStats, RelationExtractor, build_messages
from rag.extraction.ontology import response_format
from rag.extraction.units import UnitPack
from rag.models import ExtractionResult, ExtractionUnit
from rag.observability.cost import CostTracker

logger = logging.getLogger(__name__)

# Azure's documented REST path for a chat-completions batch line and for the
# `endpoint` argument to `batches.create` -- verified live against this
# deployment: both this value and the OpenAI-public `/v1/chat/completions`
# form are accepted structurally, but `/chat/completions` (no `/v1`) is what
# the Azure SDK itself recognises as a deployments-routed path (see
# `openai.lib.azure._deployments_endpoints`), so it is the one used here.
BATCH_ENDPOINT = "/chat/completions"

# The only value Azure's Batch API accepts today; a module constant rather
# than a setting because it is not a tuning knob, it is what the product
# supports.
COMPLETION_WINDOW = "24h"

# Azure OpenAI Batch API: flat 50% off both input and output tokens, applied
# uniformly to whatever `CostTracker`'s own per-token arithmetic computes at
# the online rate -- see `BatchCostTracker` below.
BATCH_DISCOUNT = 0.5

# Documented Batch API ceilings (OpenAI/Azure). Enforced at build time so an
# oversized submission is a clear error here, not a rejected upload found
# hours into what was meant to be a 24h wait.
MAX_BATCH_LINES = 50_000
MAX_BATCH_FILE_BYTES = 200 * 1024 * 1024


class BatchLimitExceeded(ValueError):
    """A JSONL build would exceed the Batch API's line-count or byte-size cap."""


class BatchNotReadyError(RuntimeError):
    """`apply()` was called before `poll()` reports a terminal status."""


class BatchCostTracker(CostTracker):
    """`CostTracker` billed at the Batch API's flat 50% discount.

    Deliberately a thin subclass rather than a second pricing formula: every
    call still goes through `CostTracker.record`/`record_usage`, so the
    per-token math (cached vs. uncached input, the clamp, the embedding rate)
    is defined in exactly one place. Only the aggregate `chat_usd` -- and
    therefore `total_usd` -- is halved.
    """

    @property
    def chat_usd(self) -> float:
        return super().chat_usd * BATCH_DISCOUNT


@dataclass(frozen=True)
class BatchStatus:
    """What `poll()` knows about a job, flattened out of the SDK's `Batch`."""
    job_id: str
    status: str
    total: int
    completed: int
    failed: int
    input_file_id: str | None
    output_file_id: str | None
    error_file_id: str | None

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "failed", "expired", "cancelled"}

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


def _namespace(value: Any) -> Any:
    """Recursively turn a plain JSON dict/list into attribute-accessible
    `SimpleNamespace`s.

    A downloaded batch output line is plain JSON; `RelationExtractor._distribute`
    and the `_token_shares` helper it calls read `.choices[0].message.content`
    and `.usage.prompt_tokens_details.cached_tokens` via `getattr`, exactly as
    they would on the SDK's own response object from the online path. This is
    the one piece of glue that lets the same validation code run over both
    transports unmodified.
    """
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(v) for v in value]
    return value


def _manifest_sidecar(jsonl_path: Path) -> Path:
    return Path(str(jsonl_path) + ".manifest.json")


def _unit_from_dict(data: dict) -> ExtractionUnit:
    return ExtractionUnit(**data)


def _line_failure_message(error_block: dict | None, status_code: Any) -> str:
    if error_block:
        code = error_block.get("code", "?")
        message = error_block.get("message", "")
        return f"error: batch line {code}: {message}"[:300]
    return f"error: batch line returned status {status_code!r}"


class BatchExtractor:
    """Submits packed units to the Azure OpenAI Batch API and applies the
    result with the same validation `RelationExtractor` uses online.

    Every collaborator is injectable, matching `RelationExtractor`'s pattern:
    a fake client proves `submit`/`poll`/`apply` compose correctly without a
    network, and a real one is built lazily so importing this module needs no
    credentials.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: ExtractionCache | None = None,
        client: Any | None = None,
        cost: BatchCostTracker | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._cache = cache if cache is not None else ExtractionCache()
        self._client = client
        self.cost = cost if cost is not None else BatchCostTracker(self._settings)
        self.stats = ExtractionStats()
        # `_distribute`/`_failed` hold the validation this module must not
        # duplicate. Sharing `stats` and `_cache` (rather than letting the
        # validator keep its own) means their counters and cache writes are
        # this extractor's counters and cache writes, not a shadow copy of
        # them.
        self._validator = RelationExtractor(settings=self._settings, cache=self._cache)
        self._validator.stats = self.stats

    @property
    def client(self) -> Any:
        if self._client is None:
            # Deliberately the batch-specific values, not the online ones.
            # They fall back to the online settings when unset, so a single
            # batch-capable deployment needs no extra configuration -- but when
            # batch lives on its own resource, as it usually must, this is what
            # keeps the online path pointed at the low-latency deployment.
            self._client = AsyncAzureOpenAI(
                azure_endpoint=self._settings.batch_endpoint,
                api_key=self._settings.batch_key,
                api_version=self._settings.batch_api_version,
            )
        return self._client

    def _manifest_store_path(self, job_id: str) -> Path:
        return self._settings.extraction_cache_path.parent / "batch_jobs" / f"{job_id}.json"

    # ---- 1. build ----

    def build_batch_file(
        self,
        packs: list[UnitPack],
        path: Path | None = None,
        *,
        max_lines: int = MAX_BATCH_LINES,
        max_bytes: int = MAX_BATCH_FILE_BYTES,
    ) -> Path:
        """One line per pack, `body` byte-identical to what `llm.py` would send.

        Raises `BatchLimitExceeded` rather than writing a file the Batch API
        would reject anyway -- at 5M-document scale a rejected 200MB upload
        discovered after the fact is a wasted day, not a wasted second.
        """
        if path is None:
            batch_dir = self._settings.extraction_cache_path.parent / "batch_jobs"
            batch_dir.mkdir(parents=True, exist_ok=True)
            path = batch_dir / f"batch-{uuid.uuid4().hex[:12]}.jsonl"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        encoded_lines: list[str] = []
        manifest: dict[str, list[dict]] = {}
        total_bytes = 0
        for index, pack in enumerate(packs):
            custom_id = f"pack-{index:06d}"
            body = {
                "model": self._settings.batch_deployment,
                "messages": build_messages(pack),
                "response_format": response_format(),
                "temperature": _TEMPERATURE,
            }
            line = {"custom_id": custom_id, "method": "POST", "url": BATCH_ENDPOINT, "body": body}
            encoded = json.dumps(line, separators=(",", ":")) + "\n"
            total_bytes += len(encoded.encode("utf-8"))
            encoded_lines.append(encoded)
            manifest[custom_id] = [asdict(unit) for unit in pack.units]

        if len(encoded_lines) > max_lines:
            raise BatchLimitExceeded(
                f"{len(encoded_lines)} requests exceeds the Batch API's {max_lines}-line limit"
            )
        if total_bytes > max_bytes:
            raise BatchLimitExceeded(
                f"{total_bytes} bytes exceeds the Batch API's {max_bytes}-byte limit"
            )

        path.write_text("".join(encoded_lines), encoding="utf-8")
        _manifest_sidecar(path).write_text(
            json.dumps({"packs": manifest}), encoding="utf-8"
        )
        return path

    # ---- 2. submit ----

    async def submit(self, jsonl_path: Path) -> str:
        """Upload the file, create the job, and re-key its manifest under the
        job id so `apply(job_id)` can find it from the id alone.

        A batch job that fails to create must not leave its upload behind --
        that upload counts against the same file-storage quota a real
        backfill needs, so a failure here deletes what it just uploaded
        before re-raising.
        """
        jsonl_path = Path(jsonl_path)
        sidecar = _manifest_sidecar(jsonl_path)
        if not sidecar.exists():
            raise ValueError(
                f"{jsonl_path} has no manifest sidecar; build it with build_batch_file()"
            )

        uploaded = await self.client.files.create(
            file=(jsonl_path.name, jsonl_path.read_bytes()), purpose="batch"
        )
        try:
            job = await self.client.batches.create(
                input_file_id=uploaded.id,
                endpoint=BATCH_ENDPOINT,
                completion_window=COMPLETION_WINDOW,
            )
        except Exception:
            await self.client.files.delete(uploaded.id)
            raise

        manifest_data = json.loads(sidecar.read_text(encoding="utf-8"))
        manifest_data["input_file_id"] = uploaded.id
        store_path = self._manifest_store_path(job.id)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(json.dumps(manifest_data), encoding="utf-8")
        return job.id

    # ---- 3. poll ----

    async def poll(self, job_id: str) -> BatchStatus:
        batch = await self.client.batches.retrieve(job_id)
        counts = getattr(batch, "request_counts", None)
        return BatchStatus(
            job_id=batch.id,
            status=batch.status,
            total=getattr(counts, "total", 0) if counts else 0,
            completed=getattr(counts, "completed", 0) if counts else 0,
            failed=getattr(counts, "failed", 0) if counts else 0,
            input_file_id=getattr(batch, "input_file_id", None),
            output_file_id=getattr(batch, "output_file_id", None),
            error_file_id=getattr(batch, "error_file_id", None),
        )

    # ---- lifecycle helpers (used by the live test to leave nothing running) ----

    async def cancel(self, job_id: str) -> Any:
        return await self.client.batches.cancel(job_id)

    async def delete_input_file(self, file_id: str) -> None:
        await self.client.files.delete(file_id)

    # ---- 4. apply ----

    def _load_manifest(self, job_id: str) -> dict[str, list[dict]]:
        store_path = self._manifest_store_path(job_id)
        if not store_path.exists():
            raise ValueError(
                f"no manifest for batch job {job_id!r}; apply() must run against the "
                "cache directory that called submit() for this job"
            )
        data = json.loads(store_path.read_text(encoding="utf-8"))
        return data["packs"]

    async def _download_lines(self, file_id: str) -> list[tuple[str | None, dict]]:
        content = await self.client.files.content(file_id)
        lines: list[tuple[str | None, dict]] = []
        for raw in content.text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            lines.append((obj.get("custom_id"), obj))
        return lines

    def _apply_line(
        self, custom_id: str, line: dict, units_by_pack: dict[str, list[ExtractionUnit]]
    ) -> dict[str, ExtractionResult]:
        """One output-file line -> one pack's worth of `ExtractionResult`s.

        Mirrors `RelationExtractor._run_pack`'s post-response handling: a
        request-level error, a non-200 response, and a truncated
        (`finish_reason == "length"`) completion are each caught and named
        *before* attempting to parse, exactly the explicit handling item 4 of
        the task requires. Everything that survives those checks is hand
        to `_distribute`, the same code the online path uses.
        """
        units = units_by_pack.get(custom_id)
        if units is None:
            self.stats.unattributed_units += 1
            return {}
        pack = UnitPack(units=units)

        error_block = line.get("error")
        response_block = line.get("response") or {}
        status_code = response_block.get("status_code")
        if error_block or status_code != 200:
            return self._validator._failed(
                pack, RuntimeError(_line_failure_message(error_block, status_code))
            )

        body = _namespace(response_block.get("body") or {})
        choices = getattr(body, "choices", None) or []
        choice = choices[0] if choices else None
        if choice is None:
            return self._validator._failed(pack, RuntimeError("batch response had no choices"))
        if getattr(choice, "finish_reason", None) == "length":
            return self._validator._failed(
                pack, RuntimeError("response truncated (finish_reason=length)")
            )

        try:
            payload = json.loads(choice.message.content)
        except (AttributeError, TypeError, ValueError) as exc:
            return self._validator._failed(pack, exc)

        self.stats.llm_calls += 1
        self.cost.record_usage(getattr(body, "usage", None))
        return self._validator._distribute(pack, payload, body)

    def _fail_all(
        self, units_by_pack: dict[str, list[ExtractionUnit]], reason: str
    ) -> dict[str, ExtractionResult]:
        results: dict[str, ExtractionResult] = {}
        for units in units_by_pack.values():
            results.update(self._validator._failed(UnitPack(units=units), RuntimeError(reason)))
        return results

    async def apply(self, job_id: str) -> list[ExtractionResult]:
        """Download whatever the job produced, validate and cache it exactly
        as the online path would, and return one result per unit that was
        submitted -- including for units the job never answered.
        """
        status = await self.poll(job_id)
        manifest = self._load_manifest(job_id)
        units_by_pack = {
            custom_id: [_unit_from_dict(d) for d in units] for custom_id, units in manifest.items()
        }
        self.stats.units_in += sum(len(units) for units in units_by_pack.values())

        if not status.is_terminal:
            raise BatchNotReadyError(
                f"batch job {job_id} is still {status.status!r}; call poll() again before apply()"
            )

        if status.succeeded:
            results: dict[str, ExtractionResult] = {}
            if status.output_file_id:
                for custom_id, line in await self._download_lines(status.output_file_id):
                    if custom_id is not None:
                        results.update(self._apply_line(custom_id, line, units_by_pack))
            if status.error_file_id:
                for custom_id, line in await self._download_lines(status.error_file_id):
                    already = custom_id is not None and any(
                        u.unit_id in results for u in units_by_pack.get(custom_id, [])
                    )
                    if custom_id is not None and not already:
                        results.update(self._apply_line(custom_id, line, units_by_pack))
            # A pack the output/error files never mention at all (dropped by
            # the service rather than errored) is still a failure, and still
            # counted -- the case item 4 calls "not silently dropped".
            for custom_id, units in units_by_pack.items():
                if not any(u.unit_id in results for u in units):
                    results.update(self._validator._failed(
                        UnitPack(units=units),
                        RuntimeError("no response for this request in the batch output"),
                    ))
        else:
            logger.warning("batch job %s ended in status %r; failing its %d packs",
                            job_id, status.status, len(units_by_pack))
            results = self._fail_all(units_by_pack, f"batch job ended in status {status.status!r}")

        ordered_units = [unit for units in units_by_pack.values() for unit in units]
        return [results[unit.unit_id] for unit in ordered_units]
