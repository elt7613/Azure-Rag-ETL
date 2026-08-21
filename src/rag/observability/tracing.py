"""Tracing, on the assumption that the interesting failures are silent.

A RAG turn touches six or seven components and the ones that go wrong do not
raise. Retrieval returns the wrong chunk; the reranker demotes the right one;
the model uses its own knowledge instead of the evidence. None of that produces
a stack trace, so the only way to debug it later is to have recorded what each
stage saw and decided *at the time*.

Two levels, deliberately:

- **Always on, no dependency.** Every node emits a span with its inputs,
  outputs and cost attributes. With no exporter configured these are no-ops
  costing a few microseconds, and the same information is still returned in the
  response's `diagnostics` — so a developer can debug without provisioning
  anything.
- **Exported when asked.** Setting `APPLICATIONINSIGHTS_CONNECTION_STRING`
  sends them to Application Insights. Nothing else changes.

Attribute naming follows OpenTelemetry's GenAI semantic conventions where they
exist (`gen_ai.*`), because a trace that a standard dashboard can read is worth
more than one with prettier names.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_TRACER = None
_CONFIGURED = False


def _noop_tracer():
    from opentelemetry import trace

    return trace.get_tracer("rag")


def configure_tracing(service_name: str = "rag-retrieval") -> None:
    """Wire up exporting, if this deployment asked for it.

    Called once at service startup. Safe to call when nothing is configured:
    the OpenTelemetry API is a no-op without a provider, so instrumentation
    code does not need to branch on whether tracing is on.
    """
    global _TRACER, _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    connection = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    if not connection:
        _TRACER = _noop_tracer()
        logger.info(
            "tracing: no APPLICATIONINSIGHTS_CONNECTION_STRING; spans are local only"
        )
        return

    try:
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                AzureMonitorTraceExporter(connection_string=connection)
            )
        )
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("rag")
        logger.info("tracing: exporting to Application Insights")
    except ImportError:
        # The exporter is an optional dependency. Missing it must degrade
        # observability, never availability -- a service that will not start
        # because it cannot report on itself is worse than one that is quiet.
        _TRACER = _noop_tracer()
        logger.warning(
            "tracing: azure-monitor-opentelemetry-exporter not installed; "
            "spans are local only. `pip install azure-monitor-opentelemetry-exporter`"
        )
    except Exception:
        _TRACER = _noop_tracer()
        logger.warning("tracing: exporter failed to initialise", exc_info=True)


def get_tracer():
    global _TRACER
    if _TRACER is None:
        _TRACER = _noop_tracer()
    return _TRACER


def _flatten(value: Any) -> Any:
    """Coerce a value into something a span attribute can hold.

    Spans take primitives and homogeneous sequences of them. A list of dicts
    silently drops, so it is rendered as a count plus a joined summary instead
    — losing detail is fine, losing the attribute without saying so is not.
    """
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (str, bool, int, float)) for v in value):
            return list(value)
        return "; ".join(str(v)[:80] for v in value)[:900]
    return str(value)[:900]


@contextmanager
def span(name: str, **attributes: Any):
    """One pipeline stage.

    Attributes set on entry describe the input; set more on the yielded span to
    describe what the stage decided. Recording the decision is the point — a
    span that only says "retrieval took 400ms" cannot answer why the wrong
    chunk came back.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            current.set_attribute(key, _flatten(value))
        try:
            yield current
        except Exception as exc:
            current.record_exception(exc)
            raise


def record_llm_call(current, *, step: str, model: str, usage) -> None:
    """Attach GenAI-convention attributes for one model call."""
    if current is None or usage is None:
        return
    current.set_attribute("gen_ai.system", "az.ai.openai")
    current.set_attribute("gen_ai.request.model", model)
    current.set_attribute("gen_ai.operation.name", step)
    current.set_attribute(
        "gen_ai.usage.input_tokens", int(getattr(usage, "input_tokens", 0) or 0)
    )
    current.set_attribute(
        "gen_ai.usage.output_tokens", int(getattr(usage, "output_tokens", 0) or 0)
    )


def record_retrieval(current, *, query: str, chunk_ids: list[str],
                     scores: list[float]) -> None:
    """Attach what retrieval actually returned.

    The chunk ids are the important part and the part most often omitted:
    without them a trace shows that retrieval happened but not what it found,
    and the turn cannot be replayed.
    """
    if current is None:
        return
    current.set_attribute("rag.query", query[:400])
    current.set_attribute("rag.chunk_count", len(chunk_ids))
    current.set_attribute("rag.chunk_ids", _flatten(chunk_ids[:20]))
    if scores:
        current.set_attribute("rag.top_score", float(max(scores)))
