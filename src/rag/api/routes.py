"""HTTP surface.

Thin on purpose: every route resolves the caller's scope, hands off to the
LangGraph pipeline, and shapes the result. No retrieval logic lives here --
having it in the graph is what makes the same pipeline reachable from the
evaluation harness without going through HTTP.
"""
from __future__ import annotations

import json
import logging
import pathlib
import time
import uuid

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from rag.agents.condense import Turn
from rag.agents.graph_app import run_turn
from rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StatsResponse,
)
from rag.api.security import DEPARTMENT_HEADER, require_scope, resolve_scope

logger = logging.getLogger(__name__)
router = APIRouter()

_STATIC = pathlib.Path(__file__).parent / "static"


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """A minimal chat page.

    The brief says UI is not the focus, and this is not one -- it is a thin
    client over the same `/chat` endpoint the evaluation harness uses. It earns
    its place by showing what the API returns *besides* the answer: the
    citations, the groundedness verdict, whether the system abstained, whether
    a version conflict was resolved, and the full diagnostics. Those are the
    interesting part of this system, and a chat bubble alone hides all of them.
    """
    return FileResponse(_STATIC / "index.html")


def _services(request: Request):
    return request.app.state.services


def _turns(history) -> list[Turn]:
    return [Turn(role=m.role, content=m.content) for m in history]


async def _answer(request: Request, body: ChatRequest, scope: list[str]) -> ChatResponse:
    services = _services(request)
    started = time.perf_counter()
    history = _turns(body.history)
    conversation_id = body.conversation_id or str(uuid.uuid4())

    if body.use_cache and services.settings.query_cache_enabled:
        cached = await services.cache.get(body.message, scope, history)
        if cached is not None:
            return ChatResponse(
                **{k: v for k, v in cached.items() if not k.startswith("_")},
                conversation_id=conversation_id,
                cached=True,
            )

    try:
        state = await run_turn(
            services.pipeline,
            body.message,
            departments=scope,
            history=history,
            conversation_id=conversation_id,
        )
    except Exception:
        services.metrics.errors += 1
        logger.exception("pipeline failed for conversation %s", conversation_id)
        raise HTTPException(
            status_code=500, detail="The assistant failed to process that question."
        ) from None

    latency_ms = (time.perf_counter() - started) * 1000
    clarified = bool(state.get("clarification"))
    services.metrics.record(
        answered=bool(state.get("answered")),
        abstained=bool(state.get("abstained")),
        clarified=clarified,
        latency_ms=latency_ms,
    )

    payload = {
        "answer": state.get("answer", ""),
        "answered": bool(state.get("answered")),
        "abstained": bool(state.get("abstained")),
        "clarification": state.get("clarification", ""),
        "citations": state.get("citations", []),
        "confidence": float(state.get("confidence", 0.0)),
        "diagnostics": state.get("diagnostics", {}),
    }

    if body.use_cache and services.settings.query_cache_enabled:
        await services.cache.put(body.message, scope, payload, history)

    return ChatResponse(**payload, conversation_id=conversation_id, cached=False)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    body: ChatRequest,
    x_user_departments: str | None = Header(default=None, alias=DEPARTMENT_HEADER),
) -> ChatResponse:
    scope = resolve_scope(body.departments, x_user_departments)
    require_scope(scope)
    return await _answer(request, body, scope)


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    x_user_departments: str | None = Header(default=None, alias=DEPARTMENT_HEADER),
) -> StreamingResponse:
    """Server-sent events carrying pipeline progress, then the answer.

    The answer itself is not token-streamed: it is only trustworthy after
    verification has run, and streaming tokens the verifier may reject would
    show the user a claim that is about to be withdrawn. What is streamed is
    *progress* -- which is the part that makes a multi-second answer feel
    responsive without promising something unchecked.
    """
    scope = resolve_scope(body.departments, x_user_departments)
    require_scope(scope)

    async def events():
        for stage in ("understanding the question", "searching the knowledge base",
                      "checking the evidence"):
            yield f"event: progress\ndata: {json.dumps({'stage': stage})}\n\n"
        response = await _answer(request, body, scope)
        yield f"event: answer\ndata: {response.model_dump_json()}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness plus a real probe of each dependency.

    A health check that only reports "the process is up" is the one that stays
    green while every answer fails.
    """
    services = _services(request)
    dependencies: dict[str, str] = {}

    try:
        await services.searcher.search("health probe", departments=["__none__"], top=1)
        dependencies["azure_search"] = "ok"
    except Exception as exc:
        dependencies["azure_search"] = f"error: {type(exc).__name__}"

    try:
        await services.embedder.embed_one("health probe")
        dependencies["azure_openai"] = "ok"
    except Exception as exc:
        dependencies["azure_openai"] = f"error: {type(exc).__name__}"

    if services.graph_retriever is None:
        dependencies["neo4j"] = "disabled"
    else:
        try:
            await services.graph_retriever.ping()
            dependencies["neo4j"] = "ok"
        except Exception as exc:
            dependencies["neo4j"] = f"error: {type(exc).__name__}"

    degraded = any(v.startswith("error") for v in dependencies.values())
    return HealthResponse(
        status="degraded" if degraded else "ok", dependencies=dependencies
    )


@router.get("/stats", response_model=StatsResponse)
async def stats(request: Request) -> StatsResponse:
    services = _services(request)
    return StatsResponse(
        queries=services.metrics.queries,
        answered=services.metrics.answered,
        abstained=services.metrics.abstained,
        clarified=services.metrics.clarified,
        errors=services.metrics.errors,
        latency_ms=services.metrics.percentiles(),
        cache=services.cache.stats.as_dict(),
        departments=services.registry.names(),
    )


@router.get("/departments")
async def departments(request: Request) -> dict:
    """The configured department list.

    Exposed because it is the observable proof that departments are
    configuration: change DEPARTMENTS in the environment, restart, and this
    changes with it -- along with what is monitored, ingested and retrievable.
    """
    registry = _services(request).registry
    return {
        "departments": [
            {"name": d.name, "source_prefix": d.source_prefix}
            for d in registry.all_departments()
        ]
    }
