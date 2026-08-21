"""The retrieval service.

Everything expensive to construct -- the Azure AI Search client, the embedder,
the Neo4j driver, the compiled LangGraph -- is built once in the lifespan and
shared. Building them per request would add a TLS handshake and a graph
compilation to every question, and the aiohttp transport underneath the Azure
clients cannot be reopened once closed, so per-request construction would fail
outright after the first call.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.agents.graph_app import RetrievalContext, build_graph
from rag.config import get_settings
from rag.departments import get_registry
from rag.embedding.azure_openai import AzureOpenAIEmbedder
from rag.observability.tracing import configure_tracing
from rag.retrieval.cache import QueryCache
from rag.retrieval.searcher import HybridSearcher

logger = logging.getLogger(__name__)


class Services:
    """Handles to everything the routes need, with one owner and one shutdown."""

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.embedder = AzureOpenAIEmbedder()
        self.searcher = HybridSearcher(embedder=self.embedder)
        self.graph_retriever = None
        self.cache = QueryCache(
            threshold=settings.semantic_cache_threshold,
            ttl_seconds=settings.query_cache_ttl_seconds,
            max_entries=settings.query_cache_max_entries,
            embedder=self.embedder if settings.query_cache_enabled else None,
        )
        self.registry = get_registry()
        self.pipeline = None
        self.metrics = ServiceMetrics()

    async def start(self) -> None:
        if self.settings.graph_enabled:
            try:
                from rag.retrieval.graph_retriever import GraphRetriever

                self.graph_retriever = GraphRetriever(self.searcher)
            except ImportError:
                # The graph retriever lands with the extracted-entity layer.
                # Until then the service runs vector-only rather than refusing
                # to start -- a missing retriever should cost recall, not uptime.
                logger.info("graph retriever unavailable; running vector-only")
            except Exception:
                logger.warning("graph retriever failed to initialise", exc_info=True)

        self.pipeline = build_graph(
            RetrievalContext(self.searcher, self.graph_retriever)
        )

    async def stop(self) -> None:
        await self.searcher.aclose()
        if self.graph_retriever is not None:
            close = getattr(self.graph_retriever, "aclose", None)
            if close is not None:
                await close()


class ServiceMetrics:
    """Counters and a latency sample, without an external dependency.

    Latencies are kept as a bounded ring so percentiles stay honest without
    the memory growing without limit. This is the local view; the same numbers
    go to Application Insights when a connection string is configured.
    """

    MAX_SAMPLES = 1000

    def __init__(self) -> None:
        self.queries = 0
        self.answered = 0
        self.abstained = 0
        self.clarified = 0
        self.errors = 0
        self.latencies: list[float] = []

    def record(self, *, answered: bool, abstained: bool, clarified: bool,
               latency_ms: float) -> None:
        self.queries += 1
        self.answered += int(answered)
        self.abstained += int(abstained)
        self.clarified += int(clarified)
        self.latencies.append(latency_ms)
        if len(self.latencies) > self.MAX_SAMPLES:
            del self.latencies[0]

    def percentiles(self) -> dict[str, float]:
        if not self.latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        ordered = sorted(self.latencies)

        def at(fraction: float) -> float:
            index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
            return round(ordered[index], 1)

        return {
            "p50": at(0.50),
            "p95": at(0.95),
            "p99": at(0.99),
            "max": round(ordered[-1], 1),
        }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Held on app.state rather than in a module global so a test can run two
    # apps in one process, and so nothing can reach these handles without
    # going through a request that has them.
    configure_tracing()
    services = Services()
    await services.start()
    app.state.services = services
    logger.info(
        "retrieval service ready; departments=%s graph=%s",
        services.registry.names(),
        services.graph_retriever is not None,
    )
    try:
        yield
    finally:
        await services.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Azure RAG ETL",
        description=(
            "Grounded question answering over your own documents, with "
            "citations, version awareness, abstention and department scoping."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    from rag.api.routes import router

    app.include_router(router)
    return app


app = create_app()
