"""Typed LLM steps.

Each module here owns exactly one decision and returns a validated Pydantic
model rather than free text. The split of responsibility with LangGraph is
deliberate and follows current practice: **LangGraph owns orchestration**
(control flow, branching, conversation state), **pydantic-ai owns each
individual LLM call** (typed input, validated output, retries). Stacking two
agent loops on the same decision buys nothing and makes failures impossible to
attribute.

Everything here shares one model instance so token accounting, deployment
choice and API version are decided in a single place.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider

from rag.config import get_settings


@lru_cache
def get_model() -> OpenAIChatModel:
    """The shared Azure OpenAI chat model.

    The model *name* passed here is the Azure **deployment** name, not the
    upstream model id -- an easy and silent mistake, since a wrong value fails
    at call time rather than construction time.
    """
    settings = get_settings()
    return OpenAIChatModel(
        settings.azure_openai_chat_deployment,
        provider=AzureProvider(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_chat_api_version,
        ),
    )


def build_agent(output_type: type, system_prompt: str, **kwargs) -> Agent:
    """An agent with a fixed output schema and no tools.

    None of these steps needs tool calling: each is one prompt in, one typed
    structure out. Tools would add a loop, and a loop adds latency and failure
    modes to a decision that does not need either.
    """
    return Agent(
        get_model(),
        output_type=output_type,
        system_prompt=system_prompt,
        **kwargs,
    )


@dataclass
class Usage:
    """Token counts for the LLM calls made on this process's behalf.

    Recorded at the point of the call rather than estimated afterwards: a
    tokeniser run over the prompt text is an approximation that silently
    diverges from what is billed, and cost numbers that cannot be reconciled
    with the invoice are not worth reporting.

    Aggregated per step so a cost report can say *which* step is expensive --
    the useful question is never "what did this cost" but "what would I remove".
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    by_step: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, step: str, usage) -> None:
        if usage is None:
            return
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cached = int(getattr(usage, "cache_read_tokens", 0) or 0)

        self.requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_input_tokens += cached

        bucket = self.by_step.setdefault(
            step, {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        )
        bucket["requests"] += 1
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens

    def cost_usd(self, settings=None) -> float:
        settings = settings or get_settings()
        billable_input = max(0, self.input_tokens - self.cached_input_tokens)
        return (
            billable_input / 1_000_000 * settings.cost_per_1m_input
            + self.cached_input_tokens / 1_000_000 * settings.cost_per_1m_cached_input
            + self.output_tokens / 1_000_000 * settings.cost_per_1m_output
        )

    def snapshot(self) -> dict:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cost_usd": round(self.cost_usd(), 6),
            "by_step": {k: dict(v) for k, v in self.by_step.items()},
        }

    def reset(self) -> None:
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0
        self.by_step = {}


# Process-wide, guarded because the pipeline fans retrieval and generation out
# across tasks. A per-request tracker would be tidier but would have to be
# threaded through every node signature; the eval harness resets it between
# runs, which is the only place the distinction matters.
USAGE = Usage()
_usage_lock = threading.Lock()


def record_usage(step: str, result) -> None:
    """Record one pydantic-ai run's usage. Never raises."""
    try:
        usage = result.usage
        # `usage` is a property in pydantic-ai 2.x, but was a method in earlier
        # releases -- tolerate both rather than pinning the behaviour of a
        # dependency into an accounting helper.
        if callable(usage):
            usage = usage()
        with _usage_lock:
            USAGE.record(step, usage)
    except Exception:  # pragma: no cover - accounting must never break a turn
        pass


def is_content_filter_error(exc: Exception) -> bool:
    """Whether `exc` is Azure's content filter rejecting the prompt.

    Azure OpenAI enforces its own Responsible AI policy before the model sees
    the prompt, and a rejection arrives as a 400 with `code: content_filter` --
    not as a normal model response. A prompt-injection attempt is a real,
    routine trigger for this, so an agent that lets the exception escape turns
    a correctly-blocked hostile input into a 500 for the user. Every step that
    passes user text to the model treats this as a decision the platform made,
    not as an outage.
    """
    from pydantic_ai.exceptions import ModelHTTPError

    if not isinstance(exc, ModelHTTPError):
        return False
    body = exc.body
    if isinstance(body, dict):
        if body.get("code") == "content_filter":
            return True
        inner = body.get("innererror")
        if isinstance(inner, dict) and inner.get("code") == "ResponsibleAIPolicyViolation":
            return True
    return "content_filter" in str(body).lower()


__all__ = [
    "get_model", "build_agent", "is_content_filter_error",
    "Usage", "USAGE", "record_usage",
]
