"""What the run actually cost, in dollars, from the provider's own numbers.

Six of the seven cost controls in the design are claims about money -- triage
skips most units, section units collapse five calls into one, the memo store
makes a re-ingest free, the fixed prefix gets cached at a quarter price,
packing amortises that prefix, batch mode halves it. None of those claims is
worth anything unasserted, and the only way to assert them is to bill every
call from `response.usage` and print the total.

Two things this module is careful about.

**Cached prompt tokens are a separate line item.** Azure bills a prompt-cache
hit at roughly a quarter of the input rate, which is the entire reason the
system prompt is padded past 1024 tokens. Folding cached tokens into the
prompt total would over-state the bill *and* hide whether the cache is
engaging at all -- so `cache_hit_rate` (`prompt_tokens_details.cached_tokens /
prompt_tokens`) is a first-class output. A run reporting 0% is a bug report:
something is varying in the prefix.

**A cache hit bills nothing.** `ExtractionCache` returns results with zeroed
token counts for exactly this reason. The report must say what the run spent,
not what it would have spent -- an inflated "savings" number is how a cost
control gets believed without being true.

The tracker is deliberately dumb about concurrency: it is `+=` under the
GIL from coroutines on one event loop, not threads, so no lock is needed and
adding one would only imply a guarantee that is not being made.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.config import Settings, get_settings

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class CallCost:
    """One API call's tokens and the dollars they came to."""
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    usd: float

    @property
    def uncached_prompt_tokens(self) -> int:
        return self.prompt_tokens - self.cached_tokens


class CostTracker:
    """Per-run token and dollar totals for chat and embedding calls."""

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self._input = cfg.cost_per_1m_input
        self._cached_input = cfg.cost_per_1m_cached_input
        self._output = cfg.cost_per_1m_output
        self._embedding = cfg.cost_per_1m_embedding
        self.calls: list[CallCost] = []
        self.embedding_tokens = 0
        self._embedding_usd = 0.0

    # ---- recording ----

    def record(self, prompt_tokens: int, completion_tokens: int,
               cached_tokens: int = 0) -> CallCost:
        """Bill one chat completion and return what it cost.

        `cached_tokens` is clamped to `prompt_tokens`: a provider reporting
        more cached than prompt tokens would otherwise produce a negative
        uncached count and an under-stated bill, and under-stating is the one
        direction a cost report must never fail in.
        """
        cached = max(0, min(cached_tokens, prompt_tokens))
        uncached = prompt_tokens - cached
        usd = (
            uncached * self._input / _PER_MILLION
            + cached * self._cached_input / _PER_MILLION
            + completion_tokens * self._output / _PER_MILLION
        )
        cost = CallCost(prompt_tokens, completion_tokens, cached, usd)
        self.calls.append(cost)
        return cost

    def record_usage(self, usage: Any) -> CallCost | None:
        """Bill from an OpenAI SDK `usage` object (or None, if the call had no
        usage block -- a streamed or failed response).

        `prompt_tokens_details` is absent on older api-versions and on
        deployments without prompt caching. Absent is read as "nothing was
        cached", the pessimistic reading: assuming a discount that did not
        happen is how a projection ends up half the real invoice.
        """
        if usage is None:
            return None
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        return self.record(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens=cached,
        )

    def record_embedding(self, tokens: int) -> float:
        """Bill embedding tokens, which price differently and are not a chat call."""
        usd = tokens * self._embedding / _PER_MILLION
        self.embedding_tokens += tokens
        self._embedding_usd += usd
        return usd

    # ---- totals ----

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def cached_tokens(self) -> int:
        return sum(c.cached_tokens for c in self.calls)

    @property
    def uncached_prompt_tokens(self) -> int:
        return self.prompt_tokens - self.cached_tokens

    @property
    def chat_usd(self) -> float:
        return sum(c.usd for c in self.calls)

    @property
    def embedding_usd(self) -> float:
        return self._embedding_usd

    @property
    def total_usd(self) -> float:
        return self.chat_usd + self._embedding_usd

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens Azure served from its prompt cache.

        The single number that says whether the fixed >=1024-token prefix is
        earning its keep. Zero over many calls means the prefix is varying --
        a dict ordering, a timestamp, a document id that leaked forward.
        """
        prompt = self.prompt_tokens
        return self.cached_tokens / prompt if prompt else 0.0

    def usd_per_document(self, documents: int) -> float:
        """Run cost divided across the documents it covered, for extrapolation."""
        return self.total_usd / documents if documents else 0.0

    def summary(self) -> str:
        """One loggable line: calls, tokens, cache hit rate, dollars."""
        calls = self.call_count
        parts = [
            f"{calls} call{'' if calls == 1 else 's'}",
            f"{self.prompt_tokens:,} prompt tokens ({self.cache_hit_rate:.1%} cached)",
            f"{self.completion_tokens:,} completion",
        ]
        if self.embedding_tokens:
            parts.append(f"{self.embedding_tokens:,} embedding")
        parts.append(f"${self.total_usd:.6f}")
        return " | ".join(parts)
