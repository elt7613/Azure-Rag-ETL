"""The query cache — correctness first, hit rate second."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag.retrieval.cache import QueryCache, cosine, normalize_query, scope_key


@dataclass
class FakeTurn:
    role: str
    content: str


class FakeEmbedder:
    """Deterministic toy embeddings: one dimension per keyword."""

    KEYWORDS = ["pto", "leave", "vpn", "price", "enterprise", "standard"]

    def __init__(self) -> None:
        self.calls = 0

    async def embed_one(self, text: str) -> list[float]:
        self.calls += 1
        lowered = text.lower()
        return [1.0 if k in lowered else 0.0 for k in self.KEYWORDS]


ANSWER = {"answered": True, "answer": "20 days", "citations": [{"doc_id": "HR/x.pdf"}]}


def test_normalization_is_conservative():
    assert normalize_query("  What is  the  LIMIT? ") == "what is the limit"
    # Questions that mean different things must not collapse together.
    assert normalize_query("is the limit 500") != normalize_query("is the limit over 500")


def test_scope_key_is_order_independent_and_case_insensitive():
    assert scope_key(["HR", "finance"]) == scope_key(["finance", "hr"])
    assert scope_key([]) == ""
    assert scope_key(None) == ""


def test_cosine_handles_degenerate_input():
    assert cosine([], [1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


async def test_exact_hit_returns_the_stored_answer():
    cache = QueryCache(threshold=0.93)
    await cache.put("How much PTO do I get?", ["HR"], ANSWER)

    hit = await cache.get("how much pto do i get", ["HR"])
    assert hit is not None
    assert hit["answer"] == "20 days"
    assert hit["_cached"] is True
    assert cache.stats.exact_hits == 1


async def test_a_different_scope_never_reads_another_scopes_entry():
    """The documented way semantic caches leak data between tenants."""
    cache = QueryCache(threshold=0.93, embedder=FakeEmbedder())
    await cache.put("What is the PTO policy?", ["HR"], ANSWER)

    assert await cache.get("What is the PTO policy?", ["IT"]) is None
    assert await cache.get("What is the PTO policy?", ["HR", "IT"]) is None
    assert await cache.get("What is the PTO policy?", ["HR"]) is not None


async def test_unscoped_caller_neither_reads_nor_writes():
    cache = QueryCache(threshold=0.93)
    await cache.put("What is the PTO policy?", [], ANSWER)
    assert cache.size() == 0
    assert await cache.get("What is the PTO policy?", []) is None


async def test_semantic_hit_within_scope():
    cache = QueryCache(threshold=0.9, embedder=FakeEmbedder())
    await cache.put("What is the PTO leave policy?", ["HR"], ANSWER)

    hit = await cache.get("Tell me about the PTO leave policy", ["HR"])
    assert hit is not None
    assert cache.stats.semantic_hits == 1


async def test_dissimilar_question_is_a_miss_not_a_near_hit():
    cache = QueryCache(threshold=0.93, embedder=FakeEmbedder())
    await cache.put("What is the PTO leave policy?", ["HR"], ANSWER)

    assert await cache.get("How do I set up the VPN?", ["HR"]) is None
    assert cache.stats.misses >= 1


async def test_refusals_are_never_cached():
    """Caching 'I don't know' hides the day the document arrives."""
    cache = QueryCache(threshold=0.93)
    await cache.put(
        "What is the severance policy?",
        ["HR"],
        {"answered": False, "abstained": True, "answer": "I don't have that."},
    )
    assert cache.size() == 0
    assert await cache.get("What is the severance policy?", ["HR"]) is None


async def test_follow_ups_are_keyed_by_conversation():
    """'What about Standard?' means different things in different threads."""
    cache = QueryCache(threshold=0.93)
    pricing = [FakeTurn("user", "What is the Enterprise price?")]
    cancellation = [FakeTurn("user", "What is the Enterprise cancellation policy?")]

    await cache.put("What about Standard?", ["sales"], ANSWER, history=pricing)

    assert await cache.get("What about Standard?", ["sales"], history=pricing) is not None
    assert await cache.get("What about Standard?", ["sales"], history=cancellation) is None


async def test_assistant_wording_does_not_affect_the_key():
    cache = QueryCache(threshold=0.93)
    first = [FakeTurn("user", "What is the PTO policy?"),
             FakeTurn("assistant", "Full-time employees accrue 15 days.")]
    second = [FakeTurn("user", "What is the PTO policy?"),
              FakeTurn("assistant", "Employees get 15 days of PTO per year.")]

    await cache.put("And after five years?", ["HR"], ANSWER, history=first)
    assert await cache.get("And after five years?", ["HR"], history=second) is not None


async def test_entries_expire():
    cache = QueryCache(threshold=0.93, ttl_seconds=0)
    await cache.put("What is the PTO policy?", ["HR"], ANSWER)
    assert await cache.get("What is the PTO policy?", ["HR"]) is None


async def test_oldest_entries_are_evicted_past_the_cap():
    cache = QueryCache(threshold=0.93, max_entries=3)
    for i in range(6):
        await cache.put(f"question number {i}", ["HR"], ANSWER)
    assert cache.size() <= 3
    assert cache.stats.evictions >= 3


async def test_a_failing_embedder_degrades_to_a_miss():
    class Broken:
        async def embed_one(self, text: str) -> list[float]:
            raise RuntimeError("embedding service down")

    cache = QueryCache(threshold=0.9, embedder=Broken())
    await cache.put("What is the PTO policy?", ["HR"], ANSWER)
    # Exact match still works; only the semantic layer is lost.
    assert await cache.get("What is the PTO policy?", ["HR"]) is not None
    assert await cache.get("Tell me about time off", ["HR"]) is None


async def test_stats_report_the_hit_rate():
    cache = QueryCache(threshold=0.93)
    await cache.put("What is the PTO policy?", ["HR"], ANSWER)
    await cache.get("What is the PTO policy?", ["HR"])
    await cache.get("Something else entirely", ["HR"])

    stats = cache.stats.as_dict()
    assert stats["exact_hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)
