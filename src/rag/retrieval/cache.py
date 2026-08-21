"""Caching answers, without leaking one caller's data to another.

Two layers, in order of trustworthiness:

1. **Exact match** on the normalized question within a scope. Cheap, certain,
   and it covers the real repeat traffic in an enterprise assistant -- the same
   handful of policy questions asked over and over.
2. **Semantic match**, gated behind a deliberately tight similarity threshold.
   This is where caching goes wrong: the documented production failure is a
   loose threshold plus a shared namespace returning one customer's answer to
   another customer's question. So the department scope is part of the cache
   *key*, not a filter applied to the result -- two callers with different
   scopes cannot share an entry even if they ask an identical question, because
   they would be entitled to different evidence.

Entries expire. A policy document changes and yesterday's correct answer
becomes today's wrong one, so a cache with no TTL is a slow-motion
correctness bug.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

_WHITESPACE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Casefold and collapse whitespace, drop trailing punctuation.

    Deliberately conservative. Aggressive normalisation (stemming, stopword
    removal) would collapse questions that mean different things -- "is the
    limit 500" and "is the limit over 500" must not share an entry.
    """
    cleaned = _WHITESPACE.sub(" ", query.strip().lower())
    return cleaned.rstrip("?!. ")


def scope_key(departments: list[str] | None) -> str:
    """A stable identity for what this caller is allowed to see."""
    return "|".join(sorted(d.lower() for d in (departments or [])))


@dataclass
class CacheEntry:
    payload: dict[str, Any]
    stored_at: float
    scope: str
    query: str
    embedding: list[float] | None = None
    hits: int = 0


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def lookups(self) -> int:
        return self.exact_hits + self.semantic_hits + self.misses

    def hit_rate(self) -> float:
        return 0.0 if not self.lookups else (
            (self.exact_hits + self.semantic_hits) / self.lookups
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "lookups": self.lookups,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate(), 3),
        }


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 0.0 if norm == 0 else dot / norm


class QueryCache:
    """Scope-partitioned answer cache with an exact and a semantic layer.

    In-process by design: at this scale a shared Redis buys nothing but an
    operational dependency, and the interface is small enough to swap when a
    second replica exists. The failure modes that matter -- scope leakage and
    staleness -- are properties of the key and the TTL, not of where it lives.
    """

    def __init__(
        self,
        *,
        threshold: float,
        ttl_seconds: int = 900,
        max_entries: int = 512,
        embedder=None,
    ) -> None:
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._embedder = embedder
        # scope -> exact key -> entry. Partitioned by scope so a semantic scan
        # physically cannot reach another scope's entries.
        self._by_scope: dict[str, dict[str, CacheEntry]] = {}
        self.stats = CacheStats()

    @staticmethod
    def _exact_key(query: str, scope: str, history_signature: str) -> str:
        payload = "\x1f".join([normalize_query(query), scope, history_signature])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def history_signature(history) -> str:
        """Follow-ups are only equivalent if their conversation is too.

        "What about Standard?" means something different after a question about
        pricing than after one about cancellation, so history is part of the
        key. Only the user turns matter -- the assistant's own wording varies
        run to run and would defeat the cache without changing the meaning.
        """
        if not history:
            return ""
        user_turns = [
            normalize_query(getattr(t, "content", ""))
            for t in history
            if getattr(t, "role", "") == "user"
        ]
        return hashlib.sha256("\x1f".join(user_turns).encode("utf-8")).hexdigest()[:16]

    def _expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.stored_at) > self._ttl

    def _evict_if_needed(self, bucket: dict[str, CacheEntry]) -> None:
        while len(bucket) > self._max_entries:
            oldest = min(bucket, key=lambda k: bucket[k].stored_at)
            del bucket[oldest]
            self.stats.evictions += 1

    async def get(
        self, query: str, departments: list[str] | None, history=None
    ) -> dict[str, Any] | None:
        scope = scope_key(departments)
        if not scope:
            # An unscoped caller retrieves nothing, so there is nothing to
            # cache or serve for them.
            self.stats.misses += 1
            return None

        bucket = self._by_scope.setdefault(scope, {})
        signature = self.history_signature(history)

        key = self._exact_key(query, scope, signature)
        entry = bucket.get(key)
        if entry is not None:
            if self._expired(entry):
                del bucket[key]
            else:
                entry.hits += 1
                self.stats.exact_hits += 1
                return entry.payload

        semantic = await self._semantic_get(query, bucket, signature)
        if semantic is not None:
            self.stats.semantic_hits += 1
            return semantic

        self.stats.misses += 1
        return None

    async def _semantic_get(
        self, query: str, bucket: dict[str, CacheEntry], signature: str
    ) -> dict[str, Any] | None:
        if self._embedder is None or not bucket:
            return None

        candidates = [
            entry
            for entry in bucket.values()
            if entry.embedding and not self._expired(entry)
            # A cached answer only transfers to another question asked in the
            # same conversational context.
            and entry.payload.get("_history_signature", "") == signature
        ]
        if not candidates:
            return None

        try:
            vector = await self._embedder.embed_one(query)
        except Exception:
            # The cache is an optimisation. If embedding fails, miss.
            return None

        best, best_score = None, 0.0
        for entry in candidates:
            score = cosine(vector, entry.embedding)
            if score > best_score:
                best, best_score = entry, score

        if best is not None and best_score >= self._threshold:
            best.hits += 1
            return best.payload
        return None

    async def put(
        self,
        query: str,
        departments: list[str] | None,
        payload: dict[str, Any],
        history=None,
    ) -> None:
        scope = scope_key(departments)
        if not scope:
            return
        # Never cache an answer the system itself was unsure of: a refusal or a
        # clarification is a function of what happened to be retrieved, and
        # replaying it hides the moment the corpus gains the missing document.
        if not payload.get("answered"):
            return

        signature = self.history_signature(history)
        stored = {**payload, "_history_signature": signature, "_cached": True}

        embedding = None
        if self._embedder is not None:
            try:
                embedding = await self._embedder.embed_one(query)
            except Exception:
                embedding = None

        bucket = self._by_scope.setdefault(scope, {})
        bucket[self._exact_key(query, scope, signature)] = CacheEntry(
            payload=stored,
            stored_at=time.time(),
            scope=scope,
            query=normalize_query(query),
            embedding=embedding,
        )
        self._evict_if_needed(bucket)

    def clear(self) -> None:
        self._by_scope.clear()

    def size(self) -> int:
        return sum(len(bucket) for bucket in self._by_scope.values())
