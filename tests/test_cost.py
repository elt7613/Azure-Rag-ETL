"""Cost accounting: the arithmetic that decides whether this design is affordable.

Every number here is hand-computed in the test body rather than asserted
against whatever the implementation happens to produce. A cost tracker that
agrees with itself is worthless -- the whole point of the module is to be
checkable by someone holding an Azure invoice.

The rates used are pinned on the Settings object rather than read from `.env`,
so a price change in the environment cannot silently turn these into
tautologies.
"""
from __future__ import annotations

import pytest

from rag.config import get_settings
from rag.observability.cost import CallCost, CostTracker

# Rates chosen to make the arithmetic legible: input is 4x cached, output 16x.
RATES = {
    "cost_per_1m_input": 0.40,
    "cost_per_1m_cached_input": 0.10,
    "cost_per_1m_output": 1.60,
    "cost_per_1m_embedding": 0.02,
}


@pytest.fixture
def tracker() -> CostTracker:
    return CostTracker(get_settings().model_copy(update=RATES))


# ---- the arithmetic ----

def test_one_million_uncached_prompt_tokens_costs_exactly_the_input_rate(tracker):
    cost = tracker.record(prompt_tokens=1_000_000, completion_tokens=0)
    assert cost.usd == pytest.approx(0.40)
    assert tracker.total_usd == pytest.approx(0.40)


def test_a_mixed_call_bills_cached_and_uncached_prompt_tokens_at_different_rates(tracker):
    # 2000 prompt tokens of which 1024 were served from the prompt cache.
    cost = tracker.record(prompt_tokens=2000, completion_tokens=500, cached_tokens=1024)
    expected = (
        (2000 - 1024) * 0.40 / 1_000_000   # uncached prompt
        + 1024 * 0.10 / 1_000_000          # cached prompt
        + 500 * 1.60 / 1_000_000           # completion
    )
    assert expected == pytest.approx(0.0012928)
    assert cost.usd == pytest.approx(expected)


def test_the_cache_discount_is_real_money(tracker):
    """Same call, cached vs not: the ratio is what justifies the >=1024-token
    fixed prefix, so assert it rather than trusting the prose."""
    cold = CostTracker(get_settings().model_copy(update=RATES))
    warm = CostTracker(get_settings().model_copy(update=RATES))
    cold.record(prompt_tokens=1024, completion_tokens=0)
    warm.record(prompt_tokens=1024, completion_tokens=0, cached_tokens=1024)
    assert warm.total_usd == pytest.approx(cold.total_usd / 4)


def test_embedding_tokens_are_billed_on_their_own_rate(tracker):
    usd = tracker.record_embedding(500_000)
    assert usd == pytest.approx(0.01)
    assert tracker.embedding_tokens == 500_000
    assert tracker.total_usd == pytest.approx(0.01)
    # An embedding call is not a chat call and must not inflate the chat counters.
    assert tracker.call_count == 0
    assert tracker.prompt_tokens == 0


def test_totals_accumulate_across_calls(tracker):
    tracker.record(prompt_tokens=1000, completion_tokens=100, cached_tokens=0)
    tracker.record(prompt_tokens=3000, completion_tokens=200, cached_tokens=2048)
    assert tracker.call_count == 2
    assert tracker.prompt_tokens == 4000
    assert tracker.completion_tokens == 300
    assert tracker.cached_tokens == 2048
    assert tracker.uncached_prompt_tokens == 4000 - 2048
    expected = (
        (4000 - 2048) * 0.40 / 1_000_000
        + 2048 * 0.10 / 1_000_000
        + 300 * 1.60 / 1_000_000
    )
    assert tracker.total_usd == pytest.approx(expected)


# ---- the cache hit rate: the number that tells us the prefix is working ----

def test_cache_hit_rate_is_cached_over_prompt_tokens(tracker):
    tracker.record(prompt_tokens=1000, completion_tokens=10, cached_tokens=0)
    tracker.record(prompt_tokens=1000, completion_tokens=10, cached_tokens=1000)
    assert tracker.cache_hit_rate == pytest.approx(0.5)


def test_cache_hit_rate_of_a_run_with_no_calls_is_zero_not_a_crash(tracker):
    assert tracker.cache_hit_rate == 0.0
    assert tracker.total_usd == 0.0


def test_cached_tokens_can_never_exceed_prompt_tokens(tracker):
    """Defensive: a provider reporting cached > prompt would otherwise produce
    a negative uncached count and an under-stated bill."""
    cost = tracker.record(prompt_tokens=100, completion_tokens=0, cached_tokens=500)
    assert cost.cached_tokens == 100
    assert tracker.uncached_prompt_tokens == 0
    assert cost.usd == pytest.approx(100 * 0.10 / 1_000_000)


# ---- reading usage straight off an SDK response ----

class _Details:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class _Usage:
    def __init__(self, prompt, completion, cached):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.prompt_tokens_details = _Details(cached)


def test_records_an_openai_usage_object_including_its_cached_token_detail(tracker):
    cost = tracker.record_usage(_Usage(2000, 500, 1024))
    assert (cost.prompt_tokens, cost.completion_tokens, cost.cached_tokens) == (2000, 500, 1024)
    assert cost.usd == pytest.approx(0.0012928)


def test_a_usage_object_without_cached_detail_is_treated_as_a_full_miss(tracker):
    """Older api-versions omit prompt_tokens_details entirely. Absent must mean
    'assume nothing was cached' -- the pessimistic reading -- not a crash."""
    class _Bare:
        prompt_tokens = 1000
        completion_tokens = 0

    cost = tracker.record_usage(_Bare())
    assert cost.cached_tokens == 0
    assert cost.usd == pytest.approx(1000 * 0.40 / 1_000_000)


def test_missing_usage_records_nothing_rather_than_guessing(tracker):
    assert tracker.record_usage(None) is None
    assert tracker.call_count == 0


# ---- extrapolation, which is the whole reason anyone reads this module ----

def test_per_document_cost_divides_the_run_across_the_documents_it_covered(tracker):
    tracker.record(prompt_tokens=1_000_000, completion_tokens=0)  # $0.40
    assert tracker.usd_per_document(11) == pytest.approx(0.40 / 11)
    assert tracker.usd_per_document(0) == 0.0


# ---- the summary line ----

def test_summary_reports_calls_tokens_hit_rate_and_dollars(tracker):
    tracker.record(prompt_tokens=2000, completion_tokens=500, cached_tokens=1024)
    line = tracker.summary()
    assert "1 call" in line
    assert "2,000" in line          # prompt tokens, thousands-separated
    assert "51.2%" in line          # 1024/2000 cached
    assert "$0.0012" in line or "$0.0013" in line
    assert "\n" not in line         # one line, loggable as-is


def test_summary_of_an_empty_run_still_reads_as_a_sentence(tracker):
    assert "0 calls" in tracker.summary()


def test_call_costs_are_retained_for_per_call_inspection(tracker):
    tracker.record(prompt_tokens=10, completion_tokens=1)
    assert isinstance(tracker.calls[0], CallCost)
    assert tracker.calls[0].prompt_tokens == 10
