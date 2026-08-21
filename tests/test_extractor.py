"""The LLM relationship extractor.

Three kinds of test live here, and the split is deliberate.

*Structural* tests pin the properties that make the design affordable and
auditable and that no live call can prove: the fixed prefix clears Azure's
1024-token prompt-cache threshold, the variable text goes last, and a relation
whose evidence span is not in the source is dropped.

*Fake-client* tests prove the composition -- triage, then cache, then network
-- by counting calls that never happen. "A skipped unit costs nothing" is not
observable from a live run; it is observable from a client that records every
request it is handed.

*Live* tests run the real corpus through the real deployment, because the
claim being made is about cost and quality on real documents, and a mocked
extractor can be made to produce anything.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from rag.config import get_settings
from rag.enrich.chunker import chunk_document
from rag.enrich.metadata import extract_metadata
from rag.extraction.cache import ExtractionCache
from rag.extraction.llm import (
    PROMPT_CACHE_MIN_TOKENS,
    RelationExtractor,
    SYSTEM_PROMPT,
    build_messages,
    evidence_supported,
)
from rag.extraction.ontology import ENTITY_TYPES, RELATION_TYPES, build_extraction_schema
from rag.extraction.units import build_units, count_tokens, pack_units
from rag.models import ExtractionUnit
from rag.observability.cost import CostTracker
from rag.parsing.local_parser import LocalParser

from tests.conftest import azure_configured

SOURCE_DIR = Path(__file__).resolve().parents[1] / "source_data"

# A unit that triage keeps: money, a duration, a deontic verb, proper nouns.
KEPT_TEXT = (
    "Employees who have completed twelve months of continuous service are "
    "eligible for tuition reimbursement of up to $5,250 per calendar year. "
    "Reimbursement requires prior written approval from the employee's manager "
    "and a passing grade of C or better. Courses that do not relate to the "
    "employee's role are excluded from the Northwind Traders program."
)


def _unit(text=KEPT_TEXT, unit_id="unit-1", content_type="prose") -> ExtractionUnit:
    return ExtractionUnit(
        unit_id=unit_id,
        doc_id="HR/Benefits.pdf",
        department="HR",
        section_path="6 Tuition Reimbursement",
        text=text,
        page=3,
        chunk_ids=["chunk-a", "chunk-b"],
        content_type=content_type,
    )


# --------------------------------------------------------------------------
# A fake Azure client: same call surface, no network, and it counts.
# --------------------------------------------------------------------------


class FakeClient:
    """Stands in for `AsyncAzureOpenAI`, recording every request it receives."""

    def __init__(self, responder=None, usage=(1200, 90, 1024)):
        self.requests: list[dict] = []
        self._responder = responder or self._echo_nothing
        self._usage = usage
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @staticmethod
    def _echo_nothing(request: dict) -> dict:
        return {"units": []}

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        payload = self._responder(kwargs)
        if isinstance(payload, BaseException):
            raise payload
        prompt, completion, cached = self._usage
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
                prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            ),
        )


def _labels(request: dict) -> list[str]:
    """The short per-unit ids the extractor put in the user message."""
    import re
    return re.findall(r"^\[(u\d+)\]", request["messages"][-1]["content"], re.MULTILINE)


def _one_relation(label: str, evidence: str, **overrides) -> dict:
    relation = {
        "subject": "Tuition Reimbursement",
        "subject_type": "Benefit",
        "predicate": "REQUIRES",
        "object": "manager approval",
        "object_type": "Obligation",
        "confidence": 0.92,
        "evidence_span": evidence,
    }
    relation.update(overrides)
    return {
        "units": [{
            "unit_id": label,
            "entities": [{"name": "Tuition Reimbursement", "type": "Benefit",
                          "description": "up to $5,250 per calendar year"}],
            "relations": [relation],
        }]
    }


def _extractor(client, cache_path, **kwargs) -> RelationExtractor:
    return RelationExtractor(
        client=client,
        cache=ExtractionCache(cache_path),
        settings=get_settings(),
        **kwargs,
    )


def _run(coro):
    return asyncio.run(coro)


# ==========================================================================
# 1. The fixed prefix -- a hard Azure threshold, not a style preference
# ==========================================================================


def test_the_system_prefix_clears_azures_prompt_cache_threshold():
    """Azure does not cache prompt prefixes below 1024 tokens. Under the line
    every one of millions of calls pays full input price for the ontology and
    the examples; over it, the same tokens cost a quarter."""
    tokens = count_tokens(SYSTEM_PROMPT)
    assert PROMPT_CACHE_MIN_TOKENS == 1024
    assert tokens >= PROMPT_CACHE_MIN_TOKENS, (
        f"system prefix is {tokens} tokens; prompt caching will not engage"
    )


def test_the_prefix_teaches_the_whole_closed_ontology():
    """It is padding if it is not the ontology. Every type the model is allowed
    to emit must be named and explained in the cached prefix."""
    for name in ENTITY_TYPES:
        assert name in SYSTEM_PROMPT, f"entity type {name} missing from the prefix"
    for name in RELATION_TYPES:
        assert name in SYSTEM_PROMPT, f"relation type {name} missing from the prefix"


def test_the_prefix_demonstrates_the_hard_cases():
    """The three cases a naive extractor gets wrong: a conditional rule, a
    statement next to a table, and prose with nothing to extract."""
    assert SYSTEM_PROMPT.count('"units"') >= 3, "expected several worked examples"
    assert '"relations": []' in SYSTEM_PROMPT, (
        "no example shows the empty answer, so the model will invent one"
    )


def test_the_prefix_is_byte_identical_between_calls():
    """Prompt caching keys on an exact prefix match: a timestamp, a doc id, or
    a dict iteration order leaking in here silently disables it."""
    from importlib import reload

    import rag.extraction.llm as llm

    first = SYSTEM_PROMPT
    reload(llm)
    assert llm.SYSTEM_PROMPT == first


def test_the_variable_text_goes_last_and_the_prefix_goes_first():
    pack = pack_units([_unit()], 3000)[0]
    messages = build_messages(pack)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[-1]["role"] == "user"
    assert KEPT_TEXT[:40] in messages[-1]["content"]
    # Nothing document-specific may appear before the user turn.
    assert "Tuition" not in messages[0]["content"]


def test_the_call_uses_strict_structured_outputs_against_the_ontology_schema(tmp_path):
    client = FakeClient()
    _run(_extractor(client, tmp_path / "c.db").extract([_unit()]))
    request = client.requests[0]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"] == build_extraction_schema()
    assert request["model"] == get_settings().azure_openai_chat_deployment


# ==========================================================================
# 2. Composition: triage -> cache -> network, in that order
# ==========================================================================


def test_a_unit_triage_skips_never_reaches_the_network(tmp_path):
    client = FakeClient()
    table = _unit(text="| Tier | Discount |\n| --- | --- |\n| Gold | 15% |",
                  unit_id="t1", content_type="table")
    tiny = _unit(text="See section 4.", unit_id="t2")
    results = _run(_extractor(client, tmp_path / "c.db").extract([table, tiny]))

    assert client.call_count == 0
    assert [r.skipped_reason for r in results] == ["table", "too_short"]
    assert all(r.relations == [] for r in results)


def test_a_result_is_returned_for_every_unit_in_the_order_given(tmp_path):
    client = FakeClient(lambda req: _one_relation(_labels(req)[0], "requires prior written approval"))
    units = [_unit(unit_id="skip-me", text="Too short."), _unit(unit_id="keep-me")]
    results = _run(_extractor(client, tmp_path / "c.db").extract(units))
    assert [r.unit_id for r in results] == ["skip-me", "keep-me"]


def test_a_second_run_over_identical_units_makes_zero_llm_calls(tmp_path):
    """The content-hash memo is the control that makes re-ingest free. If a
    re-run costs anything, it is not working."""
    cache_path = tmp_path / "c.db"
    responder = lambda req: _one_relation(_labels(req)[0], "requires prior written approval")

    first_client = FakeClient(responder)
    first = _run(_extractor(first_client, cache_path).extract([_unit()]))
    assert first_client.call_count == 1
    assert first[0].relations and first[0].from_cache is False

    second_client = FakeClient(responder)
    extractor = _extractor(second_client, cache_path)
    second = _run(extractor.extract([_unit()]))

    assert second_client.call_count == 0, "a cached unit was sent to the network"
    assert second[0].from_cache is True
    assert extractor.stats.cache_hits == 1
    assert [r.predicate for r in second[0].relations] == \
           [r.predicate for r in first[0].relations]


def test_a_cache_hit_is_rebound_to_the_asking_units_provenance(tmp_path):
    """The same clause in two documents must cite the document that asked."""
    cache_path = tmp_path / "c.db"
    responder = lambda req: _one_relation(_labels(req)[0], "requires prior written approval")
    _run(_extractor(FakeClient(responder), cache_path).extract([_unit()]))

    other = _unit(unit_id="unit-2")
    other.doc_id = "finance/ExpensePolicy.pdf"
    other.department = "finance"
    other.chunk_ids = ["chunk-z"]
    client = FakeClient(responder)
    results = _run(_extractor(client, cache_path).extract([other]))

    assert client.call_count == 0
    relation = results[0].relations[0]
    assert relation.doc_id == "finance/ExpensePolicy.pdf"
    assert relation.source_chunk_id == "chunk-z"
    assert relation.department == "finance"


def test_small_units_are_packed_into_one_call(tmp_path):
    """Packing is a cost control: 12 sections must not be 12 calls."""
    units = [_unit(unit_id=f"u{i}", text=KEPT_TEXT + f" Clause {i}.") for i in range(12)]
    client = FakeClient(lambda req: {"units": [
        {"unit_id": label, "entities": [], "relations": []} for label in _labels(req)
    ]})
    _run(_extractor(client, tmp_path / "c.db").extract(units))
    assert client.call_count == 1
    assert len(_labels(client.requests[0])) == 12


# ==========================================================================
# 3. Provenance and the evidence-span guard
# ==========================================================================


@pytest.mark.parametrize("span,expected", [
    ("requires prior written approval", True),
    ("Requires   prior\nwritten approval", True),      # whitespace/case normalized
    ("“requires prior written approval”", True),        # smart quotes stripped
    ("requires prior verbal approval", False),          # one word changed = fabricated
    ("", False),
])
def test_evidence_span_membership_is_checked_against_the_source_text(span, expected):
    assert evidence_supported(span, KEPT_TEXT) is expected


def test_a_relation_quoting_text_that_is_not_there_is_dropped_and_counted(tmp_path):
    """The clearest hallucination signal available, and free to check: the
    model cannot quote a span the source does not contain."""
    client = FakeClient(lambda req: _one_relation(
        _labels(req)[0], "requires approval from the Chief Executive Officer"))
    extractor = _extractor(client, tmp_path / "c.db")
    results = _run(extractor.extract([_unit()]))

    assert results[0].relations == []
    assert extractor.stats.relations_dropped_evidence == 1


def test_a_relation_whose_span_is_real_survives_with_full_provenance(tmp_path):
    client = FakeClient(lambda req: _one_relation(
        _labels(req)[0], "requires prior written approval"))
    results = _run(_extractor(client, tmp_path / "c.db").extract([_unit()]))

    relation = results[0].relations[0]
    assert relation.predicate == "REQUIRES"
    assert relation.doc_id == "HR/Benefits.pdf"
    assert relation.source_chunk_id == "chunk-a"
    assert relation.section_path == "6 Tuition Reimbursement"
    assert relation.page == 3
    assert relation.department == "HR"
    assert 0.0 <= relation.confidence <= 1.0
    assert relation.evidence_span in KEPT_TEXT
    assert relation.deterministic is False


def test_an_off_ontology_predicate_is_coerced_not_written_raw(tmp_path):
    client = FakeClient(lambda req: _one_relation(
        _labels(req)[0], "requires prior written approval", predicate="mandates"))
    results = _run(_extractor(client, tmp_path / "c.db").extract([_unit()]))
    assert [r.predicate for r in results[0].relations] == ["REQUIRES"]


def test_an_uncoercible_predicate_is_dropped_and_counted(tmp_path):
    client = FakeClient(lambda req: _one_relation(
        _labels(req)[0], "requires prior written approval", predicate="vibes_with"))
    extractor = _extractor(client, tmp_path / "c.db")
    results = _run(extractor.extract([_unit()]))
    assert results[0].relations == []
    assert extractor.stats.relations_dropped_predicate == 1


def test_entities_are_stamped_with_the_units_department(tmp_path):
    client = FakeClient(lambda req: _one_relation(
        _labels(req)[0], "requires prior written approval"))
    results = _run(_extractor(client, tmp_path / "c.db").extract([_unit()]))
    assert results[0].entities[0].department == "HR"
    assert results[0].entities[0].type == "Benefit"


def test_a_response_naming_a_unit_that_was_not_sent_is_ignored(tmp_path):
    client = FakeClient(lambda req: _one_relation("u99", "requires prior written approval"))
    extractor = _extractor(client, tmp_path / "c.db")
    results = _run(extractor.extract([_unit()]))
    assert results[0].relations == []
    assert extractor.stats.unattributed_units == 1


# ==========================================================================
# 4. Failure handling: a bad pack must not take the batch down
# ==========================================================================


def _rate_limit() -> RateLimitError:
    request = httpx.Request("POST", "https://example.invalid/chat")
    return RateLimitError("429", response=httpx.Response(429, request=request), body=None)


def test_a_429_is_retried_with_backoff_and_then_succeeds(tmp_path):
    attempts = {"n": 0}

    def responder(req):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return _rate_limit()
        return _one_relation(_labels(req)[0], "requires prior written approval")

    client = FakeClient(responder)
    extractor = _extractor(client, tmp_path / "c.db", backoff_base=0.001)
    results = _run(extractor.extract([_unit()]))

    assert attempts["n"] == 3
    assert len(results[0].relations) == 1


def test_a_permanently_failing_pack_is_recorded_and_the_batch_continues(tmp_path):
    def responder(req):
        if "Clause 0." in req["messages"][-1]["content"]:
            return _rate_limit()
        return _one_relation(_labels(req)[0], "requires prior written approval")

    doomed = _unit(unit_id="doomed", text=KEPT_TEXT + " Clause 0.")
    healthy = _unit(unit_id="healthy")
    # A budget that fits one of these units but not two, so the two land in
    # separate packs and the failure of one is visible in the other's result.
    extractor = _extractor(FakeClient(responder), tmp_path / "c.db",
                           backoff_base=0.001, pack_tokens=100)
    results = _run(extractor.extract([doomed, healthy]))

    by_id = {r.unit_id: r for r in results}
    assert by_id["doomed"].skipped_reason.startswith("error")
    assert by_id["healthy"].relations, "one bad pack took the batch down"
    assert extractor.stats.packs_failed == 1


def test_a_failed_pack_is_not_cached_as_an_empty_result(tmp_path):
    """Caching a failure would make it permanent and free -- the worst of both."""
    cache_path = tmp_path / "c.db"
    extractor = _extractor(FakeClient(lambda req: _rate_limit()), cache_path,
                           backoff_base=0.001)
    _run(extractor.extract([_unit()]))

    retry_client = FakeClient(lambda req: _one_relation(
        _labels(req)[0], "requires prior written approval"))
    results = _run(_extractor(retry_client, cache_path).extract([_unit()]))
    assert retry_client.call_count == 1
    assert results[0].relations


def test_unparseable_json_is_a_failed_pack_not_an_exception(tmp_path):
    class Truncating(FakeClient):
        async def _create(self, **kwargs):
            self.requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"units": [{"unit_id": '),
                    finish_reason="length")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4096,
                                      total_tokens=4106, prompt_tokens_details=None),
            )

    extractor = _extractor(Truncating(), tmp_path / "c.db", backoff_base=0.001)
    results = _run(extractor.extract([_unit()]))
    assert results[0].relations == []
    assert extractor.stats.packs_failed == 1


# ==========================================================================
# 5. Cost accounting is wired to the real usage numbers
# ==========================================================================


def test_every_call_is_billed_including_its_cached_prompt_tokens(tmp_path):
    client = FakeClient(lambda req: _one_relation(
        _labels(req)[0], "requires prior written approval"), usage=(1200, 90, 1024))
    cost = CostTracker(get_settings())
    extractor = _extractor(client, tmp_path / "c.db", cost=cost)
    _run(extractor.extract([_unit()]))

    assert cost.call_count == 1
    assert cost.prompt_tokens == 1200
    assert cost.cached_tokens == 1024
    assert cost.total_usd > 0
    assert cost.cache_hit_rate == pytest.approx(1024 / 1200)


def test_a_cache_hit_bills_nothing(tmp_path):
    cache_path = tmp_path / "c.db"
    responder = lambda req: _one_relation(_labels(req)[0], "requires prior written approval")
    _run(_extractor(FakeClient(responder), cache_path).extract([_unit()]))

    cost = CostTracker(get_settings())
    _run(_extractor(FakeClient(responder), cache_path, cost=cost).extract([_unit()]))
    assert cost.call_count == 0
    assert cost.total_usd == 0.0


# ==========================================================================
# 6. Live: the real corpus, the real deployment
# ==========================================================================


@pytest.fixture(scope="module")
def corpus_units() -> list[ExtractionUnit]:
    """Every section of the real `source_data/` corpus as extraction units."""
    async def build() -> list[ExtractionUnit]:
        units: list[ExtractionUnit] = []
        for path in sorted(SOURCE_DIR.rglob("*")):
            if not path.is_file():
                continue
            doc_id = path.relative_to(SOURCE_DIR).as_posix()
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                continue
            parsed = await LocalParser().parse(data, doc_id)
            meta = extract_metadata(parsed, doc_id)
            units.extend(build_units(chunk_document(parsed, meta), meta))
        return units

    units = asyncio.run(build())
    assert len(units) > 60, f"corpus fixture looks wrong: {len(units)} units"
    return units


live = pytest.mark.skipif(
    not azure_configured("azure_openai_endpoint", "azure_openai_key",
                         "azure_openai_chat_deployment"),
    reason="Azure OpenAI chat deployment not configured",
)


@live
def test_live_extraction_over_the_real_corpus(corpus_units, tmp_path, capsys):
    """The headline test: extract the whole corpus, then check every relation
    against the source text it claims to come from."""
    cost = CostTracker(get_settings())
    extractor = RelationExtractor(cache=ExtractionCache(tmp_path / "live.db"), cost=cost)
    results = asyncio.run(extractor.extract(corpus_units))
    by_id = {u.unit_id: u for u in corpus_units}

    relations = [r for res in results for r in res.relations]
    assert relations, "live extraction produced no relations at all"

    for relation in relations:
        unit = by_id[next(r.unit_id for r in results if relation in r.relations)]
        assert relation.predicate in RELATION_TYPES
        assert relation.subject_type in ENTITY_TYPES
        assert relation.object_type in ENTITY_TYPES
        assert relation.doc_id == unit.doc_id
        assert relation.source_chunk_id in unit.chunk_ids
        assert relation.section_path == unit.section_path
        assert relation.department == unit.department
        assert evidence_supported(relation.evidence_span, unit.text), (
            f"evidence not in source: {relation.evidence_span!r}"
        )

    entities = [e for res in results for e in res.entities]
    assert all(e.type in ENTITY_TYPES for e in entities)
    assert all(e.department for e in entities)

    documents = len({u.doc_id for u in corpus_units})
    with capsys.disabled():
        print("\n" + "=" * 72)
        print(f"LIVE CORPUS EXTRACTION over {documents} documents")
        print(f"  {extractor.stats.summary()}")
        print(f"  {cost.summary()}")
        print(f"  entities={len(entities)} relations={len(relations)}")
        print(f"  $/document = {cost.usd_per_document(documents):.6f}"
              f"   $/1000 documents = {cost.usd_per_document(documents) * 1000:.2f}")
        print("=" * 72)


@live
def test_live_second_run_is_free(corpus_units, tmp_path):
    """Re-ingesting an unchanged corpus must cost nothing."""
    cache = ExtractionCache(tmp_path / "twice.db")
    sample = [u for u in corpus_units if u.content_type == "prose"][:6]

    asyncio.run(RelationExtractor(cache=cache).extract(sample))

    cost = CostTracker(get_settings())
    second = RelationExtractor(cache=cache, cost=cost)
    asyncio.run(second.extract(sample))
    assert cost.call_count == 0
    assert cost.total_usd == 0.0


@live
def test_live_prompt_cache_actually_engages(corpus_units, tmp_path, capsys):
    """The >=1024-token prefix is only worth having if Azure reports it as
    cached. One unit per call, several calls in a row: the prefix is identical
    across them, so after the first Azure must report cached prompt tokens."""
    from rag.extraction.triage import triage_units

    keepers = [u for u, d in zip(corpus_units, triage_units(corpus_units))
               if d.extract and u.content_type == "prose"][:4]
    assert len(keepers) == 4

    cost = CostTracker(get_settings())
    extractor = RelationExtractor(cache=ExtractionCache(tmp_path / "warm.db"), cost=cost)
    for unit in keepers:
        asyncio.run(extractor.extract([unit]))

    assert cost.call_count >= 2, "need at least two calls to observe a warm cache"
    with capsys.disabled():
        print(f"\nprompt-cache hit rate over {cost.call_count} live calls: "
              f"{cost.cache_hit_rate:.1%} "
              f"({cost.cached_tokens:,}/{cost.prompt_tokens:,} tokens)")
    assert cost.cached_tokens > 0, (
        "Azure reported zero cached prompt tokens across "
        f"{cost.call_count} calls -- the fixed prefix is not being cached"
    )
