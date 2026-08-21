"""The extractor: triaged text in, provenance-carrying typed relations out.

This module is where the six preceding cost controls are composed into one
call path, and where the seventh -- the cache-friendly prompt shape -- lives.
The order is the design:

    triage  ->  cache  ->  pack  ->  one call per pack  ->  distribute  ->  cache

A unit that triage skipped and a unit the cache already knows must never reach
the network. That is not an optimisation applied later; it is the reason the
requirement ("LLM relationship extraction for any document type", 5M documents
of 100-200 pages) is answerable at all.

**Why the system prompt is padded to >=1024 tokens.** Azure's prompt cache does
not engage below 1024 tokens, and it matches on an exact prefix. So the
ontology, the extraction rules and the worked examples are assembled once, at
import, into a byte-identical block that leads every request; the unit text is
the only thing that varies and it goes last. Above the threshold those fixed
tokens bill at roughly a quarter of the input rate on every call after the
first. Below it they bill at full rate on all of them. At corpus scale that is
the difference between a line item and a project. The padding is not filler:
the examples cover the three cases a naive extractor gets wrong (a rule with a
condition attached, a sentence that describes a table without being one, and
prose with no extractable relation at all). `CostTracker.cache_hit_rate`
reports whether the cache is actually engaging, because a prefix that silently
starts varying costs 4x and changes nothing else observable.

**Why the evidence span is checked, not trusted.** The stated risk of an
LLM-built graph is invented relationships. The schema forces the model to quote
the substring of the source that justifies each relation, and this module then
checks that the quote is really in the source. A model that fabricates a
relation usually fabricates its evidence too, so a string search catches the
clearest class of hallucination for free -- no second model, no extra call.
Relations that fail are dropped and *counted*, because a silently-filtered
hallucination and a model that never hallucinated look identical in the graph.

**Why a truncated response is split rather than discarded.** Packing trades
one risk for its saving: the budget it packs to counts *input* tokens, but
what overflows is the *output*. A full pack of small units can produce ten
thousand tokens of JSON, and when the completion ceiling cuts that off
mid-token the naive handling loses the entire pack -- thirty-five sections,
billed in full, absent from the graph, and absent identically on every
subsequent run because temperature 0 truncates in the same place. So the
ceiling is named explicitly (`MAX_COMPLETION_TOKENS`), `finish_reason` is
read rather than inferred from a parse failure, and a truncated pack is
halved and retried until it fits. Only a single unit whose own output
overruns is unrecoverable, and that one is reported by name.

**Why not pydantic-ai.** The agents in `rag.agents` use it and should. This is
a batch data path, not a conversation: it needs the raw `usage` block
(including `prompt_tokens_details.cached_tokens`), it needs to survive a pack
failing without unwinding the batch, and it needs the request shape to stay
byte-stable so the prompt cache keeps hitting. Those are reasons to hold the
client directly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAzureOpenAI,
    InternalServerError,
    RateLimitError,
)

from rag.config import Settings, get_settings
from rag.extraction.cache import ExtractionCache
from rag.extraction.ontology import (
    ENTITY_TYPES,
    RELATION_TYPES,
    coerce_entity_type,
    coerce_relation_type,
    response_format,
)
from rag.extraction.triage import BoilerplateIndex, triage_units
from rag.extraction.units import UNIT_FRAMING_TOKENS, UnitPack, count_tokens, pack_units
from rag.models import Entity, ExtractionResult, ExtractionUnit, Relation
from rag.observability.cost import CostTracker

logger = logging.getLogger(__name__)

# Azure serves a prompt-cache hit only for prefixes at or above this length.
# It is a provider constant, not a tuning knob -- hence a module constant with
# a test asserting the assembled prefix clears it.
PROMPT_CACHE_MIN_TOKENS = 1024

# Retry policy. Five attempts at 1s doubling covers the multi-second 429 bursts
# a provisioned deployment produces under concurrency; beyond that the quota is
# genuinely exhausted and sleeping longer just holds a worker hostage.
_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0

# Extraction is a reading task with one correct answer, and a stable request
# keeps the prompt cache warm. Sampling buys nothing here.
_TEMPERATURE = 0.0

# The completion ceiling, stated explicitly rather than left to whatever the
# deployment defaults to.
#
# This is the sharpest cliff in the whole extraction path. A 3000-token pack
# holds around thirty-five units and the JSON it produces runs to roughly ten
# thousand completion tokens -- comfortably inside gpt-4.1-mini's 16k output
# window, but not by the margin one would want. When the model runs out of
# room the JSON stops mid-token, `json.loads` fails, and the ENTIRE pack is
# lost: thirty-five sections that were paid for in full and produced nothing.
# Worse, at temperature 0 it fails the same way on every subsequent run, so
# those sections are permanently absent from the graph rather than
# temporarily.
#
# Two defences, both needed. Naming the budget here means the cliff stops
# moving when the deployment changes underneath us. Detecting
# `finish_reason == "length"` and halving the pack means hitting it costs one
# wasted call rather than the pack's entire contents -- half as many units
# produce half as much JSON, and the split repeats until it fits.
MAX_COMPLETION_TOKENS = 16384

# How many times a pack may be halved before giving up. Halving reaches a
# single unit from any realistic pack in well under this, so the bound exists
# only to make "cannot recurse forever" a property of the code rather than of
# the arithmetic.
_MAX_SPLIT_DEPTH = 8

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


# --------------------------------------------------------------------------
# The fixed prefix
#
# Assembled from the ontology tuples rather than typed out, so a type added to
# `ontology.py` cannot end up in the JSON schema but missing from the prompt --
# the model would then be permitted to emit a label it was never taught. The
# glosses are hand-written because they are the actual teaching content; the
# assertion below makes forgetting one a failure at import, not at runtime.
# --------------------------------------------------------------------------

_ENTITY_GLOSS: dict[str, str] = {
    "Policy": "a named rule set or policy document subject ('Remote Work Policy')",
    "Department": "an organisational unit ('Finance', 'IT Service Desk')",
    "Role": "a job title, person-in-a-position, or class of worker ('Manager', 'contractor')",
    "Benefit": "something the organisation provides to workers ('dental coverage')",
    "Plan": "a named tier, programme, or product ('Silver Plan', 'Enterprise tier')",
    "Rate": "a percentage, discount, multiplier, or unit price ('15% discount')",
    "Vendor": "an external company or counterparty ('Meridian Health Group')",
    "System": "a tool, application, or platform ('the VPN client', 'Concur')",
    "Obligation": "a required action or prohibition stated as a thing ('written approval')",
    "Condition": "a precondition or eligibility criterion ('90 days of service')",
    "Period": "a duration, deadline, date, or recurrence ('12 weeks', 'per calendar year')",
    "Amount": "a currency amount or countable limit ('$5,250', 'three occurrences')",
    "Location": "a place, region, or site ('the London office', 'EU')",
    "Process": "a named procedure or workflow ('expense submission')",
    "Metric": "a measure, threshold, or service level ('99.9% uptime')",
    "Document": "a referenced document, agreement, form, or exhibit ('Exhibit A')",
}

_RELATION_GLOSS: dict[str, str] = {
    "APPLIES_TO": "a rule/policy governs a role, department, plan, or location",
    "ELIGIBLE_FOR": "a role or group qualifies for a benefit, plan, or rate",
    "REQUIRES": "something is a precondition, obligation, or mandatory step for something else",
    "GRANTS": "a policy/benefit confers an entitlement, amount, or period",
    "LIMITS": "a cap, maximum, or ceiling constrains something",
    "EXCLUDES": "something is explicitly not covered, not permitted, or carved out",
    "EXCEPTION_TO": "a specific case overrides or is exempt from a general rule",
    "DEFINED_IN": "a term, rate, or rule is stated in a named document, section, or table",
    "REFERENCES": "a passage points at another document, section, policy, or exhibit",
    "EFFECTIVE_DURING": "something holds for a stated period, date range, or from a date",
    "HAS_VALUE": "an attribute takes a specific amount, rate, or metric value",
    "OWNED_BY": "a system, process, or policy is owned, managed, or maintained by someone",
    "APPROVED_BY": "an action needs sign-off from a named role, committee, or department",
    "SUPERSEDES": "a document or version replaces an earlier one",
}

assert set(_ENTITY_GLOSS) == set(ENTITY_TYPES), "entity glossary is out of sync"
assert set(_RELATION_GLOSS) == set(RELATION_TYPES), "relation glossary is out of sync"


def _glossary(names: tuple[str, ...], glosses: dict[str, str]) -> str:
    # Iterates the ontology tuple, never the dict, so the rendered order is the
    # declared order and cannot shift with a dict edit -- byte-stability of the
    # prefix is what the prompt cache keys on.
    return "\n".join(f"- {name}: {glosses[name]}" for name in names)


_EXAMPLE_CONDITIONAL = """\
EXAMPLE 1 -- a rule with a condition and an approver attached.

INPUT
[u1] HR/Handbook.pdf | 4 Dental Coverage
Contractors who work more than 20 hours per week may enrol in the Meridian
Dental Plan after 90 days of continuous service. Enrolment must be approved by
the Benefits Committee. Coverage is capped at $2,000 per calendar year.

OUTPUT
{"units": [{"unit_id": "u1",
  "entities": [
    {"name": "Meridian Dental Plan", "type": "Plan", "description": "dental coverage for contractors"},
    {"name": "contractors", "type": "Role", "description": "workers over 20 hours per week"},
    {"name": "Benefits Committee", "type": "Role", "description": "approves enrolment"},
    {"name": "90 days of continuous service", "type": "Condition", "description": ""},
    {"name": "$2,000 per calendar year", "type": "Amount", "description": "coverage cap"}],
  "relations": [
    {"subject": "contractors", "subject_type": "Role", "predicate": "ELIGIBLE_FOR",
     "object": "Meridian Dental Plan", "object_type": "Plan", "confidence": 0.95,
     "evidence_span": "Contractors who work more than 20 hours per week may enrol in the Meridian"},
    {"subject": "Meridian Dental Plan", "subject_type": "Plan", "predicate": "REQUIRES",
     "object": "90 days of continuous service", "object_type": "Condition", "confidence": 0.93,
     "evidence_span": "after 90 days of continuous service"},
    {"subject": "Meridian Dental Plan", "subject_type": "Plan", "predicate": "APPROVED_BY",
     "object": "Benefits Committee", "object_type": "Role", "confidence": 0.94,
     "evidence_span": "Enrolment must be approved by the Benefits Committee"},
    {"subject": "Meridian Dental Plan", "subject_type": "Plan", "predicate": "LIMITS",
     "object": "$2,000 per calendar year", "object_type": "Amount", "confidence": 0.9,
     "evidence_span": "Coverage is capped at $2,000 per calendar year"}]}]}

Note what did NOT happen: the condition was not folded into the eligibility
edge and lost, and no edge was invented between the contractor and the
committee -- the text never connects them."""

_EXAMPLE_TABLE_ADJACENT = """\
EXAMPLE 2 -- a statement sitting next to a table. Extract the statement; the
table's own rows are read separately by a deterministic parser, so do not
transcribe rows and do not invent edges for values you can only see in the
grid.

INPUT
[u2] sales/Pricing2026.pdf | 5 Volume Discounts
Volume discount tiers are listed in Table 4 below. Annual prepaid contracts
receive the tier discount shown. Monthly contracts do not qualify for volume
discounts and are billed at list price.

OUTPUT
{"units": [{"unit_id": "u2",
  "entities": [
    {"name": "volume discount", "type": "Rate", "description": "tiered discount"},
    {"name": "Table 4", "type": "Document", "description": "lists the tiers"},
    {"name": "annual prepaid contracts", "type": "Plan", "description": ""},
    {"name": "monthly contracts", "type": "Plan", "description": ""}],
  "relations": [
    {"subject": "volume discount", "subject_type": "Rate", "predicate": "DEFINED_IN",
     "object": "Table 4", "object_type": "Document", "confidence": 0.92,
     "evidence_span": "Volume discount tiers are listed in Table 4 below"},
    {"subject": "annual prepaid contracts", "subject_type": "Plan", "predicate": "ELIGIBLE_FOR",
     "object": "volume discount", "object_type": "Rate", "confidence": 0.9,
     "evidence_span": "Annual prepaid contracts receive the tier discount shown"},
    {"subject": "volume discount", "subject_type": "Rate", "predicate": "EXCLUDES",
     "object": "monthly contracts", "object_type": "Plan", "confidence": 0.93,
     "evidence_span": "Monthly contracts do not qualify for volume discounts"}]}]}"""

_EXAMPLE_EMPTY = """\
EXAMPLE 3 -- prose with nothing to extract. An empty answer is a correct
answer and is expected often. Do not manufacture a relation to avoid returning
nothing; a wrong edge is far more expensive than a missing one, because it is
indistinguishable from a right one until someone acts on it.

INPUT
[u3] HR/Handbook.pdf | 1 Welcome
The Company was founded in 1994 and has grown steadily ever since. We are
proud of the culture our colleagues have built together, and this handbook is
printed on recycled paper.

OUTPUT
{"units": [{"unit_id": "u3", "entities": [], "relations": []}]}"""

_RULES = """\
RULES

1. Evidence is mandatory and must be VERBATIM. `evidence_span` must be an
   exact, contiguous substring of that unit's input text -- copy it, do not
   paraphrase, do not join fragments with an ellipsis, do not fix the spelling
   or the line breaks. A relation whose span is not found in the source is
   discarded automatically, so a paraphrase is the same as no answer.
2. Extract only what the text states. Do not use background knowledge, do not
   complete a rule the passage leaves open, and do not infer an edge because
   two entities appear in the same sentence.
3. Use the closed vocabularies above and nothing else, for `type`,
   `subject_type`, `object_type` and `predicate`.
4. Direction matters. `subject` acts, is governed, or is qualified;
   `object` is what it is governed by, entitled to, or limited to. If the
   natural reading is the inverse of the available predicate, swap the two
   entities rather than inventing a predicate.
5. Name entities as the text names them, minus articles ("the", "a") and
   trailing punctuation. Keep the qualifier when it is part of the identity
   ("full-time employees", not "employees"). Do not expand abbreviations the
   text did not expand.
6. One relation per stated fact. Do not restate the same fact with two
   predicates, and do not emit an edge whose subject and object are the same
   entity.
7. `confidence` is your confidence that the TEXT STATES the relation, not that
   the relation is true in the world. Use 0.9+ for an explicit statement, 0.7-0.9
   when the wording is indirect but unambiguous, and below 0.7 for a reading
   that another careful reader could dispute.
8. Return exactly one object per unit you were given, echoing its `unit_id`
   label ("u1", "u2", ...) exactly. Never merge units and never invent a label.
9. Entities may be listed without participating in a relation; relations may
   reference entities you also listed. Both lists may be empty."""

SYSTEM_PROMPT = f"""\
You extract a knowledge graph from enterprise documents: policies, contracts,
handbooks, price lists, and IT procedures. Each request gives you one or more
short passages, each labelled with an id, a document, and a section heading.
For each passage you return the entities it names and the typed, evidenced
relations it states between them.

The graph you are building is used to answer employee questions with a
citation attached to every claim. That is why evidence spans are mandatory and
why the vocabularies are closed: an edge nobody can trace back to a sentence is
worse than a missing edge, because it looks trustworthy.

ENTITY TYPES (use exactly these labels)
{_glossary(ENTITY_TYPES, _ENTITY_GLOSS)}

RELATION TYPES (use exactly these labels)
{_glossary(RELATION_TYPES, _RELATION_GLOSS)}

{_RULES}

WORKED EXAMPLES

{_EXAMPLE_CONDITIONAL}

{_EXAMPLE_TABLE_ADJACENT}

{_EXAMPLE_EMPTY}

Respond only with the JSON object required by the response schema."""


# --------------------------------------------------------------------------
# Request rendering
# --------------------------------------------------------------------------

# Short, per-pack labels rather than the 40-hex unit ids: a model asked to echo
# a sha1 gets a character wrong often enough to matter, and the label costs ~2
# tokens against ~20. The mapping back to real unit ids is held locally, so a
# garbled label is caught (counted as unattributed) instead of silently
# attaching one section's relations to another's provenance.


def label_units(pack: UnitPack) -> dict[str, ExtractionUnit]:
    """Deterministic short label -> unit, for one pack."""
    return {f"u{i}": unit for i, unit in enumerate(pack.units, start=1)}


def render_pack(pack: UnitPack) -> str:
    """The user turn: the only part of the request that varies per call."""
    blocks = []
    for label, unit in label_units(pack).items():
        heading = unit.section_path or "(no section heading)"
        blocks.append(f"[{label}] {unit.doc_id} | {heading}\n{unit.text.strip()}")
    count = len(pack.units)
    header = f"Extract from the following {count} passage{'' if count == 1 else 's'}."
    return f"{header}\n\n" + "\n\n".join(blocks)


def build_messages(pack: UnitPack) -> list[dict[str, str]]:
    """System prefix first and unchanging, variable text last. That ordering is
    the whole prompt-caching contract; anything document-specific placed in the
    system turn breaks the prefix match and quadruples the input bill."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_pack(pack)},
    ]


# --------------------------------------------------------------------------
# The evidence-span guard
# --------------------------------------------------------------------------

# Quote characters a model wraps a citation in, and the unicode punctuation a
# PDF extractor and a model disagree about. Normalising both sides means a
# smart-quoted or re-wrapped copy of a real sentence still verifies, while a
# changed *word* -- the thing that actually signals fabrication -- still fails.
_QUOTES = "\"'`‘’“”«»"
_PUNCT_FOLD = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
})


def _normalize_for_match(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).translate(_PUNCT_FOLD)
    return " ".join(folded.split()).casefold()


def evidence_supported(span: str, text: str) -> bool:
    """True when `span` really appears in `text`.

    The cheapest hallucination detector available and the only one that costs
    no tokens: a model that invented a relation almost always invented the
    sentence it claims to be quoting. Whitespace, case, and quote-style
    differences are forgiven because they are artefacts of copying; a
    different word is not, because that is the signal.
    """
    if not span or not span.strip():
        return False
    needle = _normalize_for_match(span).strip(_QUOTES + " ")
    if not needle:
        return False
    return needle in _normalize_for_match(text)


# --------------------------------------------------------------------------
# The output-truncation guard
# --------------------------------------------------------------------------


class TruncatedResponse(RuntimeError):
    """The model hit the completion ceiling before it finished the JSON.

    Raised only for the case splitting cannot fix -- a single unit whose own
    output does not fit. It exists as a named type so the failure reads as
    "this section is too dense for the current budget" in the report rather
    than as a generic JSON parse error, which is what it looked like before
    the guard existed and is why the cliff went unnoticed.
    """


def _was_truncated(response: Any) -> bool:
    """True when the provider says it stopped for length rather than because
    the answer was finished.

    Read from `finish_reason` rather than inferred from a JSON parse failure.
    Strict structured outputs make malformed JSON essentially impossible for
    any other reason, so the two are nearly the same test -- but only this one
    can tell "the answer did not fit" from "the answer was wrong", and they
    call for opposite responses: split and retry versus record and move on.
    """
    try:
        return response.choices[0].finish_reason == "length"
    except (AttributeError, IndexError, TypeError):
        return False


def _halve(pack: UnitPack) -> list[UnitPack]:
    """Split a pack into two, recomputing each half's token count.

    Order is preserved, so the units a split pack sends are the same units in
    the same order -- which keeps the request bodies stable enough that the
    fixed prefix still caches.
    """
    midpoint = len(pack.units) // 2
    halves = [pack.units[:midpoint], pack.units[midpoint:]]
    return [
        UnitPack(units=units,
                 token_count=sum(count_tokens(u.text) + UNIT_FRAMING_TOKENS
                                 for u in units))
        for units in halves if units
    ]


# --------------------------------------------------------------------------
# Run accounting
# --------------------------------------------------------------------------


@dataclass
class ExtractionStats:
    """What the run did, and -- as importantly -- what it refused to do.

    The dropped counters exist because a filtered hallucination and a model
    that never hallucinated produce an identical graph. If `dropped_evidence`
    starts climbing, the extraction prompt or the model changed underneath us,
    and this is the only place that would show it.
    """
    units_in: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    cache_hits: int = 0
    duplicate_units: int = 0
    units_extracted: int = 0
    llm_calls: int = 0
    packs_failed: int = 0
    # A truncated pack is a call that was billed in full and returned nothing
    # usable. `packs_split` says how many of those were recovered by halving;
    # the difference between the two is how many were unrecoverable.
    packs_truncated: int = 0
    packs_split: int = 0
    entities_kept: int = 0
    entities_dropped: int = 0
    relations_kept: int = 0
    relations_dropped_evidence: int = 0
    relations_dropped_predicate: int = 0
    relations_dropped_entity_type: int = 0
    relations_dropped_malformed: int = 0
    unattributed_units: int = 0

    @property
    def units_skipped(self) -> int:
        return sum(self.skipped.values())

    def summary(self) -> str:
        skipped = ", ".join(f"{k}={v}" for k, v in sorted(self.skipped.items())) or "none"
        dropped = (
            f"evidence={self.relations_dropped_evidence}, "
            f"predicate={self.relations_dropped_predicate}, "
            f"entity_type={self.relations_dropped_entity_type}, "
            f"malformed={self.relations_dropped_malformed}"
        )
        return (
            f"units={self.units_in} | triaged out {self.units_skipped} ({skipped}) | "
            f"cache hits {self.cache_hits} | duplicates {self.duplicate_units} | "
            f"extracted {self.units_extracted} in {self.llm_calls} calls "
            f"({self.packs_failed} failed, {self.packs_truncated} truncated, "
            f"{self.packs_split} split) | "
            f"relations kept {self.relations_kept}, dropped ({dropped}) | "
            f"entities kept {self.entities_kept}, dropped {self.entities_dropped}"
        )


# --------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------


class RelationExtractor:
    """Turns extraction units into typed relations, as cheaply as the design allows.

    Every collaborator is injectable so the composition can be tested by
    counting calls that never happen: "a triaged-out unit costs nothing" is not
    observable from a live run, only from a client that records its requests.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cache: ExtractionCache | None = None,
        client: Any | None = None,
        cost: CostTracker | None = None,
        boilerplate: BoilerplateIndex | None = None,
        pack_tokens: int | None = None,
        concurrency: int | None = None,
        backoff_base: float = _BACKOFF_BASE_SECONDS,
        max_attempts: int = _MAX_ATTEMPTS,
    ) -> None:
        self._settings = settings or get_settings()
        self._cache = cache if cache is not None else ExtractionCache()
        self._client = client
        self.cost = cost if cost is not None else CostTracker(self._settings)
        self._boilerplate = boilerplate
        self._pack_tokens = pack_tokens or self._settings.graph_extract_pack_tokens
        self._concurrency = concurrency or self._settings.graph_extract_concurrency
        self._backoff_base = backoff_base
        self._max_attempts = max_attempts
        self.stats = ExtractionStats()

    @property
    def client(self) -> Any:
        """Built on first use so importing this module needs no credentials."""
        if self._client is None:
            self._client = AsyncAzureOpenAI(
                azure_endpoint=self._settings.azure_openai_endpoint,
                api_key=self._settings.azure_openai_key,
                # Structured outputs need a newer api-version than embeddings,
                # which is why config carries a separate chat api_version.
                api_version=self._settings.azure_openai_chat_api_version,
            )
        return self._client

    # ---- the pipeline ----

    async def extract(self, units: list[ExtractionUnit]) -> list[ExtractionResult]:
        """One result per input unit, in the order given.

        Skipped and cached units get a result too, carrying `skipped_reason` or
        `from_cache`. The caller therefore sees every unit's outcome and can
        report on the ones that cost nothing -- which is most of them, and the
        whole argument for this design.
        """
        results: dict[str, ExtractionResult] = {}
        self.stats.units_in += len(units)

        # 1. Triage: the units that never earn a call.
        survivors: list[ExtractionUnit] = []
        for unit, decision in zip(
            units, triage_units(units, self._settings, self._boilerplate)
        ):
            if decision.extract:
                survivors.append(unit)
                continue
            reason = decision.reason.value if decision.reason else "skipped"
            self.stats.skipped[reason] += 1
            results[unit.unit_id] = ExtractionResult(unit_id=unit.unit_id,
                                                     skipped_reason=reason)

        # 2. Cache: the units already extracted, here or in a previous run.
        pending: dict[str, ExtractionUnit] = {}
        twins: list[ExtractionUnit] = []
        for unit in survivors:
            content_hash = unit.content_hash()
            hit = self._cache.get(content_hash, unit)
            if hit is not None:
                self.stats.cache_hits += 1
                results[unit.unit_id] = hit
            elif content_hash in pending:
                # Two units with identical text in the same batch: extract one
                # and serve the other from the cache once it lands. This is the
                # within-run half of the boilerplate saving -- the cross-run
                # half is the store itself.
                self.stats.duplicate_units += 1
                twins.append(unit)
            else:
                pending[content_hash] = unit

        # 3. Pack and call.
        packs = pack_units(list(pending.values()), self._pack_tokens)
        if packs:
            semaphore = asyncio.Semaphore(self._concurrency)
            for pack_results in await asyncio.gather(
                *(self._run_pack(pack, semaphore) for pack in packs)
            ):
                results.update(pack_results)

        # 4. The in-batch twins, now that their text has been extracted once.
        for unit in twins:
            hit = self._cache.get(unit.content_hash(), unit)
            results[unit.unit_id] = hit if hit is not None else ExtractionResult(
                unit_id=unit.unit_id,
                skipped_reason="error: extraction of the identical unit failed",
            )

        return [results[unit.unit_id] for unit in units]

    async def _run_pack(
        self, pack: UnitPack, semaphore: asyncio.Semaphore, depth: int = 0
    ) -> dict[str, ExtractionResult]:
        """Extract one pack. Never raises: a dead pack is data, not an abort."""
        if pack.oversized:
            logger.warning(
                "extraction pack holds a single %d-token unit (%s), over the %d budget",
                pack.token_count, pack.units[0].unit_id, self._pack_tokens,
            )
        async with semaphore:
            try:
                response = await self._call_with_retry(pack)
            except Exception as exc:  # noqa: BLE001 - deliberately total
                return self._failed(pack, exc)

        self.stats.llm_calls += 1
        # Billed before anything else looks at the content. A truncated
        # response consumed its whole completion budget and has to appear in
        # the cost report whether or not any of it was usable -- a cost
        # control that hides its own waste is not a cost control.
        self.cost.record_usage(getattr(response, "usage", None))

        if _was_truncated(response):
            return await self._split_and_retry(pack, semaphore, depth)

        try:
            payload = json.loads(response.choices[0].message.content)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            # Strict structured outputs make malformed JSON near-impossible
            # except by truncation, which is handled above -- so anything
            # reaching here is a genuinely broken response, recorded rather
            # than retried, because at temperature 0 it would break identically.
            return self._failed(pack, exc)

        return self._distribute(pack, payload, response)

    async def _split_and_retry(
        self, pack: UnitPack, semaphore: asyncio.Semaphore, depth: int
    ) -> dict[str, ExtractionResult]:
        """Halve a pack that hit the completion ceiling and extract both halves.

        The alternative -- what this replaces -- is to let the truncated JSON
        fail to parse and discard the pack. That throws away up to thirty-five
        sections that were charged for at full price, and does so
        reproducibly: temperature 0 means the next run truncates in exactly
        the same place, so those sections never enter the graph at all. The
        packing budget is measured in *input* tokens, so a pack that fits the
        context can still overrun the output ceiling; halving addresses the
        side that actually overflowed.

        Halving rather than re-packing to a smaller budget because the split
        preserves unit order and unit boundaries -- the cache is keyed on unit
        content, so both halves' units stay individually cacheable and a later
        re-ingest pays nothing for either.

        Bottoming out at a single unit is the one unrecoverable case: its own
        output does not fit, and there is nothing left to split. That is
        recorded as a failed pack and deliberately not cached, so raising
        `MAX_COMPLETION_TOKENS` retries it instead of serving the emptiness
        forever.
        """
        self.stats.packs_truncated += 1
        if len(pack.units) < 2 or depth >= _MAX_SPLIT_DEPTH:
            return self._failed(pack, TruncatedResponse(
                f"response hit the {MAX_COMPLETION_TOKENS}-token completion ceiling "
                f"and the pack ({len(pack.units)} unit(s), depth {depth}) "
                f"cannot be split further"
            ))

        halves = _halve(pack)
        self.stats.packs_split += 1
        logger.info(
            "extraction pack of %d units was truncated; retrying as %d smaller packs",
            len(pack.units), len(halves),
        )
        results: dict[str, ExtractionResult] = {}
        for half_results in await asyncio.gather(
            *(self._run_pack(half, semaphore, depth + 1) for half in halves)
        ):
            results.update(half_results)
        return results

    async def _call_with_retry(self, pack: UnitPack) -> Any:
        """One chat completion, retrying the transient failures only.

        Backoff is exponential with jitter: without jitter, `GRAPH_EXTRACT_
        CONCURRENCY` workers throttled by the same 429 wake up together and
        reproduce the burst that caused it.
        """
        delay = self._backoff_base
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await self.client.chat.completions.create(
                    model=self._settings.azure_openai_chat_deployment,
                    messages=build_messages(pack),
                    response_format=response_format(),
                    temperature=_TEMPERATURE,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )
            except _RETRYABLE as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code < 500:
                    raise  # 400s are our bug; repeating them just costs latency
                last_error = exc
            if attempt == self._max_attempts:
                break
            wait = _retry_after(last_error)
            if wait is None:
                wait = min(delay, _BACKOFF_CAP_SECONDS) * (1.0 + random.random() * 0.25)
                delay *= 2
            logger.info("extraction call attempt %d/%d failed (%s); retrying in %.2fs",
                        attempt, self._max_attempts, type(last_error).__name__, wait)
            await asyncio.sleep(wait)
        raise last_error if last_error else RuntimeError("extraction call failed")

    # ---- turning a response into relations we are willing to write ----

    def _distribute(self, pack: UnitPack, payload: Any, response: Any
                    ) -> dict[str, ExtractionResult]:
        """Attribute a pack's findings back to the units that produced them.

        Every unit in the pack gets a result, including the ones the model
        returned nothing for: "there is nothing here" is a real answer worth
        caching, and re-asking it on the next ingest would be paying twice for
        the same silence.
        """
        labels = label_units(pack)
        entries: dict[str, dict] = {}
        for entry in (payload or {}).get("units", []) if isinstance(payload, dict) else []:
            label = entry.get("unit_id") if isinstance(entry, dict) else None
            if label not in labels:
                self.stats.unattributed_units += 1
                continue
            entries[label] = entry

        usage = getattr(response, "usage", None)
        prompt_share, completion_share, cached_share = _token_shares(usage, len(pack.units))

        results: dict[str, ExtractionResult] = {}
        for label, unit in labels.items():
            entry = entries.get(label, {})
            result = ExtractionResult(
                unit_id=unit.unit_id,
                entities=self._entities(entry.get("entities") or [], unit),
                relations=self._relations(entry.get("relations") or [], unit),
                prompt_tokens=prompt_share,
                completion_tokens=completion_share,
                cached_tokens=cached_share,
            )
            self.stats.units_extracted += 1
            self._cache.put(unit.content_hash(), result)
            results[unit.unit_id] = result
        return results

    def _entities(self, raw: list, unit: ExtractionUnit) -> list[Entity]:
        entities: list[Entity] = []
        for item in raw:
            if not isinstance(item, dict):
                self.stats.entities_dropped += 1
                continue
            name = str(item.get("name") or "").strip()
            entity_type = coerce_entity_type(item.get("type"))
            if not name or entity_type is None:
                self.stats.entities_dropped += 1
                continue
            entities.append(Entity(
                name=name,
                type=entity_type,
                department=unit.department,
                description=str(item.get("description") or "").strip(),
            ))
            self.stats.entities_kept += 1
        return entities

    def _relations(self, raw: list, unit: ExtractionUnit) -> list[Relation]:
        """Validate, stamp with provenance, or drop -- and count every drop.

        The four rejection paths are independent checks on independent failure
        modes: a malformed edge (the model lost the plot), an off-ontology
        predicate (it invented a label), an off-ontology endpoint type (it
        invented a node label), and an unfindable evidence span (it invented
        the sentence). Only the last is a hallucination in the interesting
        sense, which is why it is counted separately rather than lumped in.
        """
        chunk_id = unit.chunk_ids[0] if unit.chunk_ids else ""
        relations: list[Relation] = []
        for item in raw:
            if not isinstance(item, dict):
                self.stats.relations_dropped_malformed += 1
                continue
            subject = str(item.get("subject") or "").strip()
            obj = str(item.get("object") or "").strip()
            if not subject or not obj or subject.casefold() == obj.casefold():
                self.stats.relations_dropped_malformed += 1
                continue

            predicate = coerce_relation_type(item.get("predicate"))
            if predicate is None:
                self.stats.relations_dropped_predicate += 1
                continue

            subject_type = coerce_entity_type(item.get("subject_type"))
            object_type = coerce_entity_type(item.get("object_type"))
            if subject_type is None or object_type is None:
                self.stats.relations_dropped_entity_type += 1
                continue

            span = str(item.get("evidence_span") or "")
            if not evidence_supported(span, unit.text):
                self.stats.relations_dropped_evidence += 1
                logger.debug("dropped relation with unsupported evidence in %s: %r",
                             unit.unit_id, span[:120])
                continue

            relations.append(Relation(
                subject=subject,
                predicate=predicate,
                object=obj,
                subject_type=subject_type,
                object_type=object_type,
                doc_id=unit.doc_id,
                # The unit is a whole section, so any of its chunks is a valid
                # citation; the first is the one a reader lands on.
                source_chunk_id=chunk_id,
                section_path=unit.section_path,
                page=unit.page,
                department=unit.department,
                confidence=_clamp(item.get("confidence")),
                evidence_span=span.strip(),
            ))
            self.stats.relations_kept += 1
        return relations

    def _failed(self, pack: UnitPack, exc: BaseException) -> dict[str, ExtractionResult]:
        """Record a dead pack and hand back empty results for its units.

        Deliberately NOT written to the cache. Memoizing a failure would make
        it permanent and free: the next run would serve the empty result
        instantly and the sections would never be extracted at all.
        """
        self.stats.packs_failed += 1
        reason = f"error: {type(exc).__name__}: {exc}"[:300]
        logger.warning("extraction pack of %d units failed: %s", len(pack.units), reason)
        return {
            unit.unit_id: ExtractionResult(unit_id=unit.unit_id, skipped_reason=reason)
            for unit in pack.units
        }


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _retry_after(error: Exception | None) -> float | None:
    """The server's own back-off instruction, when it sent one.

    Azure returns `retry-after` on a 429 and it is better information than any
    exponential curve -- it knows when the quota window rolls over.
    """
    response = getattr(error, "response", None)
    header = getattr(response, "headers", {}) or {}
    try:
        seconds = float(header.get("retry-after", ""))
    except (TypeError, ValueError):
        return None
    return min(max(seconds, 0.0), _BACKOFF_CAP_SECONDS)


def _token_shares(usage: Any, units: int) -> tuple[int, int, int]:
    """Split a pack's usage evenly across its units, for per-unit reporting only.

    An even split is an approximation and the honest place for the real number
    is `CostTracker`, which bills the call once and exactly. This exists so a
    single `ExtractionResult` can say roughly what it cost without the caller
    having to reconstruct its pack.
    """
    if usage is None or units <= 0:
        return 0, 0, 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    return (
        (getattr(usage, "prompt_tokens", 0) or 0) // units,
        (getattr(usage, "completion_tokens", 0) or 0) // units,
        cached // units,
    )
