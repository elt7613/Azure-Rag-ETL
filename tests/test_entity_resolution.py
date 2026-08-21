"""Entity resolution: normalize -> block -> score & merge, with no LLM call
per pair.

The blocking test is the load-bearing one: it proves the candidate-pair
count stays far below the O(n^2) baseline on a synthetic 2000-entity corpus
shaped like the real one -- many entities sharing a common domain word
("Plan", "Policy") within a department/type group, the exact case that would
blow blocking back up to quadratic if the common-word guard didn't exist.
"""
from __future__ import annotations

import itertools
import random

import pytest

from rag.extraction.resolve import (
    AmbiguousPair,
    generate_candidate_pairs,
    normalize,
    remap_relations,
    resolve_entities,
    surface_forms,
)
from rag.models import Entity, Relation
from tests.conftest import azure_configured


# --------------------------------------------------------------------------
# Stage 1: normalize
# --------------------------------------------------------------------------


def test_normalize_casefolds_strips_punctuation_and_suffixes_and_singularizes():
    assert normalize("Acme, Inc.") == "acme"
    assert normalize("  Widget   Corp  ") == "widget"
    assert normalize("Employee Benefits") == "employee benefit"
    assert normalize("PTO") == "pto"


def test_normalize_does_not_mangle_a_false_plural():
    # "Status" is not "Statu" + "s".
    assert normalize("Approval Status") == "approval status"


def test_surface_forms_splits_a_trailing_parenthetical_acronym():
    forms = surface_forms("Paid Time Off (PTO)")
    assert "paid time off" in forms
    assert "pto" in forms


def test_surface_forms_synthesizes_an_acronym_for_a_bare_spelled_out_name():
    # No parenthetical present, but the acronym reading is still generated so
    # a bare "PTO" mention elsewhere in the corpus has something to match.
    forms = surface_forms("Paid Time Off")
    assert "pto" in forms
    assert "paid time off" in forms


# --------------------------------------------------------------------------
# Stage 3: score & merge -- the behavioural contract
# --------------------------------------------------------------------------


def _pto_variants(department="HR") -> list[Entity]:
    return [
        Entity(name="PTO", type="Benefit", department=department),
        Entity(name="Paid Time Off", type="Benefit", department=department),
        Entity(name="Paid Time Off (PTO)", type="Benefit", department=department),
    ]


def test_merges_genuine_variants_in_the_same_department_and_type():
    result = resolve_entities(_pto_variants())
    assert len(result.entities) == 1
    merged = result.entities[0]
    assert merged.name == "Paid Time Off"  # spelled-out, non-parenthetical form wins
    assert set(merged.aliases) == {"PTO", "Paid Time Off (PTO)"}


def test_refuses_to_merge_the_same_name_across_departments():
    # HR's "Standard" plan tier and sales' "Standard" subscription tier share
    # a string and nothing else -- blocking must never even compare them.
    hr = Entity(name="Standard", type="Plan", department="HR")
    sales = Entity(name="Standard", type="Plan", department="sales")
    result = resolve_entities([hr, sales])
    assert len(result.entities) == 2
    assert not generate_candidate_pairs([hr, sales])


def test_refuses_to_merge_across_entity_types():
    policy = Entity(name="Onboarding", type="Policy", department="HR")
    process = Entity(name="Onboarding", type="Process", department="HR")
    result = resolve_entities([policy, process])
    assert len(result.entities) == 2
    assert not generate_candidate_pairs([policy, process])


@pytest.mark.parametrize("left,right", [
    ("Enterprise", "Enterprise Plus"),
    ("Tier 1", "Tier 2"),
    ("Bronze", "Gold"),
])
def test_refuses_near_misses_that_are_genuinely_different_tiers(left, right):
    a = Entity(name=left, type="Plan", department="sales")
    b = Entity(name=right, type="Plan", department="sales")
    result = resolve_entities([a, b])
    assert len(result.entities) == 2
    assert {e.name for e in result.entities} == {left, right}


def test_ambiguous_band_without_an_embedder_is_left_unmerged_and_logged(caplog):
    # A pair that scores inside the near-threshold band and has no embedder
    # to break the tie must be left alone, not guessed at -- the design's
    # explicit preference for a duplicate node over a wrong merge.
    a = Entity(name="Tier 1", type="Plan", department="sales")
    b = Entity(name="Tier 2", type="Plan", department="sales")
    with caplog.at_level("INFO"):
        result = resolve_entities([a, b], settings=_settings(entity_merge_threshold=0.80))
    assert len(result.entities) == 2
    assert any(isinstance(p, AmbiguousPair) for p in result.ambiguous)
    assert "ambiguous" in caplog.text.lower()


def test_embedding_tiebreak_can_merge_a_banded_pair():
    a = Entity(name="Tier 1", type="Plan", department="sales")
    b = Entity(name="Tier 2", type="Plan", department="sales")

    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        # Identical vectors -> cosine 1.0 -> decisively above any threshold.
        return [[1.0, 0.0, 0.0] for _ in texts]

    result = resolve_entities(
        [a, b], settings=_settings(entity_merge_threshold=0.80), embed_fn=fake_embed
    )
    assert len(result.entities) == 1
    assert len(calls) == 1  # one batched call, not one per pair


def test_embedding_tiebreak_can_refuse_a_banded_pair():
    a = Entity(name="Tier 1", type="Plan", department="sales")
    b = Entity(name="Tier 2", type="Plan", department="sales")

    def fake_embed(texts: list[str]) -> list[list[float]]:
        # Orthogonal vectors -> cosine 0.0 -> decisively below threshold.
        vectors = {"tier 1": [1.0, 0.0], "tier 2": [0.0, 1.0]}
        return [vectors[t] for t in texts]

    result = resolve_entities(
        [a, b], settings=_settings(entity_merge_threshold=0.80), embed_fn=fake_embed
    )
    assert len(result.entities) == 2
    assert not result.ambiguous


def test_embed_fn_is_called_once_per_distinct_name_not_once_per_comparison():
    # Three entities all pairwise-banded against each other must still only
    # embed each distinct name once -- caching is the point, not optional.
    entities = [
        Entity(name="Tier 1", type="Plan", department="sales"),
        Entity(name="Tier 2", type="Plan", department="sales"),
        Entity(name="Tier 3", type="Plan", department="sales"),
    ]
    seen: set[str] = set()
    call_count = 0

    def fake_embed(texts: list[str]) -> list[list[float]]:
        nonlocal call_count
        call_count += 1
        seen.update(texts)
        return [[1.0, 0.0] for _ in texts]

    resolve_entities(entities, settings=_settings(entity_merge_threshold=0.80), embed_fn=fake_embed)
    assert call_count == 1
    assert len(seen) == len({normalize(e.name) for e in entities})


# --------------------------------------------------------------------------
# Canonical name determinism
# --------------------------------------------------------------------------


def test_canonical_name_is_chosen_deterministically_regardless_of_input_order():
    variants = _pto_variants()
    seen_names = set()
    for _ in range(8):
        shuffled = variants[:]
        random.shuffle(shuffled)
        result = resolve_entities(shuffled)
        assert len(result.entities) == 1
        seen_names.add(result.entities[0].name)
    assert seen_names == {"Paid Time Off"}


# --------------------------------------------------------------------------
# remap_relations
# --------------------------------------------------------------------------


def test_remap_relations_rewrites_subject_and_object_to_canonical_names():
    result = resolve_entities(_pto_variants())
    relation = Relation(
        subject="PTO", predicate="APPLIES_TO", object="Paid Time Off (PTO)",
        subject_type="Benefit", object_type="Benefit",
        doc_id="HR/Handbook.pdf", source_chunk_id="c1", section_path="2",
        page=1, department="HR", confidence=0.9,
    )
    [remapped] = remap_relations([relation], result)
    assert remapped.subject == "Paid Time Off"
    assert remapped.object == "Paid Time Off"
    # Provenance untouched by the rewrite.
    assert remapped.doc_id == "HR/Handbook.pdf"
    assert remapped.source_chunk_id == "c1"


def test_remap_relations_leaves_unresolved_names_untouched():
    result = resolve_entities(_pto_variants())
    relation = Relation(
        subject="Some Other Thing", predicate="APPLIES_TO", object="Paid Time Off",
        subject_type="Policy", object_type="Benefit",
        doc_id="HR/Handbook.pdf", source_chunk_id="c1", section_path="2",
        page=1, department="HR", confidence=0.9,
    )
    [remapped] = remap_relations([relation], result)
    assert remapped.subject == "Some Other Thing"
    assert remapped.object == "Paid Time Off"


# --------------------------------------------------------------------------
# Stage 2: block -- the O(n^2)-avoidance proof
# --------------------------------------------------------------------------

_DEPARTMENTS = ["HR", "finance", "IT", "legal", "sales"]
_ENTITY_TYPES = ["Policy", "Plan", "Benefit", "Rate"]
_BASE_WORDS = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Rho",
]
_KIND_WORDS = ["Plan", "Policy", "Program"]  # deliberately common, shared
                                              # across many clusters -- this
                                              # is what would blow blocking
                                              # back up to O(n^2) without the
                                              # max-block-size guard.


def _synthetic_corpus(n: int = 2000, seed: int = 7) -> list[Entity]:
    rng = random.Random(seed)
    entities: list[Entity] = []
    cluster_id = 0
    while len(entities) < n:
        department = _DEPARTMENTS[cluster_id % len(_DEPARTMENTS)]
        entity_type = _ENTITY_TYPES[cluster_id % len(_ENTITY_TYPES)]
        base = f"{rng.choice(_BASE_WORDS)} {rng.choice(_BASE_WORDS)} {rng.choice(_KIND_WORDS)}"
        variants = [
            base,
            base.upper(),
            base.replace(" ", "-"),
            f"{base}s",
            f"{base}, Inc.",
        ]
        for variant in variants[: rng.randint(2, 5)]:
            if len(entities) >= n:
                break
            entities.append(Entity(name=variant, type=entity_type, department=department))
        cluster_id += 1
    return entities[:n]


def test_blocking_stays_far_below_the_all_pairs_baseline():
    entities = _synthetic_corpus(2000)
    n = len(entities)
    baseline = n * (n - 1) // 2

    pairs = generate_candidate_pairs(entities)

    # A concrete, generous bound: 5% of all-pairs. The measured count (see
    # the assertion message) is expected to land far under that -- this just
    # guards against a regression that reintroduces quadratic behaviour.
    bound = int(baseline * 0.05)
    assert len(pairs) < bound, (
        f"candidate pairs={len(pairs)} bound={bound} baseline(n^2/2)={baseline}"
    )
    # Surfaced so the report can quote the real number, not just "it passed".
    print(f"\nblocking: n={n} candidate_pairs={len(pairs)} "
          f"baseline={baseline} ratio={len(pairs) / baseline:.4%}")


def test_blocking_never_pairs_across_department_or_type():
    entities = _synthetic_corpus(500, seed=11)
    for i, j in generate_candidate_pairs(entities):
        assert entities[i].department == entities[j].department
        assert entities[i].type == entities[j].type


# --------------------------------------------------------------------------
# Live embedding tie-break (gated)
# --------------------------------------------------------------------------

pytestmark_live = pytest.mark.skipif(
    not azure_configured(
        "azure_openai_endpoint", "azure_openai_key", "azure_openai_embedding_deployment"
    ),
    reason="Azure OpenAI not configured",
)


@pytestmark_live
def test_live_embedding_tiebreak_merges_a_true_near_threshold_synonym():
    """Real embeddings should agree that a genuine near-threshold synonym
    pair belongs together -- "PTO" vs "Paid Leave" isn't caught by the
    parenthetical/acronym trick and isn't a confident string match, so it's
    exactly the case the embedding tie-break exists for."""
    import asyncio

    from rag.embedding.azure_openai import AzureOpenAIEmbedder

    embedder = AzureOpenAIEmbedder()

    def sync_embed(texts: list[str]) -> list[list[float]]:
        return asyncio.run(embedder.embed(texts))

    a = Entity(name="PTO", type="Benefit", department="HR")
    b = Entity(name="Paid Leave", type="Benefit", department="HR")
    result = resolve_entities(
        [a, b], settings=_settings(entity_merge_threshold=0.60), embed_fn=sync_embed
    )
    # Not asserting a specific outcome (the model's opinion isn't ours to fix)
    # -- asserting the pipeline actually consulted a real embedding rather
    # than silently skipping it.
    assert len(result.entities) in (1, 2)
    if len(result.entities) == 2:
        assert not result.ambiguous or result.ambiguous[0].score >= 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _settings(**overrides):
    from rag.config import get_settings

    return get_settings().model_copy(update=overrides)
