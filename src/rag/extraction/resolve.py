"""Entity resolution: collapsing surface-form duplicates without an LLM call
per pair.

The extractor sees one section at a time, so it has no way to know that "PTO"
in the leave policy and "Paid Time Off" three documents later are the same
node -- each mention is minted as its own `Entity`. Left alone, the graph
fills with near-duplicates and an entity-anchored traversal starting from
"PTO" never finds the edges attached to "Paid Time Off". Asking an LLM to
judge every pair is the textbook O(n^2) mistake this whole extraction design
exists to avoid (see `extraction.triage` for the same argument applied to
extraction itself) -- at 5M documents the entity count alone rules it out.

Three stages, cheapest first:

1. **Normalize** (`normalize`, `surface_forms`) -- strip the cosmetic
   differences (case, punctuation, legal suffixes, plurals) and split a
   trailing parenthetical acronym into both of its readings, so "Paid Time
   Off (PTO)" is known by "paid time off" *and* "pto".
2. **Block** (`generate_candidate_pairs`) -- only compare entities that share
   a `(department, type)` and a normalized token or character trigram. This
   is what keeps the pair count near-linear rather than quadratic.
3. **Score & merge** (`resolve_entities`) -- rapidfuzz string similarity
   decides the easy majority of pairs outright; only the pairs that land in
   the ambiguous band around `entity_merge_threshold` pay for a cached
   embedding cosine tie-break, and only the residue that's still undecided
   after that is left unmerged and logged. A wrong merge silently fuses two
   real, different things into one node with no way back; a missed merge is
   just a duplicate, which is the safer failure.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

from rapidfuzz import fuzz

from rag.config import Settings, get_settings
from rag.models import Entity, Relation

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Stage 1: normalize
# --------------------------------------------------------------------------

# Legal-entity suffixes are noise for resolution purposes: "Acme Inc." and
# "Acme, Inc" and "Acme Corporation" name the same vendor. Matched as whole
# tokens after punctuation stripping, so "Inc" the suffix doesn't eat "Inc"
# inside an unrelated word.
_LEGAL_SUFFIXES = frozenset({
    "inc", "incorporated", "ltd", "limited", "llc", "llp", "corp",
    "corporation", "co", "company", "plc", "gmbh",
})

_PUNCT_RE = re.compile(r"[^\w\s]")

# A trailing parenthetical on an entity name is almost always the acronym
# reading of the name that precedes it ("Paid Time Off (PTO)"), never a
# separate clause -- so it's safe to always try the split.
_PAREN_RE = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*$")

# Words that end in "s" but are not plurals of anything -- stripping the "s"
# would turn "Status" into "Statu" and merge it with something it isn't.
# Short words (<=3 chars) are exempted from singularization entirely for the
# same reason: too little signal to tell "Gas" from "Ga" + plural "s".
_SINGULARIZE_EXCEPTIONS = frozenset({
    "status", "bonus", "business", "access", "process", "address",
    "campus", "focus", "basis", "analysis", "premises", "series",
})


def _singularize(word: str) -> str:
    """Strip a simple plural ending. Deliberately conservative: a missed
    plural just costs a blocking key, a wrongly-stripped one costs a false
    merge, and the latter is the more expensive mistake."""
    if len(word) <= 3 or word in _SINGULARIZE_EXCEPTIONS or word.endswith("ss"):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def normalize(text: str) -> str:
    """Casefold, drop punctuation and legal suffixes, singularize each
    token, collapse whitespace. The ruler every comparison in this module is
    measured against."""
    cleaned = _PUNCT_RE.sub(" ", text.casefold())
    tokens = [t for t in cleaned.split() if t and t not in _LEGAL_SUFFIXES]
    tokens = [_singularize(t) for t in tokens]
    return " ".join(tokens)


def _acronym(normalized_text: str) -> str | None:
    """First-letter acronym of a normalized multi-word phrase, or None for a
    single word -- a one-word name has no acronym reading distinct from
    itself."""
    words = normalized_text.split()
    if len(words) < 2:
        return None
    letters = "".join(w[0] for w in words if w)
    return letters or None


def surface_forms(name: str) -> frozenset[str]:
    """Every normalized string this raw name could plausibly be known by.

    A trailing parenthetical names the same entity two ways at once --
    "Paid Time Off (PTO)" is both "paid time off" and "pto" -- so both
    readings become candidates rather than just the string as written.
    Synthesizing the acronym of the spelled-out form closes the loop the
    other way: a document that mentions bare "PTO" with no parenthetical
    still needs something to match against.
    """
    variants = {name}
    m = _PAREN_RE.match(name.strip())
    if m:
        full, abbr = m.group(1).strip(), m.group(2).strip()
        if full:
            variants.add(full)
        if abbr:
            variants.add(abbr)

    normalized = {normalize(v) for v in variants}
    normalized.discard("")
    for form in list(normalized):
        acr = _acronym(form)
        if acr:
            normalized.add(acr)
    return frozenset(normalized)


# --------------------------------------------------------------------------
# Stage 2: block
# --------------------------------------------------------------------------

# A blocking key shared by an unusually large number of entities in one
# (department, type) group is a common domain word ("plan", "policy") doing
# no discriminating work -- keeping it would let one bucket alone blow the
# comparison count back up towards O(n^2). Dropping keys past this size is
# what makes the near-linear bound in `tests/test_entity_resolution.py` hold
# even when the corpus is full of "X Plan" / "Y Plan" style names.
_MAX_BLOCK_SIZE = 25


def _trigrams(s: str) -> set[str]:
    letters = s.replace(" ", "")
    if len(letters) < 3:
        return {letters} if letters else set()
    return {letters[i : i + 3] for i in range(len(letters) - 2)}


def _blocking_keys(forms: frozenset[str]) -> set[str]:
    keys: set[str] = set()
    for form in forms:
        keys.update(form.split())
        keys.update(_trigrams(form))
    return keys


def _entity_forms(entity: Entity) -> frozenset[str]:
    forms = set(surface_forms(entity.name))
    for alias in entity.aliases:
        forms.update(surface_forms(alias))
    return frozenset(forms)


def generate_candidate_pairs(entities: list[Entity]) -> set[tuple[int, int]]:
    """Index pairs worth comparing: same `(department, type)`, sharing a
    normalized token or character trigram.

    This is the control that makes resolution scale. An all-pairs comparison
    over n entities is O(n^2) -- the same unaffordable pattern
    `extraction.triage` exists to keep out of the LLM path, just restated in
    string-matching form. Two entities that share no word and no three-letter
    fragment are, in practice, never the same thing, so the comparison is
    skipped rather than paid for.
    """
    buckets: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for idx, entity in enumerate(entities):
        group = (entity.department, entity.type)
        for key in _blocking_keys(_entity_forms(entity)):
            buckets[group][key].append(idx)

    pairs: set[tuple[int, int]] = set()
    for keymap in buckets.values():
        for idxs in keymap.values():
            if len(idxs) < 2 or len(idxs) > _MAX_BLOCK_SIZE:
                continue
            uniq = sorted(set(idxs))
            for a in range(len(uniq)):
                for b in range(a + 1, len(uniq)):
                    pairs.add((uniq[a], uniq[b]))
    return pairs


# --------------------------------------------------------------------------
# Stage 3: score & merge
# --------------------------------------------------------------------------

# Width of the band, centered on `entity_merge_threshold`, where a plain
# string score isn't trusted alone and an embedding tie-break is consulted
# (when one is available) before falling back to "leave it, log it".
_BAND = 0.05


def fuzzy_score(left: Entity, right: Entity) -> float:
    """Best pairwise similarity between two entities' surface forms, 0..1.

    Uses the *minimum* of `token_set_ratio` and `token_sort_ratio`, not
    `token_set_ratio` alone. `token_set_ratio` treats the shorter name's
    tokens as a subset of the longer's and scores "Enterprise" vs.
    "Enterprise Plus" at 100 -- exactly the false merge the design calls out
    by name, since those are two distinct subscription tiers in this corpus.
    `token_sort_ratio` (order-independent but not subset-tolerant) is
    sensitive to that extra token, so the minimum of the two keeps the
    property that makes token-set matching useful -- word order doesn't
    matter for genuine synonyms -- without rewarding one name for merely
    containing the other.
    """
    left_forms = _entity_forms(left)
    right_forms = _entity_forms(right)
    best = 0
    for lform in left_forms:
        for rform in right_forms:
            score = min(fuzz.token_set_ratio(lform, rform), fuzz.token_sort_ratio(lform, rform))
            if score > best:
                best = score
    return best / 100.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class AmbiguousPair:
    """A candidate pair whose similarity landed in the uncertain band with no
    trusted signal to decide it. Left as two distinct entities on purpose --
    guessing wrong here silently fuses two real things into one node, which
    is worse than leaving a duplicate for a later pass to clean up."""
    left: str
    right: str
    department: str
    type: str
    score: float


@dataclass
class ResolutionResult:
    """The output of one resolution pass: canonical entities plus the lookup
    `remap_relations` needs to rewrite edges written before resolution ran."""

    entities: list[Entity]
    # (department, entity_type, normalized surface form) -> canonical name.
    # Keyed on every surface form of every merged member, not just the
    # canonical entity's own name, so a relation that used "PTO" resolves
    # even though the canonical spelling is "Paid Time Off".
    name_map: dict[tuple[str, str, str], str] = field(default_factory=dict)
    ambiguous: list[AmbiguousPair] = field(default_factory=list)

    def canonical_name(self, department: str, entity_type: str, name: str) -> str:
        """Canonical spelling for `name`, or `name` itself if resolution
        never saw it -- an unresolved name is left untouched, not dropped."""
        return self.name_map.get((department, entity_type, normalize(name)), name)


def _select_canonical(members: list[Entity]) -> tuple[str, list[str]]:
    """Deterministic canonical name + aliases for a merged cluster.

    Order-independent: the same set of member names always produces the same
    choice no matter what order the extractor happened to emit them in,
    which matters because packing (`extraction.units.pack_units`) processes
    units, and therefore entity mentions, in whatever order a backfill
    schedules them.

    Preference order: a name without a trailing parenthetical (the spelled-
    out or bare form, not the redundant "X (Y)" combination) that was seen
    most often, breaking further ties toward the more descriptive (more
    words) reading, and finally alphabetically so the choice never depends
    on iteration order.
    """
    counts = Counter(entity.name for entity in members)

    def sort_key(name: str) -> tuple:
        has_paren = _PAREN_RE.match(name.strip()) is not None
        word_count = len(normalize(name).split())
        return (has_paren, -counts[name], -word_count, name.casefold())

    canonical = min(counts, key=sort_key)
    aliases = sorted(
        {n for n in counts if n != canonical}
        | {a for e in members for a in e.aliases if a and a != canonical}
    )
    return canonical, aliases


def _select_description(members: list[Entity]) -> str:
    """Longest non-empty description wins -- more text usually means more
    context preserved, and length is a deterministic, order-independent tie
    key like everything else this stage picks."""
    descriptions = [e.description for e in members if e.description]
    if not descriptions:
        return ""
    return max(sorted(descriptions), key=len)


class _UnionFind:
    """Plain union-find over entity indices. Merging is transitive by
    design: if A~B and B~C both clear the bar, A/B/C end up one cluster even
    when A and C were never directly compared (blocking may not have paired
    them). Blocking already restricts every edge to one (department, type)
    group, which keeps that transitivity from reaching across unrelated
    entities -- the risk it can't rule out is a same-group chain like
    A~B~C where A and C alone wouldn't have merged, which is the accepted
    cost of transitive clustering anywhere it's used."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Lower index wins arbitrarily; only cluster *membership* needs
            # to be deterministic here, not which index ends up the root --
            # canonical name selection is a separate, explicitly
            # order-independent step.
            self._parent[max(ra, rb)] = min(ra, rb)


def resolve_entities(
    entities: list[Entity],
    settings: Settings | None = None,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> ResolutionResult:
    """Run all three stages and return canonical entities + a remap table.

    `embed_fn` is an optional, synchronous, *batched* embedding function --
    pure logic stays testable with no network by default (`embed_fn=None`),
    and the near-threshold band simply falls back to "leave it, log it" when
    no embedder is supplied. When one is supplied, it is called at most once
    with the deduplicated set of names actually needed to break a tie, never
    once per pair -- that batching, not the embedding itself, is what keeps
    the tie-break affordable at scale.
    """
    cfg = settings or get_settings()
    if not entities:
        return ResolutionResult(entities=[], name_map={})

    band_lo = cfg.entity_merge_threshold - _BAND
    band_hi = min(cfg.entity_merge_threshold + _BAND, 1.0)

    decisive: list[tuple[int, int, float]] = []
    banded: list[tuple[int, int, float]] = []
    for i, j in sorted(generate_candidate_pairs(entities)):
        score = fuzzy_score(entities[i], entities[j])
        (banded if band_lo <= score < band_hi else decisive).append((i, j, score))

    embed_cache: dict[str, list[float]] = {}
    if banded and embed_fn is not None:
        needed = sorted(
            {normalize(entities[i].name) for i, _, _ in banded}
            | {normalize(entities[j].name) for _, j, _ in banded}
        )
        embed_cache = dict(zip(needed, embed_fn(needed)))

    uf = _UnionFind(len(entities))
    ambiguous: list[AmbiguousPair] = []

    for i, j, score in decisive:
        if score >= cfg.entity_merge_threshold:
            uf.union(i, j)
        # Otherwise a plain non-match: blocking's keys are cheap and
        # imprecise on purpose, so most rejected pairs land here and don't
        # merit a log line, unlike the genuinely ambiguous ones below.

    for i, j, score in banded:
        left_vec = embed_cache.get(normalize(entities[i].name))
        right_vec = embed_cache.get(normalize(entities[j].name))
        if left_vec is not None and right_vec is not None:
            # The embedder gave a decisive answer either way -- that's a
            # resolved tie-break, not ambiguity, whichever way it went.
            if _cosine(left_vec, right_vec) >= cfg.entity_merge_threshold:
                uf.union(i, j)
            continue
        ambiguous.append(
            AmbiguousPair(
                left=entities[i].name, right=entities[j].name,
                department=entities[i].department, type=entities[i].type,
                score=score,
            )
        )
        logger.info(
            "entity resolution: ambiguous pair left unmerged (%s / %s, %s/%s, score=%.3f)",
            entities[i].name, entities[j].name,
            entities[i].department, entities[i].type, score,
        )

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(entities)):
        clusters[uf.find(idx)].append(idx)

    result_entities: list[Entity] = []
    name_map: dict[tuple[str, str, str], str] = {}
    for member_idxs in clusters.values():
        members = [entities[i] for i in member_idxs]
        # Guaranteed by construction: every union came from a candidate pair
        # blocked on a shared (department, type), so a cluster can never mix
        # them.
        department, entity_type = members[0].department, members[0].type
        canonical, aliases = _select_canonical(members)
        description = _select_description(members)
        result_entities.append(
            Entity(
                name=canonical, type=entity_type, department=department,
                aliases=aliases, description=description,
            )
        )
        for member in members:
            for form in _entity_forms(member):
                name_map.setdefault((department, entity_type, form), canonical)

    result_entities.sort(key=lambda e: (e.department, e.type, e.name.casefold()))
    return ResolutionResult(entities=result_entities, name_map=name_map, ambiguous=ambiguous)


def remap_relations(relations: list[Relation], resolution: ResolutionResult) -> list[Relation]:
    """Rewrite relation subject/object to their post-resolution canonical
    names.

    Relations are written during extraction, one section at a time, before
    resolution has seen the whole corpus -- the extractor has no way to know
    "PTO" here is "Paid Time Off" three documents over. A merge that leaves
    relations pointing at the pre-merge name is a merge that broke the
    graph: the canonical `Entity` node exists, but nothing points at it.
    Names resolution never touched are passed through unchanged.
    """
    remapped: list[Relation] = []
    for relation in relations:
        subject = resolution.canonical_name(relation.department, relation.subject_type, relation.subject)
        obj = resolution.canonical_name(relation.department, relation.object_type, relation.object)
        if subject == relation.subject and obj == relation.object:
            remapped.append(relation)
        else:
            remapped.append(
                Relation(
                    subject=subject, predicate=relation.predicate, object=obj,
                    subject_type=relation.subject_type, object_type=relation.object_type,
                    doc_id=relation.doc_id, source_chunk_id=relation.source_chunk_id,
                    section_path=relation.section_path, page=relation.page,
                    department=relation.department, confidence=relation.confidence,
                    evidence_span=relation.evidence_span, deterministic=relation.deterministic,
                )
            )
    return remapped
