"""The gate that decides whether a unit is worth an LLM call.

The project owner's requirement is that the graph uses LLM relationship
extraction for *any* document type. The arithmetic of that requirement is
brutal: 5M documents at 100-200 pages is on the order of a billion pages, and
sending every one of them to a model is not a budget, it is a refusal to have
one. What makes the requirement affordable is that most of those pages carry
no extractable relationship. A running footer, a table of contents, a
signature block, a paragraph of scene-setting prose -- these have entities in
the loosest sense and no obligations, values, or dates for an edge to connect.

So the gate answers one question per unit: does this text look like it states
a rule? Four ways it can fail:

* **table** -- structured data is extracted deterministically and exactly by
  `extraction.tabular`; paying a language model to read a spreadsheet is both
  more expensive and less accurate;
* **too short** -- below `GRAPH_EXTRACT_MIN_TOKENS` the prompt overhead
  exceeds the payload and there is rarely a complete rule in the payload;
* **boilerplate** -- the same content hash seen in
  `BOILERPLATE_DOC_THRESHOLD` distinct documents is extracted once by the
  content-hash cache and skipped everywhere else;
* **low signal** -- weighted density of obligation-bearing markers below
  `GRAPH_EXTRACT_MIN_SIGNAL`.

The decision is a typed object, never a bare bool, because the reason is the
product: the run report ("38% skipped as low-signal, 24% as tables") is how a
threshold gets tuned, and a bool discards exactly the information that tuning
needs.

**Bias note.** The gate is deliberately asymmetric. A false skip loses one
section's relations; a false keep costs money on every one of a billion pages.
The weights below reflect that: hard obligation markers (deontic verbs, money,
percentages) carry full weight, and proper nouns -- which every page has,
including the furniture -- carry a quarter, because proper-noun density
measures "this is English about a company", not "this states a rule".
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from rag.config import Settings, get_settings
from rag.extraction.units import count_tokens
from rag.models import ExtractionUnit


class SkipReason(StrEnum):
    TABLE = "table"
    TOO_SHORT = "too_short"
    BOILERPLATE = "boilerplate"
    LOW_SIGNAL = "low_signal"


@dataclass(frozen=True)
class TriageDecision:
    """Why a unit is or is not going to the LLM, with the numbers behind it."""
    unit_id: str
    extract: bool
    reason: SkipReason | None = None
    tokens: int = 0
    signal: float = 0.0


# --------------------------------------------------------------------------
# Page furniture
#
# pdfplumber and python-docx both leak running headers and footers into the
# body text -- a running footer like "Contoso Ltd. -- Internal Use Only Page 2" turns up
# glued to the top of whatever section straddled the page break. Left in, it
# inflates proper-noun density on exactly the units that deserve to be
# skipped, so it is removed before scoring (but NOT before counting tokens:
# the tokens are still paid for if the unit is sent).
# --------------------------------------------------------------------------

_FURNITURE_PATTERNS = (
    # A whole line that is only a page marker, or ends in one.
    re.compile(r"^\s*\d{1,4}\s*$"),
    re.compile(r"\bpage\s+\d{1,4}(?:\s+of\s+\d{1,4})?\s*$", re.IGNORECASE),
)


def strip_page_furniture(text: str) -> str:
    """Drop running-header/footer lines so they cannot be mistaken for content."""
    kept = [
        line for line in text.splitlines()
        if line.strip() and not any(p.search(line) for p in _FURNITURE_PATTERNS)
    ]
    return "\n".join(kept)


# --------------------------------------------------------------------------
# Signal components -- each a pure function, each independently testable
# --------------------------------------------------------------------------

# Two or more consecutive capitalized tokens: "Meridian Health Group",
# "Employee Assistance Program". A lone capitalized word is not counted --
# every sentence starts with one.
_PROPER_PHRASE_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.'’-]*(?:\s+[A-Z][A-Za-z0-9&.'’-]*)+")

# Standalone all-caps runs: MFA, PTO, CISO, SOW, IRS. Length-capped so a
# shouted legal clause ("EXCEPT AS EXPRESSLY SET FORTH HEREIN") does not read
# as a wall of acronyms.
_ACRONYM_RE = re.compile(r"(?<![A-Za-z])[A-Z]{2,6}(?![A-Za-z])")

_MONEY_RE = re.compile(
    r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|dollars?)\b)",
    re.IGNORECASE,
)

_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent\b)", re.IGNORECASE)

_MONTHS = ("January|February|March|April|May|June|July|August|September"
           "|October|November|December")
_DATE_RE = re.compile(
    rf"\b(?:{_MONTHS})\s+\d{{1,2}}(?:\s*[-–]\s*\d{{1,2}})?(?:,\s*\d{{4}})?"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:19|20)\d{2}\b"
)

# "12 weeks", "(30) days", "90-day period", "6+ months", "30 calendar days".
_DURATION_RE = re.compile(
    r"(?<![\w$])\(?\d{1,4}\)?\+?\s*[-–]?\s*"
    r"(?:business\s+|calendar\s+|consecutive\s+|working\s+|rolling\s+)?"
    r"(?:days?|weeks?|months?|years?|hours?|minutes?)\b",
    re.IGNORECASE,
)
# Recurrence is a Period statement too: "$5,250 per calendar year" bounds an
# obligation just as "12 weeks" does.
_RECURRENCE_RE = re.compile(
    r"\bper\s+(?:calendar\s+)?(?:year|month|week|day|quarter|annum)\b"
    r"|\b(?:annually|monthly|weekly|daily|quarterly|bi-?weekly|semi-?annually)\b",
    re.IGNORECASE,
)

# The language that turns a description into a rule. Negated forms come first
# in each alternation so "must not" is one match rather than a bare "must".
#
# The design names six examples (must / shall / may / is entitled to /
# requires / is eligible for); the set below is that list generalised to the
# registers an enterprise corpus actually uses. Policy documents state rules
# with modals, contracts state them with "agrees to" / "shall" / "obligations"
# / "warrants", and procedure documents state them as negated imperatives
# ("Do not wait for confirmation"). A lexicon covering only the policy
# register silently drops the legal half of the corpus -- measured on
# `source_data/`, it dropped the NDA's obligations and exclusions clauses,
# which are the two most relation-dense sections in that document.
_DEONTIC_RE = re.compile(
    r"\b(?:"
    r"must not|must|shall not|shall|may not|may|should not|should|"
    r"do(?:es)? not|"
    r"(?:is|are|was|were|be)\s+(?:not\s+)?"
    r"(?:entitled|eligible|obligated|obliged|required|responsible|liable"
    r"|permitted|prohibited|bound|expected)|"
    r"requires?|required|agrees?\s+to|undertakes?\s+to|covenants?\s+to|"
    r"warrants?|indemnif(?:y|ies)|"
    r"obligations?|entitlements?|eligibility|prohibited|permitted|mandatory|"
    r"eligible for|entitled to|no later than|at least"
    r")\b",
    re.IGNORECASE,
)

# Full weight on markers that only appear when a rule is being stated; a
# quarter on proper nouns and less on acronyms, which are everywhere.
_WEIGHTS = {
    "deontic": 1.0,
    "money": 1.0,
    "percentages": 1.0,
    "dates": 0.75,
    "durations": 0.75,
    "proper_nouns": 0.25,
    "acronyms": 0.15,
}

# Density is measured over at least this many words. Without a floor a
# six-word fragment containing one "must" scores 0.17 and sails through, which
# is the opposite of what a *density* measure should say about a fragment.
# Set to roughly the word equivalent of `graph_extract_min_tokens`.
_MIN_DENSITY_WINDOW = 30


def count_proper_noun_phrases(text: str) -> int:
    return len(_PROPER_PHRASE_RE.findall(text))


def count_acronyms(text: str) -> int:
    return len(_ACRONYM_RE.findall(text))


def count_money(text: str) -> int:
    return len(_MONEY_RE.findall(text))


def count_percentages(text: str) -> int:
    return len(_PERCENT_RE.findall(text))


def count_dates(text: str) -> int:
    return len(_DATE_RE.findall(text))


def count_durations(text: str) -> int:
    return len(_DURATION_RE.findall(text)) + len(_RECURRENCE_RE.findall(text))


def count_deontic_verbs(text: str) -> int:
    return len(_DEONTIC_RE.findall(text))


@dataclass(frozen=True)
class SignalCounts:
    """Per-component tallies, so a misclassification can be explained."""
    proper_nouns: int
    acronyms: int
    money: int
    percentages: int
    dates: int
    durations: int
    deontic: int
    words: int

    @property
    def weighted(self) -> float:
        return (
            _WEIGHTS["proper_nouns"] * self.proper_nouns
            + _WEIGHTS["acronyms"] * self.acronyms
            + _WEIGHTS["money"] * self.money
            + _WEIGHTS["percentages"] * self.percentages
            + _WEIGHTS["dates"] * self.dates
            + _WEIGHTS["durations"] * self.durations
            + _WEIGHTS["deontic"] * self.deontic
        )

    @property
    def score(self) -> float:
        return self.weighted / max(self.words, _MIN_DENSITY_WINDOW)


def signal_components(text: str) -> SignalCounts:
    """Tally every signal component over `text`, page furniture removed."""
    body = strip_page_furniture(text)
    return SignalCounts(
        proper_nouns=count_proper_noun_phrases(body),
        acronyms=count_acronyms(body),
        money=count_money(body),
        percentages=count_percentages(body),
        dates=count_dates(body),
        durations=count_durations(body),
        deontic=count_deontic_verbs(body),
        words=len(body.split()),
    )


def signal_score(text: str) -> float:
    """Weighted density of obligation-bearing markers in `text`."""
    return signal_components(text).score


# --------------------------------------------------------------------------
# Boilerplate
# --------------------------------------------------------------------------


class BoilerplateIndex:
    """Counts how many *distinct documents* each content hash appears in.

    Distinct documents, not occurrences: a footer repeated on forty pages of
    one contract is that contract's furniture and the unit deduplication in
    `units.build_units` already collapses it. What this catches is the clause
    that is identical across the corpus -- the standard confidentiality
    paragraph, the template signature block -- which is worth extracting once
    and reusing, not re-extracting per document.

    In-memory and per-run by design. The caller owns persistence: a
    whole-corpus backfill populates it in one pass, while incremental ingest
    should carry one forward across runs (or rely on the content-hash cache,
    which reaches the same end by a different route -- it makes the repeat
    calls free rather than preventing them).
    """

    def __init__(self, threshold: int | None = None) -> None:
        self._threshold = (
            threshold if threshold is not None else get_settings().boilerplate_doc_threshold
        )
        self._docs_by_hash: dict[str, set[str]] = defaultdict(set)

    def observe(self, unit: ExtractionUnit) -> None:
        self._docs_by_hash[unit.content_hash()].add(unit.doc_id)

    def observe_all(self, units: list[ExtractionUnit]) -> None:
        for unit in units:
            self.observe(unit)

    def document_count(self, content_hash: str) -> int:
        return len(self._docs_by_hash.get(content_hash, ()))

    def is_boilerplate(self, content_hash: str) -> bool:
        return self.document_count(content_hash) >= self._threshold


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def triage_unit(
    unit: ExtractionUnit,
    settings: Settings | None = None,
    boilerplate: BoilerplateIndex | None = None,
) -> TriageDecision:
    """Decide whether `unit` earns an LLM call.

    Checks run cheapest-first: content type is a field lookup, token count is
    one encode, boilerplate is a dict lookup, and the signal score -- eight
    regex passes -- runs only for units that survived the rest.
    """
    cfg = settings or get_settings()

    if unit.content_type == "table":
        return TriageDecision(unit.unit_id, extract=False, reason=SkipReason.TABLE)

    tokens = count_tokens(unit.text)
    if tokens < cfg.graph_extract_min_tokens:
        return TriageDecision(unit.unit_id, extract=False,
                              reason=SkipReason.TOO_SHORT, tokens=tokens)

    if boilerplate is not None and boilerplate.is_boilerplate(unit.content_hash()):
        return TriageDecision(unit.unit_id, extract=False,
                              reason=SkipReason.BOILERPLATE, tokens=tokens)

    # The section heading is part of what the unit means and part of what the
    # extractor will be shown: "7 Prohibited Practices" over a list of bare
    # gerunds is a prohibition, and scoring the body alone reads it as a
    # neutral list of activities.
    score = signal_score(f"{unit.section_path}\n{unit.text}")
    if score < cfg.graph_extract_min_signal:
        return TriageDecision(unit.unit_id, extract=False, reason=SkipReason.LOW_SIGNAL,
                              tokens=tokens, signal=score)

    return TriageDecision(unit.unit_id, extract=True, tokens=tokens, signal=score)


def triage_units(
    units: list[ExtractionUnit],
    settings: Settings | None = None,
    boilerplate: BoilerplateIndex | None = None,
) -> list[TriageDecision]:
    """Triage a batch, one decision per unit, in the order given.

    When no index is supplied, one is built from `units` themselves -- correct
    for a whole-corpus pass, and a no-op for a single document, where nothing
    can yet be known to repeat across documents.
    """
    cfg = settings or get_settings()
    index = boilerplate
    if index is None:
        index = BoilerplateIndex(cfg.boilerplate_doc_threshold)
        index.observe_all(units)
    return [triage_unit(unit, cfg, index) for unit in units]
