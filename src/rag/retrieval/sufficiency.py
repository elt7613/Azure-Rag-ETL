"""Deciding whether the retrieved evidence can support an answer at all.

This is the gate that stops the system inventing an answer to a question the
corpus does not cover. It runs *before* generation, which matters: asking a
model to answer and then checking whether it should have is more expensive, and
a model given weak evidence and an instruction to answer will usually find
something to say.

The signals are deliberately cheap and independent -- no extra LLM call:

- **Reranker score of the best hit.** Azure's cross-encoder read the query
  against the passage. It is the single most informative number available, and
  its 0-4 scale is calibrated enough to threshold on.
- **Score concentration.** One strong hit followed by a cliff means a specific
  answer exists. A flat spread of mediocre hits means the query matched a
  topic, not a fact -- typical of both out-of-scope questions and ambiguous
  ones.
- **Query-term coverage.** What fraction of the question's content words appear
  in the evidence at all. A question about "severance" that retrieves nothing
  containing the word "severance" is almost certainly unanswerable here.

The output is a score plus the reasons behind it, because "why did it refuse?"
must be answerable when someone complains that it should have known.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.retrieval import RetrievedChunk

# Words carrying no topical signal; their presence or absence in the evidence
# says nothing about whether the question can be answered.
_STOPWORDS = frozenset("""
a an and are as at be by can could did do does for from had has have how i if in
is it its may me my of on or our shall should that the their there these they
this to was we what when where which who whom why will with would you your
""".split())

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]+|\$?\d[\d,\.]*%?")


@dataclass
class Sufficiency:
    """The verdict, and enough detail to explain it."""

    score: float
    sufficient: bool
    top_reranker: float | None = None
    coverage: float = 0.0
    concentration: float = 0.0
    missing_terms: list[str] = field(default_factory=list)
    reason: str = ""


def content_terms(text: str) -> list[str]:
    """Content-bearing tokens of a question: no stopwords, no bare punctuation."""
    terms: list[str] = []
    for token in _WORD_RE.findall(text):
        lowered = token.lower()
        if lowered in _STOPWORDS:
            continue
        # Single letters are noise; a single digit is not -- "5 years",
        # "3 days" and "tier 1" all turn on it.
        if len(lowered) < 2 and not lowered.isdigit():
            continue
        terms.append(lowered)
    return terms


def term_coverage(query: str, chunks: list[RetrievedChunk]) -> tuple[float, list[str]]:
    """Fraction of the query's content terms present in the evidence.

    Matching is substring-based on a normalized haystack so that "$5,250"
    matches "5,250", and "leave" matches "leaves" -- morphological near-misses
    would otherwise read as absent evidence and trigger a spurious refusal.
    """
    terms = content_terms(query)
    if not terms:
        return 1.0, []
    haystack = " ".join(c.content for c in chunks).lower()
    missing = [t for t in terms if t.strip("$%,.") not in haystack]
    return (len(terms) - len(missing)) / len(terms), missing


def score_concentration(chunks: list[RetrievedChunk]) -> float:
    """How much the top hit stands out from the rest, in [0, 1].

    0 means every hit scored the same (the query matched a topic); near 1 means
    one hit dominates (the query matched a fact).
    """
    scores = [c.rank_score() for c in chunks if c.rank_score() > 0]
    if len(scores) < 2:
        return 1.0 if scores else 0.0
    best = max(scores)
    if best <= 0:
        return 0.0
    rest = sorted(scores, reverse=True)[1:]
    mean_rest = sum(rest) / len(rest)
    return max(0.0, min(1.0, (best - mean_rest) / best))


def assess(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    threshold: float,
    min_coverage: float = 0.34,
) -> Sufficiency:
    """Judge whether `chunks` can support an answer to `query`.

    `threshold` is on the reranker's 0-4 scale and is the primary gate;
    coverage is a secondary veto that catches the case where the reranker
    liked a passage that is topically adjacent but does not mention what was
    actually asked about.
    """
    if not chunks:
        return Sufficiency(
            score=0.0,
            sufficient=False,
            reason="no evidence was retrieved for this question",
        )

    top = max(
        (c.reranker_score for c in chunks if c.reranker_score is not None),
        default=None,
    )
    coverage, missing = term_coverage(query, chunks)
    concentration = score_concentration(chunks)

    if top is None:
        # No semantic ranking available (it failed, or was disabled for a
        # baseline run). Fall back to lexical coverage alone, which is weaker
        # -- so say so in the reason rather than pretending to the same
        # confidence.
        sufficient = coverage >= min_coverage
        return Sufficiency(
            score=coverage,
            sufficient=sufficient,
            coverage=coverage,
            concentration=concentration,
            missing_terms=missing,
            reason=(
                "judged on term coverage only; no semantic ranking scores were "
                "available"
            ),
        )

    normalized_top = top / 4.0
    sufficient = top >= threshold and coverage >= min_coverage

    if not sufficient:
        if top < threshold:
            reason = (
                f"best evidence scored {top:.2f} against a required {threshold:.2f}; "
                "nothing retrieved is a close enough match to answer from"
            )
        else:
            shown = ", ".join(missing[:5])
            reason = (
                f"the evidence does not mention {shown}, so it does not address "
                "what was asked"
            )
    else:
        reason = f"best evidence scored {top:.2f} with {coverage:.0%} term coverage"

    return Sufficiency(
        score=round(0.6 * normalized_top + 0.25 * coverage + 0.15 * concentration, 4),
        sufficient=sufficient,
        top_reranker=top,
        coverage=coverage,
        concentration=concentration,
        missing_terms=missing,
        reason=reason,
    )
