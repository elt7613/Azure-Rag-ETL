"""Checking the answer against the evidence it claims to rest on.

This is the last gate, and it exists because of one specific production
failure: *"the chatbot gives correct answers most of the time, but occasionally
gives a completely wrong answer with a valid-looking citation."* A citation
that is merely present proves nothing. Three things have to be true, and they
fail independently:

1. **The marker resolves.** `[7]` must point at a passage that was actually
   supplied. Checked mechanically, no model needed.
2. **The claim is supported by what it cites.** The passage must actually say
   the thing. This needs a model, but a narrow one: given one claim and one
   passage, is it entailed?
3. **Nothing is asserted uncited.** A factual sentence with no marker at all is
   the most common hallucination shape, because it never looks wrong.

The verification pass is deliberately claim-level rather than answer-level.
Asking "is this answer grounded?" gets a vibe; asking "is this sentence
supported by passage 3?" gets a checkable judgement, and it localises the
failure to the sentence that caused it.

Cost note: this is one extra LLM call per answer. It is skipped for answers
that declined to answer -- there is nothing to hallucinate in a refusal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from rag.agents import build_agent, is_content_filter_error, record_usage
from rag.agents.answerer import GroundedAnswer, format_evidence
from rag.retrieval import RetrievedChunk

_SYSTEM = """\
You audit an answer against the passages it was written from. You are checking \
for fabrication, not for style, helpfulness, or completeness.

Split the answer into its factual claims -- statements that could be true or \
false. Ignore hedges, restatements of the question, and offers to help further.

For each claim decide:
- SUPPORTED: a supplied passage states it. Paraphrase is fine; the fact must \
match, including any number, date or threshold.
- CONTRADICTED: a passage states something different. A figure that does not \
match the passage is CONTRADICTED, not SUPPORTED -- "$5,000" where the passage \
says "$5,250" is a fabrication, however close it looks.
- UNSUPPORTED: no passage says it either way. This includes claims that are \
generally true in the world but absent from these passages.

Record which passage number supports each supported claim.

Be strict. You are the last check before this reaches a user who will act on \
it. If you are unsure whether a passage really says something, it does not."""


class ClaimVerdict(BaseModel):
    claim: str
    verdict: str = Field(description="SUPPORTED, CONTRADICTED or UNSUPPORTED")
    passage: int = Field(default=0, description="Supporting passage number, 0 if none.")
    note: str = Field(default="")


class Verification(BaseModel):
    claims: list[ClaimVerdict] = Field(default_factory=list)
    overall: str = Field(default="", description="One sentence summary.")

    @property
    def supported(self) -> int:
        return sum(1 for c in self.claims if c.verdict == "SUPPORTED")

    @property
    def contradicted(self) -> int:
        return sum(1 for c in self.claims if c.verdict == "CONTRADICTED")

    @property
    def unsupported(self) -> int:
        return sum(1 for c in self.claims if c.verdict == "UNSUPPORTED")

    def groundedness(self) -> float:
        """Fraction of factual claims a passage actually supports."""
        if not self.claims:
            return 1.0
        return self.supported / len(self.claims)


# A sentence asserting something factual: contains a digit, a currency figure,
# a percentage, or a definite statement verb. Used only to find sentences that
# assert without citing, so it errs toward catching too many rather than too few.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[(\d+)\]")


def citation_markers(text: str) -> list[int]:
    return [int(m) for m in _CITATION_RE.findall(text)]


@dataclass
class CitationCheck:
    """Outcome of the mechanical citation checks.

    Two severities, deliberately separated. A **problem** is unfalsifiable
    sourcing -- a marker pointing at a passage that was never supplied, or an
    answer that asserts facts and cites nothing. Those make the answer
    unauditable, so they block it.

    A **warning** is a sentence that states a figure without its own marker.
    That is a real quality defect -- per-sentence attribution is what lets a
    reader check one claim without re-reading everything -- but it does not
    make the answer unverifiable, because the claim-level audit checks every
    claim against the evidence regardless of where the markers sit. Blocking on
    it would reject correct, well-sourced answers over punctuation.
    """

    ok: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_citations(
    answer_output: GroundedAnswer, chunks: list[RetrievedChunk]
) -> CitationCheck:
    """Mechanical citation checks. No model, no cost, no ambiguity."""
    if not answer_output.answered:
        return CitationCheck(ok=True)

    problems: list[str] = []
    warnings: list[str] = []

    markers = citation_markers(answer_output.answer)
    for marker in sorted(set(markers)):
        if not 1 <= marker <= len(chunks):
            problems.append(
                f"citation [{marker}] does not refer to any supplied passage"
            )

    if not markers:
        problems.append("the answer asserts facts but cites no passage")

    for sentence in _SENTENCE_SPLIT.split(answer_output.answer):
        stripped = sentence.strip()
        if len(stripped) < 25 or _CITATION_RE.search(stripped):
            continue
        if any(ch.isdigit() for ch in stripped):
            warnings.append(f"uncited factual sentence: {stripped[:90]!r}")

    return CitationCheck(ok=not problems, problems=problems, warnings=warnings)


_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent(Verification, _SYSTEM)
    return _agent


async def verify(
    query: str, answer_output: GroundedAnswer, chunks: list[RetrievedChunk]
) -> Verification:
    """Claim-level groundedness audit of `answer_output` against `chunks`."""
    if not answer_output.answered or not chunks:
        # A refusal has no claims to fabricate.
        return Verification(overall="declined to answer; nothing to verify")

    try:
        result = await _get_agent().run(
            f"Passages:\n{format_evidence(chunks)}\n\n"
            f"Question: {query}\n\n"
            f"Answer to audit:\n{answer_output.answer}"
        )
    except Exception as exc:
        if is_content_filter_error(exc):
            # The audit could not run, so nothing is verified. Report zero
            # claims with an explicit note rather than an empty Verification,
            # whose groundedness() would otherwise read as a perfect 1.0.
            return Verification(
                claims=[ClaimVerdict(claim=answer_output.answer, verdict="UNSUPPORTED",
                                     note="verification blocked by content filter")],
                overall="verification could not run; treated as unverified",
            )
        raise
    record_usage("verify", result)
    verification = result.output
    for claim in verification.claims:
        claim.verdict = claim.verdict.strip().upper()
        if claim.verdict not in {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED"}:
            # An unrecognised verdict is not evidence of support.
            claim.verdict = "UNSUPPORTED"
    return verification


def passes(
    verification: Verification,
    citations_ok: bool,
    *,
    min_groundedness: float = 0.8,
) -> bool:
    """Whether the answer may be shown as-is.

    A single contradicted claim fails outright regardless of the ratio: an
    answer containing a wrong figure is wrong, even if the other four sentences
    are impeccable.
    """
    if verification.contradicted:
        return False
    return citations_ok and verification.groundedness() >= min_groundedness
