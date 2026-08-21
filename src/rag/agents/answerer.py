"""Generating a grounded, cited answer from retrieved evidence.

Three prompt decisions carry most of the weight here:

**Citations are numeric markers, not prose.** The model cites `[3]`, where 3 is
the position of a passage in the evidence list it was given. That makes
citation checking mechanical -- a marker either points at a passage that was
actually supplied or it does not -- instead of a fuzzy string match against a
document title the model may have paraphrased. Free-form citations are the
mechanism behind "a completely wrong answer with a valid-looking citation":
they look right precisely because nothing can check them.

**Not answering is an allowed, named outcome.** The model is given an explicit
`answered=false` path with a reason. A prompt that only describes how to answer
leaves refusal as a deviation, and models do not deviate readily -- they
improvise. The sufficiency gate upstream already refuses on weak evidence; this
is the second, independent chance to notice that the evidence does not actually
say what the question asked.

**Version conflicts are surfaced, not resolved silently.** When the evidence
contains two versions of the same document, the instruction is to answer from
the current one *and say so*, because a user asking about pricing usually needs
to know which rate card applies to them.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from rag.agents import build_agent, is_content_filter_error, record_usage
from rag.retrieval import RetrievedChunk

_SYSTEM = """\
You answer questions about an internal company knowledge base, using ONLY the \
numbered passages you are given.

Grounding rules -- these are absolute:
- Every factual statement must come from a supplied passage. If the passages \
do not contain it, you do not know it.
- Cite with the passage number in square brackets INSIDE the answer text, \
immediately after the statement it supports: "Full-time employees accrue 20 \
days of PTO after three years [2]."
- The bracketed markers must appear in the `answer` text itself. Listing the \
numbers in the `cited` field is NOT a citation -- an answer whose text contains \
no [n] markers is treated as uncited and rejected, however correct it is.
- Every sentence that states a figure, date, threshold or name carries its own \
marker. Do not cite once at the end and leave the rest bare.
- Cite every passage you actually used, and cite nothing else. Do not cite a \
passage you did not draw from to make the answer look better sourced.
- Never use outside knowledge, even when you are confident it is right. A \
plausible fact that is not in the passages is exactly the failure this system \
exists to prevent.
- Never invent a figure, date, threshold or name. Quote them as written.

If the passages do not answer the question:
- Set answered=false and explain in one sentence what is missing.
- Do not offer a partial answer assembled from adjacent facts. "The leave \
policy does not state a severance entitlement" is a good answer; guessing from \
the notice-period rules is not.

If the passages contain more than one version of the same document:
- Answer from the current version, name it, and give its effective date.
- Mention the superseded value only when the question needs it (for example a \
question about a contract signed while the older version applied).

Style:
- Answer the question directly in the first sentence. No preamble.
- Use the terminology the documents use.
- Prefer a short list or table when the answer is a set of values.
- Be complete but do not pad. If a table answers the question, reproduce the \
relevant rows rather than describing them.
"""


class GroundedAnswer(BaseModel):
    answered: bool = Field(
        description="False when the passages do not support an answer."
    )
    answer: str = Field(
        description="The answer text with [n] citation markers, or the reason "
        "nothing could be answered."
    )
    cited: list[int] = Field(
        default_factory=list,
        description="Passage numbers actually used, 1-based.",
    )
    missing: str = Field(
        default="",
        description="What the passages lacked, when answered=false.",
    )


def format_evidence(chunks: list[RetrievedChunk]) -> str:
    """Render passages for the prompt, numbered from 1.

    Each passage carries its document, section and version inline so the model
    can reason about currency and attribution without a second lookup -- and so
    a reader of the trace can see exactly what the model saw.
    """
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        header = f"[{index}] {chunk.doc_id}"
        if chunk.section_path:
            header += f" — {chunk.section_path}"
        if chunk.page:
            header += f" (page {chunk.page})"
        if chunk.version:
            header += f" [version {chunk.version}]"
        if not chunk.is_current:
            header += " [SUPERSEDED"
            if chunk.superseded_by:
                header += f" by {chunk.superseded_by}"
            header += "]"
        if chunk.effective_from:
            header += f" [effective {chunk.effective_from.isoformat()}]"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n".join(parts)


def _format_history(history, max_turns: int = 6) -> str:
    if not history:
        return ""
    recent = history[-max_turns:]
    rendered = "\n".join(f"{t.role}: {t.content}" for t in recent)
    return f"Conversation so far (context only — do not retrieve from it):\n{rendered}\n\n"


_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent(GroundedAnswer, _SYSTEM)
    return _agent


async def answer(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    history=None,
    conflict_note: str = "",
) -> GroundedAnswer:
    """Answer `query` from `chunks`, or decline with a reason."""
    if not chunks:
        return GroundedAnswer(
            answered=False,
            answer="I don't have anything in the knowledge base that addresses this.",
            missing="no passages were retrieved",
        )

    prompt = (
        f"{_format_history(history)}"
        f"{conflict_note + chr(10) + chr(10) if conflict_note else ''}"
        f"Passages:\n{format_evidence(chunks)}\n\n"
        f"Question: {query}"
    )
    try:
        result = await _get_agent().run(prompt)
    except Exception as exc:
        if is_content_filter_error(exc):
            return GroundedAnswer(
                answered=False,
                answer=(
                    "I can't answer that -- the request was blocked by the "
                    "content safety filter."
                ),
                missing="blocked by the platform content filter",
            )
        raise
    record_usage("answer", result)
    output = result.output
    output.answer = _unescape_newlines(output.answer)

    # Drop citation numbers that point at passages the model was never given.
    # A marker outside the supplied range cannot be checked and must not reach
    # the user as if it were a source.
    output.cited = sorted({n for n in output.cited if 1 <= n <= len(chunks)})
    return output


def _unescape_newlines(text: str) -> str:
    r"""Repair literal backslash-n sequences in structured output.

    Generating JSON under a strict schema occasionally leaves the escape in the
    decoded string, so the answer reaches the user containing a backslash
    followed by 'n' where a line break belongs. Only the newline and tab
    escapes are repaired -- a blanket `unicode_escape` decode would corrupt any
    Windows path or regex the documents legitimately contain.
    """
    if "\n" in text or "\\n" not in text:
        # Over-escaping is all-or-nothing: a response that contains real line
        # breaks was decoded correctly, and any backslash-n left in it belongs
        # to the content -- a Windows path like C:\Users\name, or a regex.
        # Repairing those would corrupt them.
        return text
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def resolve_citations(
    answer_output: GroundedAnswer, chunks: list[RetrievedChunk]
) -> list[dict]:
    """Turn cited passage numbers into user-facing source references."""
    resolved: list[dict] = []
    for number in answer_output.cited:
        chunk = chunks[number - 1]
        resolved.append(
            {
                "marker": number,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "department": chunk.department,
                "section": chunk.section_path,
                "page": chunk.page,
                "version": chunk.version,
                "is_current": chunk.is_current,
                "chunk_id": chunk.chunk_id,
                "citation": chunk.citation(),
            }
        )
    return resolved
