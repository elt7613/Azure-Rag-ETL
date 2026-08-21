"""Turning a follow-up into a question that can be retrieved on.

The problem this solves, from the assignment's own example:

    User: What is the Enterprise plan cancellation policy?
    User: What about Standard?
    User: Is there any exception?

"What about Standard?" retrieves nothing useful on its own, and pasting the
whole conversation into the retriever is worse -- the earlier turns drag in
their own vocabulary and the search drifts back toward the first question.

The approach here is the one the literature supports: **condense the history
into a standalone query, send only that to the retrievers, and keep the raw
history for generation.** Two refinements matter in practice:

1. The retrieval string is the rewrite **concatenated with the raw follow-up**,
   not the rewrite alone. The rewrite supplies the missing subject; the raw
   text supplies the user's own wording, which the keyword side of hybrid
   search needs.
2. The rewrite is conservative. Aggressive expansion -- piling on synonyms and
   hypothesised phrasing -- amplifies rare terms and pulls retrieval off
   target. When the question already stands alone, the right rewrite is no
   rewrite, and the model is told so explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from rag.agents import build_agent, is_content_filter_error, record_usage

_SYSTEM = """\
You rewrite the latest user message into a single self-contained search query.

Rules:
- Resolve pronouns and elliptical references ("it", "that one", "what about X?") \
using the conversation history.
- Carry forward the subject of the conversation ONLY when the latest message \
depends on it. "What about Standard?" following a question about the Enterprise \
plan becomes "Standard plan cancellation policy".
- If the latest message already stands on its own, return it unchanged and set \
rewritten=false. Rewriting a self-contained question is a mistake, not a \
no-op: it drifts the search.
- Do NOT add synonyms, hypothesised phrasings, or extra keywords. You are \
resolving references, not expanding the query.
- If the latest message changes the subject entirely, ignore the history.
- Keep the user's own terminology. Do not translate their words into what you \
think the documents call it.

Return the standalone query and whether you changed anything."""


class CondensedQuery(BaseModel):
    standalone_query: str = Field(
        description="The latest message rewritten to stand on its own."
    )
    rewritten: bool = Field(
        description="False when the original message already stood alone."
    )
    topic_shift: bool = Field(
        default=False,
        description="True when the latest message abandons the previous subject.",
    )


@dataclass
class Turn:
    role: str
    content: str


def _format_history(history: list[Turn], *, max_turns: int = 6) -> str:
    """Render recent history for the condenser only.

    Capped because condensation needs the immediate referent, not the whole
    session: an old turn cannot disambiguate "what about Standard?" and can
    easily mislead it.
    """
    recent = history[-max_turns:]
    return "\n".join(f"{t.role}: {t.content}" for t in recent)


def retrieval_query(condensed: CondensedQuery, original: str) -> str:
    """The string actually sent to the retrievers.

    Concatenating the rewrite with the raw message beats using either alone:
    the rewrite carries the resolved subject for the vector side, the raw
    message carries the user's literal tokens for the keyword side.
    """
    if not condensed.rewritten:
        return original
    if original.strip().lower() in condensed.standalone_query.strip().lower():
        return condensed.standalone_query
    return f"{condensed.standalone_query} {original}".strip()


_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent(CondensedQuery, _SYSTEM)
    return _agent


async def condense(message: str, history: list[Turn] | None = None) -> CondensedQuery:
    """Resolve `message` against `history` into a standalone query.

    With no history there is nothing to resolve, so this short-circuits without
    an LLM call -- the first turn of every conversation, and every single-shot
    API call, costs nothing here.
    """
    if not history:
        return CondensedQuery(standalone_query=message, rewritten=False)

    try:
        result = await _get_agent().run(
            f"Conversation so far:\n{_format_history(history)}\n\n"
            f"Latest user message: {message}"
        )
    except Exception as exc:
        if is_content_filter_error(exc):
            # Condensation is an optimisation, not a requirement. If the
            # platform will not process the history, retrieve on the raw
            # message rather than failing the turn.
            return CondensedQuery(standalone_query=message, rewritten=False)
        raise
    record_usage("condense", result)
    output = result.output
    # A rewrite that lost the question entirely is worse than no rewrite.
    if not output.standalone_query.strip():
        return CondensedQuery(standalone_query=message, rewritten=False)
    return output
