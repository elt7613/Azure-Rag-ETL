"""Resolving conflicting evidence between document versions.

The scenario: the corpus holds a 2025 rate card and a 2026 one, both say
"Enterprise", and they disagree on the price. Retrieval finds both, and a naive
system answers from whichever ranked higher -- which is a coin flip.

The rule applied here is deliberately *not* "drop the old one". A superseded
document is still authoritative for the period it governed: a contract signed
in 2025 is priced off the 2025 card, and deleting that evidence would make the
system unable to answer a legitimate question. So versioning is handled as
three separate jobs:

1. **Detect** that retrieved chunks disagree because they are different
   versions of the same thing.
2. **Rank** the current version above the superseded one.
3. **Tell the user**, so the answer says which version governs and from when,
   rather than silently picking.

Step 3 is the one that turns a wrong answer into a useful one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from rag.retrieval import RetrievedChunk


@dataclass
class VersionConflict:
    """Two or more retrieved documents that are versions of one another."""

    current_doc_id: str
    superseded_doc_ids: list[str] = field(default_factory=list)
    subject: str = ""

    def describe(self) -> str:
        others = ", ".join(self.superseded_doc_ids)
        return (
            f"{self.current_doc_id} is the current version; "
            f"{others} {'is' if len(self.superseded_doc_ids) == 1 else 'are'} superseded"
        )


def _effective_key(chunk: RetrievedChunk) -> tuple:
    """Sort key ordering evidence from most to least authoritative *today*.

    `is_current` first because it is an explicit editorial statement, then
    effective date, then version string. Sorting on the date alone would be
    fooled by a document with no date; sorting on version alone by two
    unrelated documents that both happen to be "v1.0".
    """
    return (
        chunk.is_current,
        chunk.effective_from or date.min,
        chunk.version,
    )


def group_by_document(chunks: list[RetrievedChunk]) -> dict[str, list[RetrievedChunk]]:
    grouped: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    return grouped


def detect_conflicts(chunks: list[RetrievedChunk]) -> list[VersionConflict]:
    """Find retrieved documents that supersede one another.

    Uses the explicit `superseded_by` link where the ingest pipeline resolved
    one. Falls back to same-department documents whose titles share a stem and
    whose effective dates differ, which catches version pairs the header parser
    could not link -- at the cost of an occasional false pair, which is why the
    explicit link is preferred when present.
    """
    conflicts: list[VersionConflict] = []
    by_doc = group_by_document(chunks)
    docs = {doc_id: group[0] for doc_id, group in by_doc.items()}

    claimed: set[str] = set()
    for doc_id, chunk in docs.items():
        if chunk.superseded_by and chunk.superseded_by in docs:
            successor = chunk.superseded_by
            existing = next(
                (c for c in conflicts if c.current_doc_id == successor), None
            )
            if existing:
                existing.superseded_doc_ids.append(doc_id)
            else:
                conflicts.append(
                    VersionConflict(
                        current_doc_id=successor,
                        superseded_doc_ids=[doc_id],
                        subject=docs[successor].title,
                    )
                )
            claimed.update({doc_id, successor})

    # Heuristic fallback for unlinked version pairs.
    remaining = [d for d in docs if d not in claimed]
    for i, left_id in enumerate(remaining):
        for right_id in remaining[i + 1 :]:
            left, right = docs[left_id], docs[right_id]
            if left.department != right.department:
                continue
            if not _shared_stem(left.title, right.title):
                continue
            if left.effective_from is None or right.effective_from is None:
                continue
            if left.effective_from == right.effective_from:
                continue
            newer, older = (
                (left, right)
                if left.effective_from > right.effective_from
                else (right, left)
            )
            conflicts.append(
                VersionConflict(
                    current_doc_id=newer.doc_id,
                    superseded_doc_ids=[older.doc_id],
                    subject=newer.title,
                )
            )
    return conflicts


def _shared_stem(left: str, right: str) -> bool:
    """Whether two titles look like versions of the same document.

    Strips digits so "Pricing2025" and "Pricing2026" both reduce to "pricing",
    and requires a token of real length so unrelated documents don't pair up on
    a shared "the" or "v".
    """
    def stems(title: str) -> set[str]:
        cleaned = "".join(c if c.isalpha() else " " for c in title).lower()
        return {t for t in cleaned.split() if len(t) >= 4}

    return bool(stems(left) & stems(right))


def prefer_current(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Stable-sort so current-version evidence precedes superseded evidence.

    Deliberately a reordering, not a filter. The superseded chunks stay in the
    context so the answer can say what the old value was and when it applied --
    which is usually what the person asking a version-sensitive question
    actually needs.
    """
    return sorted(chunks, key=_effective_key, reverse=True)


def resolve(
    chunks: list[RetrievedChunk],
) -> tuple[list[RetrievedChunk], list[VersionConflict]]:
    """Order evidence by authority and report any version conflicts found."""
    conflicts = detect_conflicts(chunks)
    if not conflicts:
        return chunks, []

    superseded = {
        doc_id for conflict in conflicts for doc_id in conflict.superseded_doc_ids
    }
    current = [c for c in chunks if c.doc_id not in superseded]
    older = [c for c in chunks if c.doc_id in superseded]
    return [*current, *older], conflicts


def conflict_note(conflicts: list[VersionConflict]) -> str:
    """A sentence for the answer prompt telling the model what to disclose."""
    if not conflicts:
        return ""
    lines = [c.describe() for c in conflicts]
    return (
        "The retrieved evidence includes more than one version of the same "
        "document: " + "; ".join(lines) + ". Answer from the current version, "
        "state which version you used and its effective date, and mention the "
        "superseded value only if it is relevant to the question."
    )
