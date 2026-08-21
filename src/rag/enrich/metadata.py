"""Parse the pipe-delimited document header into typed metadata.

Real examples from the corpus:
  Contoso Ltd. | People Operations | Plan Year: 2026 | Version 3.0 | Policy Owner: ...
  Contoso Ltd. | Sales Operations | Effective: January 1, 2026 | Version 2.0 | Supersedes: 2025 Rate Card (v1.0)
"""
from __future__ import annotations

import re
from datetime import date, datetime

from rag.departments import get_registry
from rag.models import DocumentMetadata, ParsedDocument

_DATE_RE = re.compile(r"([A-Z][a-z]+ \d{1,2}, \d{4})")
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_RANGE_SEP = re.compile(r"\s*[–—-]\s*")


def _parse_date(text: str) -> date | None:
    match = _DATE_RE.search(text)
    if match:
        return datetime.strptime(match.group(1), "%B %d, %Y").date()
    year = _YEAR_RE.search(text)
    return date(int(year.group(1)), 1, 1) if year else None


def _department_from_key(source_key: str) -> str:
    """Delegates to the department registry -- see `rag.departments`."""
    return get_registry().department_for_key(source_key)


def extract_metadata(doc: ParsedDocument, source_key: str) -> DocumentMetadata:
    meta = DocumentMetadata(
        doc_id=source_key,
        title=source_key.rsplit("/", 1)[-1].rsplit(".", 1)[0],
        department=_department_from_key(source_key),
    )
    for segment in (s.strip() for s in doc.raw_header.split("|")):
        low = segment.lower()
        if low.startswith("version"):
            meta.version = segment.split(None, 1)[-1].strip()
        elif low.startswith("supersedes:"):
            meta.supersedes = segment.split(":", 1)[1].strip()
        elif low.startswith("policy owner:"):
            meta.owner = segment.split(":", 1)[1].strip()
        elif low.startswith(("effective:", "plan year:")):
            value = segment.split(":", 1)[1].strip()
            parts = _RANGE_SEP.split(value)
            meta.effective_from = _parse_date(parts[0])
            if len(parts) > 1:
                meta.effective_to = _parse_date(parts[1])
    return meta


def link_versions(metas: list[DocumentMetadata]) -> list[DocumentMetadata]:
    """Resolve `Supersedes:` declarations into doc-to-doc links.

    Superseded documents remain indexed and retrievable — `is_current` is a
    ranking signal, because the 2025 rate card still governs active 2025
    contracts. It is never used as a hard exclusion filter.
    """
    by_year: dict[str, DocumentMetadata] = {}
    for m in metas:
        year = _YEAR_RE.search(m.doc_id) or _YEAR_RE.search(m.title)
        if year:
            by_year[f"{m.department}:{year.group(1)}"] = m

    for m in metas:
        if not m.supersedes:
            continue
        year = _YEAR_RE.search(m.supersedes)
        if not year:
            continue
        target = by_year.get(f"{m.department}:{year.group(1)}")
        if target and target.doc_id != m.doc_id:
            target.superseded_by = m.doc_id
            target.is_current = False
    return metas
