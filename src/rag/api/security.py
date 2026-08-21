"""Who is allowed to read what.

The requirement from the brief is blunt: *HR documents must never be retrieved
for Engineering users.* Two rules follow, and both are about failure direction.

**Deny by default.** A request that does not establish a scope reads nothing.
The tempting alternative -- treat "no scope" as "everything", so the demo is
easy -- is precisely the bug: it works perfectly until the day an
unauthenticated path reaches the endpoint, and then it hands over HR.

**Enforce at the query, not on the results.** The scope resolved here becomes
an OData filter inside the Azure AI Search request and a parameter inside every
Cypher query. Filtering after retrieval means the content already crossed into
a process that was not entitled to it -- it will show up in a log, a trace, or
a cached embedding.

Identity here is a header, because this service sits behind a gateway that
does the authenticating. In a real deployment `X-User-Departments` is set by
that gateway from the caller's Entra ID group membership and is not
client-settable; the shape of the check does not change.
"""
from __future__ import annotations

import logging

from fastapi import Header, HTTPException

from rag.config import get_settings
from rag.departments import get_registry

logger = logging.getLogger(__name__)

DEPARTMENT_HEADER = "X-User-Departments"


class Principal:
    """The caller, reduced to what retrieval needs: an id and a scope."""

    def __init__(self, user_id: str, departments: list[str]) -> None:
        self.user_id = user_id
        self.departments = departments

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Principal({self.user_id!r}, {self.departments!r})"


def _known(requested: list[str]) -> list[str]:
    """Keep only departments this deployment actually has.

    An unknown department name is dropped rather than passed through: it can
    never match a document, and letting it through would make a typo look like
    a permission problem instead of a configuration one.
    """
    registry = get_registry()
    valid = {d.lower(): d for d in registry.names()}
    kept, unknown = [], []
    for name in requested:
        canonical = valid.get(name.strip().lower())
        (kept if canonical else unknown).append(canonical or name)
    if unknown:
        logger.warning("ignoring unknown departments in request scope: %s", unknown)
    return kept


def resolve_scope(
    requested: list[str] | None, header_value: str | None
) -> list[str]:
    """The departments this request may retrieve from.

    Precedence: the explicit body field, then the gateway header, then the
    configured default (empty unless a deployment opts in). A caller can only
    ever *narrow* what the header grants, never widen it -- asking for a
    department the header did not carry gets nothing back.
    """
    settings = get_settings()
    granted = _known([d for d in (header_value or "").split(",") if d.strip()])
    if not granted:
        granted = _known(list(settings.api_default_departments))

    if requested is None:
        return granted

    asked = _known(requested)
    if not granted:
        # Nothing was granted, so nothing can be narrowed to. Honouring the
        # body field here would let any client grant itself access.
        return []
    allowed = {d.lower() for d in granted}
    return [d for d in asked if d.lower() in allowed]


async def principal_from_headers(
    x_user_id: str | None = Header(default=None),
    x_user_departments: str | None = Header(default=None),
) -> Principal:
    return Principal(x_user_id or "anonymous", _known(
        [d for d in (x_user_departments or "").split(",") if d.strip()]
    ))


def require_scope(departments: list[str]) -> None:
    """Reject a request that would retrieve nothing, with a reason.

    Returning an empty answer would be safe but indistinguishable from "the
    corpus has nothing on that", which sends the caller debugging the wrong
    thing.
    """
    if not departments:
        raise HTTPException(
            status_code=403,
            detail=(
                f"No department scope for this request. Supply the "
                f"{DEPARTMENT_HEADER} header (set by the auth gateway), pass "
                f"`departments` in the body within what that header grants, or "
                f"set API_DEFAULT_DEPARTMENTS for a single-tenant deployment."
            ),
        )
