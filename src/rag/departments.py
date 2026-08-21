"""The department registry: configuration, not code.

Which departments exist is an env setting (`DEPARTMENTS`), and every stage of
the system keys off that one list -- what gets monitored, what gets ingested,
how documents are labelled in the vector store, which nodes exist in the
graph, and what a given caller is allowed to retrieve. Adding a sixth
department is an env edit plus a folder; removing one stops its documents
being ingested and lets the delete reconciliation sweep them out of both
stores. Nothing about a department is hard-coded anywhere else.

`DEPARTMENT_SOURCES` optionally decouples a department's *name* from its
*location*, so a department called "HR" can live under `people-ops/` in the
container without either side having to change.
"""
from __future__ import annotations

from dataclasses import dataclass

from rag.config import Settings, get_settings


@dataclass(frozen=True)
class Department:
    name: str
    # Path segment / blob prefix this department's documents live under.
    source_prefix: str


class DepartmentRegistry:
    """Resolves a source key (a relative path or blob name) to a department.

    Lookup is by the key's leading path segment, matched case-insensitively
    against both department names and their configured source prefixes. A key
    that matches nothing is governed by `unknown_department_policy`:
    "skip" (default) keeps the corpus scoped to exactly the configured
    departments; "ingest" files it under its own segment name so an
    unexpected folder surfaces in the index instead of vanishing silently.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        mapping = self._settings.department_sources
        self._departments = [
            Department(name=name, source_prefix=mapping.get(name, name).strip("/"))
            for name in self._settings.departments
        ]
        # Both the department name and its prefix are accepted as the leading
        # segment, so a corpus laid out either way resolves identically.
        self._by_key: dict[str, Department] = {}
        for dept in self._departments:
            self._by_key[dept.name.lower()] = dept
            self._by_key[dept.source_prefix.lower()] = dept

    # ---- lookup ----

    def all_departments(self) -> list[Department]:
        return list(self._departments)

    def names(self) -> list[str]:
        return [d.name for d in self._departments]

    def source_prefixes(self) -> list[str]:
        return [d.source_prefix for d in self._departments]

    @staticmethod
    def _normalize(source_key: str) -> str:
        return source_key.replace("\\", "/").lstrip("/")

    @classmethod
    def _head_segment(cls, source_key: str) -> str:
        return cls._normalize(source_key).split("/")[0]

    def lookup(self, source_key: str) -> Department | None:
        """The configured department owning `source_key`, or None.

        Matches on the longest configured prefix first, so a multi-segment
        source prefix like `engineering/it` wins over a single-segment
        department that happens to share its first segment.
        """
        key = self._normalize(source_key).lower()
        for candidate, dept in sorted(
            self._by_key.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if key == candidate or key.startswith(f"{candidate}/"):
                return dept
        return None

    def department_for_key(self, source_key: str) -> str:
        """The department label to store for `source_key`.

        Returns the configured department's canonical name when one matches.
        Otherwise returns the raw leading segment -- callers that care about
        the difference should ask `is_ingestable` first rather than
        inspecting the returned string.
        """
        dept = self.lookup(source_key)
        return dept.name if dept else self._head_segment(source_key)

    def is_ingestable(self, source_key: str) -> bool:
        if self.lookup(source_key) is not None:
            return True
        return self._settings.unknown_department_policy == "ingest"

    def is_allowed(self, source_key: str, allowed: list[str] | None) -> bool:
        """Whether a caller scoped to `allowed` departments may see `source_key`.

        Deny-by-default: `None` or an empty scope grants nothing. Access
        control that defaults to "everything" is how the wrong department
        ends up reading HR documents.
        """
        if not allowed:
            return False
        wanted = {a.lower() for a in allowed}
        return self.department_for_key(source_key).lower() in wanted


def get_registry(settings: Settings | None = None) -> DepartmentRegistry:
    """Build a registry from current settings.

    Deliberately not cached: `get_settings()` is, and tests clear that cache
    to swap department configurations between cases.
    """
    return DepartmentRegistry(settings)
