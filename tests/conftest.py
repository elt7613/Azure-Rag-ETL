"""Shared test fixtures."""
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import pytest

from rag.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "source_data"


def corpus_available() -> bool:
    """Whether the document corpus these tests were written against is present.

    `source_data/` ships with the project, so normally it is. It stops being
    there the moment someone points the pipeline at their own documents and
    removes it, and those tests then *skip* rather than fail — a repository
    that greets a reader with a wall of red tells them the code is broken when
    only the input is different.
    """
    return CORPUS.is_dir() and any(CORPUS.rglob("*.pdf"))


def pytest_collection_modifyitems(config, items):
    """Skip corpus-dependent tests when the corpus is not present.

    Detected by looking for `source_data` in the test module's own source
    rather than by keeping a hand-written list of files, which would go stale
    the first time a test was added.
    """
    if corpus_available():
        return
    reason = pytest.mark.skip(
        reason="source_data/ not present -- these tests assert facts from it"
    )
    needs_corpus: dict[Path, bool] = {}
    for item in items:
        path = Path(item.fspath)
        if path not in needs_corpus:
            try:
                needs_corpus[path] = "source_data" in path.read_text()
            except OSError:  # pragma: no cover - unreadable test file
                needs_corpus[path] = False
        if needs_corpus[path]:
            item.add_marker(reason)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """get_settings() is lru_cached; without this, a test that patches env
    after the cache is warm silently reads stale config."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def azure_configured(*field_names: str) -> bool:
    """True only when every named Settings field holds a real value.

    Azure-gated tests MUST use this rather than os.getenv: credentials live in
    .env, which pydantic-settings reads WITHOUT exporting to os.environ, so an
    os.getenv gate skips even when credentials are correctly configured.
    """
    settings = get_settings()
    for name in field_names:
        value = str(getattr(settings, name, "") or "")
        if not value or "PLACEHOLDER" in value:
            return False
    return True


def neo4j_configured() -> bool:
    """True when a real Neo4j is configured. Same os.getenv caveat as above."""
    return azure_configured("neo4j_uri", "neo4j_user", "neo4j_password", "neo4j_database")


# --------------------------------------------------------------------------
# Which corpus is indexed
#
# Live retrieval tests assert real facts -- "10 days of sick leave", "$109 per
# seat" -- and those facts belong to a corpus, not to the code. `source_data/`
# ships with the project and is what these numbers come from, but the index is
# a remote service: it can just as easily hold somebody else's documents, and
# the tests then fail for a reason that has nothing to do with the code.
#
# So the facts are looked up from the corpus actually in the index, and the
# tests that need them skip when it holds something else. Pointing the suite at
# a different corpus is a second entry in `_CORPORA`.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusFacts:
    """What the live retrieval tests need to know about the indexed corpus.

    Values *and vocabulary*. An earlier version carried only the values, which
    was not enough: a corpus that says "Paid Time Off" where another says
    "annual leave" answers nothing when queried in the other's words, and the
    test fails for a reason that has nothing to do with the code.
    """

    name: str
    probe_doc: str                # a document unique to this corpus
    leave_doc: str
    leave_term: str               # how this corpus refers to accrued leave
    sick_leave_days: str
    learning_limit: str           # a distinctive exact token, e.g. "5,250"
    learning_query: str
    entry_tier: str
    entry_tier_price: str
    top_tier: str
    top_tier_price: str
    superseded_price: str
    pricing_current: str = "sales/Pricing2026.pdf"
    pricing_superseded: str = "sales/Pricing2025.pdf"
    unanswerable: str = "What is the company's severance pay policy?"

    # Filled in from the index and the configuration, not hard-coded -- the two
    # can disagree, and when they do the tests should follow reality.
    departments: tuple[str, ...] = ()
    other_department: str = ""

    @property
    def leave_query(self) -> str:
        return f"how many {self.leave_term} days after 4 years of service"


_CORPORA = (
    CorpusFacts(
        name="source_data", probe_doc="IT/VPNGuide.pdf",
        leave_doc="HR/LeavePolicy.pdf", leave_term="Paid Time Off",
        sick_leave_days="10",
        learning_limit="5,250",
        learning_query="What is the tuition reimbursement limit?",
        entry_tier="Starter", entry_tier_price="32",
        top_tier="Enterprise", top_tier_price="109", superseded_price="99",
    ),
)


@lru_cache
def indexed_corpus() -> "CorpusFacts | None":
    """Identify the corpus in the search index, or None if it holds another.

    Also resolves the department scope from the index *intersected with* the
    configured departments. Those two can disagree — an index left over from a
    different corpus, a DEPARTMENTS list edited without re-ingesting — and a
    test that assumes the configuration wins fails confusingly. Only
    departments that are both configured and present can be retrieved from, so
    that is what the tests use.

    Uses the synchronous SDK deliberately: this runs from a plain fixture,
    before any event loop exists, and reaching for async here buys nothing.
    """
    if not azure_configured("azure_search_endpoint", "azure_search_key",
                            "azure_search_index"):
        return None
    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        settings = get_settings()
        client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index,
            credential=AzureKeyCredential(settings.azure_search_key),
        )
        with client:
            match = None
            for corpus in _CORPORA:
                found = client.search(
                    search_text="*", filter=f"doc_id eq '{corpus.probe_doc}'",
                    select=["doc_id"], top=1,
                )
                if any(True for _ in found):
                    match = corpus
                    break
            if match is None:
                return None

            rows = client.search(search_text="*", select=["department"], top=1000)
            indexed = {r["department"] for r in rows if r.get("department")}
    except Exception:  # pragma: no cover - an unreachable index is a skip, not an error
        return None

    configured = {d.lower(): d for d in get_settings().departments}
    usable = sorted(
        configured[d.lower()] for d in indexed if d.lower() in configured
    )
    if not usable:
        return None
    others = [d for d in usable if d.upper() != "HR"]
    return replace(
        match,
        departments=tuple(usable),
        other_department=others[0] if others else usable[0],
    )


@pytest.fixture(scope="session")
def facts() -> CorpusFacts:
    corpus = indexed_corpus()
    if corpus is None:
        pytest.skip(
            "the search index holds no recognised corpus, or none of its "
            "departments are configured -- ingest source_data/ first: "
            "`cocoindex --app-dir . update rag.etl.app`"
        )
    return corpus
