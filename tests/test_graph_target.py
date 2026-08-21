"""Tests for the Neo4j document-structure graph target.

Node/edge shapes and `build_graph_elements` are pure, unit-testable without a
live database. `resolve_supersedes` (the fragile-match fix) is also pure and
is exercised against the real Pricing2025/Pricing2026 pair here. The
`Neo4jGraphWriter` itself is exercised end-to-end in
`tests/test_graph_live.py`, gated on live Neo4j config being present.
"""
from rag.models import Chunk, DocumentMetadata
from rag.targets.graph import DocumentNode, build_graph_elements, resolve_supersedes

META = DocumentMetadata(doc_id="sales/Pricing2026.pdf", title="Pricing2026",
                        department="sales", version="1.0",
                        supersedes="2025 Rate Card (v1.4)",
                        superseded_by="", is_current=True)


def _chunks():
    return [
        Chunk(doc_id=META.doc_id, section_path=["3 Tiers"], display_text="a",
              embed_text="a", content_type="table", page=1, chunk_index=0,
              section_number="3"),
        Chunk(doc_id=META.doc_id, section_path=["3 Tiers"], display_text="b",
              embed_text="b", content_type="prose", page=1, chunk_index=1,
              section_number="3"),
    ]


def test_builds_document_section_and_chunk_nodes():
    g = build_graph_elements(_chunks(), META)
    assert g.document.doc_id == META.doc_id
    assert len(g.sections) == 1
    assert len(g.chunks) == 2


def test_contains_and_next_edges_are_created():
    """Both chunks share one section, so CONTAINS is 1 Document->Section edge
    plus 2 Section->Chunk edges (one per chunk) = 3, not 2 -- this matches
    the real-corpus measurement (sections + chunks) used to sanity-check the
    live counts in tests/test_graph_live.py."""
    g = build_graph_elements(_chunks(), META)
    assert len(g.contains) == 3
    assert len(g.next_edges) == 1
    assert g.next_edges[0].from_id == g.chunks[0].chunk_id


def test_supersedes_edge_only_when_declared():
    g = build_graph_elements(_chunks(), META)
    assert g.supersedes and g.supersedes[0].to_hint == "2025 Rate Card (v1.4)"
    plain = DocumentMetadata(doc_id="d", title="t", department="HR")
    assert build_graph_elements(_chunks(), plain).supersedes == []


# ---- resolve_supersedes: the fragile-match fix ----

SOURCE = DocumentNode(doc_id="sales/Pricing2026.pdf", title="Pricing2026",
                      department="sales", version="1.0", is_current=True)
TARGET = DocumentNode(doc_id="sales/Pricing2025.pdf", title="Pricing2025",
                      department="sales", version="1.4", is_current=True)


def test_resolve_supersedes_matches_the_real_pricing_pair():
    match = resolve_supersedes(SOURCE, "2025 Rate Card (v1.4)", [SOURCE, TARGET])
    assert match is not None
    assert match.doc_id == "sales/Pricing2025.pdf"


def test_resolve_supersedes_rejects_version_substring_in_wrong_department():
    """A candidate whose version textually appears inside the hint must NOT
    match when it's in a different department -- this is exactly the false
    positive the naive `toLower(hint) CONTAINS toLower(b.version)` produces."""
    wrong_dept = DocumentNode(doc_id="HR/Unrelated.pdf", title="Unrelated",
                              department="HR", version="1.4", is_current=True)
    match = resolve_supersedes(SOURCE, "2025 Rate Card (v1.4)", [SOURCE, wrong_dept])
    assert match is None


def test_resolve_supersedes_rejects_bare_version_substring_without_year_or_title_overlap():
    """A same-department candidate whose version is a bare substring of the
    hint (e.g. "1.0" inside "v1.0") must not match without a year AND a
    title-token overlap backing it up."""
    decoy = DocumentNode(doc_id="sales/Unrelated.pdf", title="Unrelated",
                         department="sales", version="1.0", is_current=True)
    # Hint deliberately contains the decoy's version as a substring ("v1.0")
    # but no shared year, and no shared title token with SOURCE ("Pricing").
    match = resolve_supersedes(SOURCE, "Old Notes (v1.0)", [SOURCE, decoy])
    assert match is None


def test_resolve_supersedes_requires_title_token_overlap_even_with_matching_year():
    """A same-department, same-year candidate with no title-token overlap
    with the source document must not match."""
    same_year_diff_series = DocumentNode(doc_id="sales/Roadmap2025.pdf",
                                         title="Roadmap2025", department="sales",
                                         version="1.0", is_current=True)
    match = resolve_supersedes(SOURCE, "2025 Rate Card (v1.4)",
                               [SOURCE, same_year_diff_series])
    assert match is None


def test_resolve_supersedes_returns_none_without_year_in_hint():
    match = resolve_supersedes(SOURCE, "the prior rate card", [SOURCE, TARGET])
    assert match is None
