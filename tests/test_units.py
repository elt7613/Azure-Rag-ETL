"""Section-level extraction units and token-budget packing."""
from __future__ import annotations

import pytest

from rag.extraction.units import (
    UnitPack,
    build_units,
    count_tokens,
    pack_units,
)
from rag.models import Chunk, DocumentMetadata

META = DocumentMetadata(doc_id="HR/LeavePolicy.pdf", title="Leave Policy",
                        department="HR", version="2.0")


def _chunk(path, text, index, content_type="prose", page=1, number=""):
    return Chunk(doc_id=META.doc_id, section_path=path, display_text=text,
                 embed_text=text, content_type=content_type, page=page,
                 chunk_index=index, section_number=number)


def test_a_section_spanning_five_chunks_is_one_unit_carrying_five_chunk_ids():
    path = ["2 Types of Leave", "2.1 Annual / Paid Time Off (PTO)"]
    chunks = [_chunk(path, f"Accrual paragraph {i}.", i) for i in range(5)]
    units = build_units(chunks, META)
    assert len(units) == 1
    assert len(units[0].chunk_ids) == 5
    assert units[0].chunk_ids == [c.compute_id() for c in chunks]
    # The unit text is the whole thought, not a window into it.
    for i in range(5):
        assert f"Accrual paragraph {i}." in units[0].text


def test_distinct_sections_become_distinct_units_with_stable_ids():
    chunks = [
        _chunk(["1 Purpose"], "Purpose text.", 0),
        _chunk(["2 Sick Leave"], "Sick leave text.", 1),
    ]
    units = build_units(chunks, META)
    assert len(units) == 2
    assert units[0].unit_id != units[1].unit_id
    assert [u.unit_id for u in units] == [u.unit_id for u in build_units(chunks, META)]


def test_tables_stay_separate_from_the_prose_of_the_same_section():
    """The deterministic tabular path owns tables; mixing them into a prose
    unit would send a spreadsheet to the LLM at prose prices."""
    path = ["2 Types of Leave", "2.1 PTO"]
    chunks = [
        _chunk(path, "| Years | Accrual |\n| --- | --- |\n| 0-2 | 15 days |", 0, "table"),
        _chunk(path, "Full-time employees accrue PTO bi-weekly.", 1),
    ]
    units = build_units(chunks, META)
    assert sorted(u.content_type for u in units) == ["prose", "table"]


def test_each_table_is_its_own_unit():
    path = ["10 Contact"]
    chunks = [
        _chunk(path, "| A |\n| --- |\n| 1 |", 0, "table"),
        _chunk(path, "| B |\n| --- |\n| 2 |", 1, "table"),
    ]
    assert len(build_units(chunks, META)) == 2


def test_units_carry_document_provenance():
    units = build_units([_chunk(["1 Purpose"], "Purpose text.", 0)], META)
    assert units[0].doc_id == META.doc_id
    assert units[0].department == "HR"
    assert units[0].section_path == "1 Purpose"


def test_identical_repeated_sections_collapse_to_one_unit():
    """Same text twice in one document = one extraction, both chunks credited."""
    chunks = [
        _chunk(["A"], "Northwind Traders, Inc. Internal Use Only.", 0),
        _chunk(["B"], "Northwind Traders, Inc. Internal Use Only.", 1),
    ]
    units = build_units(chunks, META)
    assert len(units) == 1
    assert len(units[0].chunk_ids) == 2


def test_empty_chunks_produce_no_units():
    assert build_units([], META) == []
    assert build_units([_chunk(["A"], "   ", 0)], META) == []


# ---- packing ----

def _unit(text, index):
    from rag.models import ExtractionUnit
    return ExtractionUnit(unit_id=f"u{index}", doc_id=META.doc_id, department="HR",
                          section_path=f"{index}", text=text, page=1,
                          chunk_ids=[f"c{index}"])


def test_many_tiny_units_are_packed_and_never_exceed_the_budget():
    units = [_unit("Employees may claim up to $600 per year.", i) for i in range(200)]
    budget = 300
    packs = pack_units(units, budget)
    assert all(p.token_count <= budget for p in packs)
    assert len(packs) < len(units)  # packing actually amortised something


def test_packing_preserves_every_unit_exactly_once_and_in_order():
    units = [_unit(f"Unit {i} body text about reimbursement limits.", i) for i in range(50)]
    packs = pack_units(units, 200)
    packed = [u.unit_id for p in packs for u in p.units]
    assert packed == [u.unit_id for u in units]


def test_a_single_oversized_unit_is_emitted_alone_never_merged_or_dropped():
    giant = _unit(" ".join(f"clause {i} of the agreement" for i in range(4000)), 0)
    small = _unit("A short obligation clause.", 1)
    packs = pack_units([giant, small], 500)
    assert [u.unit_id for p in packs for u in p.units] == ["u0", "u1"]
    oversized = [p for p in packs if p.oversized]
    assert len(oversized) == 1
    assert oversized[0].units == [giant]
    # Every pack that is over budget holds exactly one unit -- the guarantee is
    # "never merge past the budget", which a lone unit cannot violate.
    assert all(p.token_count <= 500 or len(p.units) == 1 for p in packs)


def test_a_unit_exactly_at_budget_packs_alone_and_does_not_overflow():
    budget = 200
    filler = _unit(" ".join(["word"] * 400), 0)
    # Trim until the unit's packed cost is exactly the budget.
    words = 400
    while pack_units([_unit(" ".join(["word"] * words), 0)], budget)[0].token_count > budget:
        words -= 1
    exact = _unit(" ".join(["word"] * words), 0)
    packs = pack_units([exact, _unit("tiny", 1)], budget)
    assert packs[0].token_count <= budget
    assert packs[0].units == [exact]
    assert all(p.token_count <= budget for p in packs)


def test_packing_accounts_for_per_unit_framing_overhead():
    """A pack costs more than the sum of its bodies: each unit needs its id
    and delimiters in the prompt, and a budget that ignores that overflows."""
    units = [_unit("short", i) for i in range(10)]
    pack = pack_units(units, 10_000)[0]
    assert pack.token_count > sum(count_tokens(u.text) for u in units)


def test_pack_exposes_unit_ids_for_attribution():
    packs = pack_units([_unit("a body of policy text", i) for i in range(3)], 10_000)
    assert isinstance(packs[0], UnitPack)
    assert packs[0].unit_ids == ["u0", "u1", "u2"]


def test_packing_nothing_is_nothing():
    assert pack_units([], 3000) == []


@pytest.mark.parametrize("budget", [0, -1])
def test_a_nonsense_budget_is_rejected_loudly(budget):
    with pytest.raises(ValueError):
        pack_units([_unit("text", 0)], budget)
