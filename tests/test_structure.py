from rag.enrich.structure import build_section_tree, iter_leaf_sections
from rag.models import Block, BlockType, ParsedDocument


def _doc(blocks):
    return ParsedDocument(doc_id="d", file_format="pdf", blocks=blocks)


def test_nests_subsections_under_parents():
    blocks = [
        Block(type=BlockType.HEADING, text="2 Types of Leave", page=1, level=1),
        Block(type=BlockType.HEADING, text="2.1 Annual PTO", page=1, level=2),
        Block(type=BlockType.PARAGRAPH, text="20 days per year.", page=1),
    ]
    tree = build_section_tree(_doc(blocks))
    assert len(tree) == 1
    assert tree[0].number == "2"
    assert tree[0].children[0].number == "2.1"


def test_leaf_paths_are_breadcrumbs():
    blocks = [
        Block(type=BlockType.HEADING, text="2 Types of Leave", page=1, level=1),
        Block(type=BlockType.HEADING, text="2.1 Annual PTO", page=1, level=2),
        Block(type=BlockType.PARAGRAPH, text="20 days.", page=1),
    ]
    paths = [p for p, _ in iter_leaf_sections(build_section_tree(_doc(blocks)))]
    assert paths == [["2 Types of Leave", "2.1 Annual PTO"]]


def test_content_before_first_heading_becomes_preamble():
    blocks = [Block(type=BlockType.PARAGRAPH, text="Intro text.", page=1)]
    tree = build_section_tree(_doc(blocks))
    assert tree[0].title == "Preamble"
    assert tree[0].blocks[0].text == "Intro text."
