"""Fold a flat block list into a section tree using heading levels."""
from __future__ import annotations

from collections.abc import Iterator

from rag.models import Block, BlockType, ParsedDocument, Section


def _split_heading(text: str) -> tuple[str, str]:
    parts = text.split(None, 1)
    if parts and parts[0].rstrip(".").replace(".", "").isdigit():
        return parts[0].rstrip("."), (parts[1] if len(parts) > 1 else "")
    return "", text


def build_section_tree(doc: ParsedDocument) -> list[Section]:
    roots: list[Section] = []
    stack: list[Section] = []

    def attach(section: Section) -> None:
        while stack and stack[-1].level >= section.level:
            stack.pop()
        (stack[-1].children if stack else roots).append(section)
        stack.append(section)

    for block in doc.blocks:
        if block.type is BlockType.HEADING:
            number, title = _split_heading(block.text)
            attach(Section(number=number, title=title,
                           level=max(block.level, 1), page=block.page))
        else:
            if not stack:
                attach(Section(number="", title="Preamble", level=1, page=block.page))
            stack[-1].blocks.append(block)
    return roots


def iter_leaf_sections(
    sections: list[Section], prefix: list[str] | None = None
) -> Iterator[tuple[list[str], Section]]:
    """Yield (breadcrumb_path, section) for every section holding content."""
    for section in sections:
        path = section.path(prefix)
        if section.blocks:
            yield path, section
        if section.children:
            yield from iter_leaf_sections(section.children, path)
