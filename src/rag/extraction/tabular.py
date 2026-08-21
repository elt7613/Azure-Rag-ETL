"""Deterministic extraction from tables -- the path triage routes tables to.

The project owner's requirement is explicit: spreadsheets and detected tables
never go through the LLM. A rate table is already structured data; an LLM
reading it can transpose a digit or round a price, and a parser cannot. So
this module does with regexes and dict lookups what `extraction.llm` does
with a model call for prose -- turn a unit into entities and relations -- and
it does so for free, exactly, and with `deterministic=True` on every edge it
produces.

**Shape of a table, per the corpus.** Every table this module has been
measured against (`sales/Discounts.xlsx`, `finance/ExpensePolicy.pdf`,
`finance/TravelPolicy.docx`, `sales/Pricing2026.pdf`) follows the same
convention: the first row is the header, the first column of each data row
names the row, and the remaining columns are that row's attributes. A header
row supplies the attribute names; each data row becomes an `Entity` whose
type is inferred from the section/sheet title (the table's subject) against
the closed ontology; the row entity gets a `HAS_VALUE` relation to each of its
typed cell values and one `APPLIES_TO` relation back to the subject entity.

**Typed cell values, not just strings.** "$61.75" is a currency; "15%" and
the bare fraction "0.15" (in a column headed "Discount %") are both the same
percentage; "5-24 seats" is a range; "20 days" is a quantity with a unit. The
raw string survives regardless -- `evidence_span` carries it verbatim so an
answer can quote "$61.75" exactly as it was written, not a rounded float.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag.extraction.ontology import (
    coerce_entity_type,
    is_valid_entity_type,
    is_valid_relation_type,
)
from rag.models import Entity, ExtractionUnit, Relation

# The only two predicates this module ever emits. Asserted once at import
# time rather than checked per call: a typo here would silently write edges
# the rest of the system cannot recognise as ontology-conformant, and that is
# cheaper to catch on import than in a 5M-document run.
assert is_valid_relation_type("APPLIES_TO")
assert is_valid_relation_type("HAS_VALUE")

# A row/subject entity earns a more specific type when the section title
# names one (see `_infer_entity_type`). When it does not -- "Approval
# Matrix", "Contact" -- the row still states a measured fact about something,
# which is what `Metric` is for in this ontology; it is the closed
# vocabulary's least-specific member, not a guess at a better one.
_DEFAULT_ENTITY_TYPE = "Metric"
assert is_valid_entity_type(_DEFAULT_ENTITY_TYPE)


# --------------------------------------------------------------------------
# Typed cell values
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellValue:
    """What a table cell says, split into a machine type and the exact text.

    `raw` is always kept: a citation needs to quote "$61.75", not `round(v,
    2)` of a float that was never guaranteed to reconstruct it.
    """
    raw: str
    kind: str  # "currency" | "percentage" | "range" | "quantity" | "text"
    value: float | None = None
    low: float | None = None
    high: float | None = None
    unit: str | None = None


# A relation's object_type must itself be a closed-ontology entity type (or
# the empty default). None of the sixteen types is "a percentage" or "a
# currency amount", so each kind maps onto the closest fit: Amount already
# covers "money, cost, limit, cap, value, quantity" by alias, and a range is
# fundamentally a bounded Amount; Rate is the ontology's word for a
# percentage. Plain text carries no ontology type -- object_type stays "".
_OBJECT_TYPE_BY_KIND: dict[str, str] = {
    "currency": "Amount",
    "quantity": "Amount",
    "range": "Amount",
    "percentage": "Rate",
    "text": "",
}
for _kind_type in _OBJECT_TYPE_BY_KIND.values():
    assert _kind_type == "" or is_valid_entity_type(_kind_type)

_QUALIFIER = r"(?:up to|above|at least|over|under|no more than|no less than)\s+"
_NUM = r"\d[\d,]*(?:\.\d+)?"

# "$65", "Up to $500", "$100/person" -- a leading qualifier and a trailing
# "/unit" are both optional.
_CURRENCY_RE = re.compile(
    rf"^(?:{_QUALIFIER})?[$€£]\s?(?P<num>{_NUM})(?:\s*/\s*(?P<unit>[A-Za-z]+))?$",
    re.IGNORECASE,
)
# "15%", "Above 40%" -- the sign must be literally present; a bare fraction
# ("0.15") is only a percentage in the context of a %-headed column, handled
# below in `classify_cell_value`.
_PERCENT_RE = re.compile(rf"^(?:{_QUALIFIER})?(?P<num>{_NUM})\s?%$", re.IGNORECASE)
# "0% - 10%", "$500 - $2,500", "5-24 seats" -- two numeric tokens (each
# optionally signed with currency/percent) joined by a dash, with an
# optional trailing unit word ("seats") on the whole range.
_RANGE_RE = re.compile(
    rf"^(?P<low>[$€£]?\s?{_NUM}\s?%?)\s*[-–—]\s*(?P<high>[$€£]?\s?{_NUM}\s?%?)"
    rf"(?:\s+(?P<unit>[A-Za-z]+))?$"
)
# "20 days", "12 weeks", "90 calendar days" -- a bare count and its unit,
# nothing else in the cell. Anything trailing ("6 hours or more (domestic)")
# fails this on purpose and falls through to plain text rather than a
# mis-parsed quantity.
_QUANTITY_UNIT_RE = re.compile(
    r"^\(?\d{1,4}\)?\+?\s*"
    r"(?:business\s+|calendar\s+|consecutive\s+|working\s+|rolling\s+)?"
    r"(?P<unit>days?|weeks?|months?|years?|hours?|minutes?|seats?|calls?)\s*$",
    re.IGNORECASE,
)
_BARE_NUMBER_RE = re.compile(rf"^{_NUM}$")
_PERCENT_HEADER_RE = re.compile(r"%|percent", re.IGNORECASE)
_CURRENCY_HEADER_RE = re.compile(r"[$€£]|\bprice\b|\bcost\b", re.IGNORECASE)
_STRIP_NON_NUMERIC_RE = re.compile(r"[^\d.,]")


def _to_float(numeric_text: str) -> float:
    return float(numeric_text.replace(",", ""))


def classify_cell_value(raw: str, header: str = "") -> CellValue:
    """Classify one cell's text, using the column header for the one case
    the text alone cannot decide: a bare fraction like "0.15" is only a
    percentage because it sits under a "Discount %" header -- on its own it
    is indistinguishable from a small quantity.
    """
    text = raw.strip()
    if not text:
        return CellValue(raw=raw, kind="text")

    match = _RANGE_RE.match(text)
    if match:
        try:
            low = _to_float(_STRIP_NON_NUMERIC_RE.sub("", match.group("low")))
            high = _to_float(_STRIP_NON_NUMERIC_RE.sub("", match.group("high")))
        except ValueError:
            low = high = None
        if low is not None:
            unit = match.group("unit") or ("%" if "%" in text else ("$" if "$" in text else None))
            return CellValue(raw=raw, kind="range", low=low, high=high, unit=unit)

    match = _PERCENT_RE.match(text)
    if match:
        return CellValue(raw=raw, kind="percentage", value=_to_float(match.group("num")))

    match = _CURRENCY_RE.match(text)
    if match:
        return CellValue(raw=raw, kind="currency", value=_to_float(match.group("num")),
                          unit=match.group("unit"))

    match = _QUANTITY_UNIT_RE.match(text)
    if match:
        digits = re.search(r"\d+(?:\.\d+)?", text)
        return CellValue(raw=raw, kind="quantity",
                          value=float(digits.group()) if digits else None,
                          unit=match.group("unit").lower())

    if _BARE_NUMBER_RE.match(text):
        number = _to_float(text)
        if _PERCENT_HEADER_RE.search(header):
            # Sheet cells for a %-formatted column read back as the raw
            # fraction (0.15), not "15%"; a value already above 1 is assumed
            # to be pre-multiplied ("15" in a "%" column).
            value = number * 100 if 0 <= number <= 1 else number
            return CellValue(raw=raw, kind="percentage", value=value)
        if _CURRENCY_HEADER_RE.search(header):
            return CellValue(raw=raw, kind="currency", value=number)
        return CellValue(raw=raw, kind="quantity", value=number)

    return CellValue(raw=raw, kind="text")


# --------------------------------------------------------------------------
# Header/title handling
# --------------------------------------------------------------------------

_LEADING_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+")


def _populated(row: list[str]) -> int:
    return sum(1 for c in row if c and str(c).strip())


def _split_header_and_data(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Peel off the header row, and a leading title row above it if present.

    A caption row ("Volume Discount Schedule") that precedes the real header
    populates at most one cell while the row beneath it populates several --
    the same test `local_parser._xlsx_group_to_blocks` uses to distinguish a
    table from a label, applied here to find where the table actually starts.
    """
    if not rows:
        return [], []
    header = [c or "" for c in rows[0]]
    body = rows[1:]
    if body and _populated(header) <= 1 and _populated(body[0]) > _populated(header):
        header = [c or "" for c in body[0]]
        body = body[1:]
    return header, body


def _section_title(section_path: str) -> str:
    """The leaf section/sheet title a table sits under, number stripped.

    `section_path` is a full breadcrumb ("2 Types of Leave > 2.3 Parental
    Leave"); only the last segment is the table's own heading.
    """
    leaf = section_path.rsplit(" > ", 1)[-1]
    return _LEADING_NUMBER_RE.sub("", leaf).strip() or leaf


def _infer_entity_type(title: str) -> str:
    """Ontology type for a table's subject/row entities, from its title.

    `coerce_entity_type` only recognises the normalized phrase as a whole, so
    a two-word title ("Volume Discounts", "Approval Matrix") needs its words
    tried individually. The last word first: English noun phrases put the
    head noun last ("Volume Discounts" is a kind of Discount, not a kind of
    Volume), so trying it first picks the more informative match when both
    words happen to resolve.
    """
    hit = coerce_entity_type(title)
    if hit:
        return hit
    words = re.findall(r"[A-Za-z]+", title)
    for word in reversed(words):
        singular = word[:-1] if word.lower().endswith("s") else word
        hit = coerce_entity_type(word) or coerce_entity_type(singular)
        if hit:
            return hit
    return _DEFAULT_ENTITY_TYPE


# --------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------


def _relation(unit: ExtractionUnit, subject: str, predicate: str, obj: str,
              subject_type: str, object_type: str, evidence: str) -> Relation:
    return Relation(
        subject=subject, predicate=predicate, object=obj,
        subject_type=subject_type, object_type=object_type,
        doc_id=unit.doc_id,
        source_chunk_id=unit.chunk_ids[0] if unit.chunk_ids else "",
        section_path=unit.section_path,
        page=unit.page,
        department=unit.department,
        # A value read off a table is not a guess: nothing here was inferred,
        # so full confidence and `deterministic=True` are the honest score.
        confidence=1.0,
        evidence_span=evidence,
        deterministic=True,
    )


def extract_from_table(rows: list[list[str]], unit: ExtractionUnit) -> tuple[list[Entity], list[Relation]]:
    """Turn one table's raw cells into entities and relations, no LLM call.

    `rows` is the table exactly as the parser produced it -- header included,
    merged/blank cells as empty strings, ragged rows shorter than the header.
    `unit` supplies the provenance (doc_id, section_path, page, department)
    every relation must carry, and its section title supplies the type the
    ontology infers the table's entities to be.
    """
    header, data_rows = _split_header_and_data(rows)
    if not any(h.strip() for h in header) or not data_rows:
        return [], []

    subject_name = _section_title(unit.section_path)
    entity_type = _infer_entity_type(subject_name)

    entities: list[Entity] = [Entity(name=subject_name, type=entity_type,
                                     department=unit.department)]
    relations: list[Relation] = []
    seen_names: set[str] = set()

    for row_index, row in enumerate(data_rows):
        if not _populated(row):
            continue  # a fully blank row -- ragged tables sometimes carry one

        row_name = row[0].strip() if row and row[0] and row[0].strip() else f"Row {row_index + 1}"
        if row_name in seen_names:
            # Two rows sharing a blank/duplicate key column would otherwise
            # collide into one entity via Entity.compute_id; disambiguate
            # rather than silently merging distinct rows.
            row_name = f"{row_name} ({row_index + 1})"
        seen_names.add(row_name)

        entities.append(Entity(name=row_name, type=entity_type, department=unit.department))
        relations.append(_relation(unit, row_name, "APPLIES_TO", subject_name,
                                    entity_type, entity_type, evidence=row_name))

        for col in range(1, len(header)):
            attr = header[col].strip()
            if not attr or col >= len(row):
                continue
            raw = (row[col] or "").strip()
            if not raw:
                continue
            typed = classify_cell_value(raw, attr)
            relations.append(_relation(unit, row_name, "HAS_VALUE", raw,
                                        entity_type, _OBJECT_TYPE_BY_KIND[typed.kind],
                                        evidence=raw))

    return entities, relations
