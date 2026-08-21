"""The closed vocabularies the extractor is allowed to emit, and the strict
JSON Schema that enforces them at the API boundary.

Two reasons the sets are small and fixed rather than open-ended:

*Cost.* The ontology is rendered into the fixed system prefix of every
extraction call. A sixteen-type vocabulary keeps that prefix short enough to
stay cheap while still clearing Azure's 1024-token prompt-cache threshold; a
free-form "invent the schema as you go" prompt would be larger, would vary per
call, and would defeat prefix caching entirely.

*Usability.* A graph whose edge labels are whatever the model felt like saying
that afternoon cannot be traversed by a retriever. `APPLIES_TO` has to mean the
same thing in the HR corpus and the legal corpus for an entity-anchored
traversal to be worth writing.

The model still drifts -- `"applies to"`, `"is_eligible_for"`, `"policies"` --
so `coerce_*` maps the near misses back onto the closed set and returns `None`
for everything else. Coercion deliberately refuses to map an *inverse*
(`APPROVES` -> `APPROVED_BY`, `DEFINES` -> `DEFINED_IN`): silently reversing an
edge's direction is worse than dropping it, because a reversed edge still looks
citable.
"""
from __future__ import annotations

# Order matters: it is the order the enum appears in the prompt schema, and a
# stable order keeps the rendered prefix byte-identical between runs, which is
# what prompt caching keys on.
ENTITY_TYPES: tuple[str, ...] = (
    "Policy", "Department", "Role", "Benefit", "Plan", "Rate", "Vendor",
    "System", "Obligation", "Condition", "Period", "Amount", "Location",
    "Process", "Metric", "Document",
)

RELATION_TYPES: tuple[str, ...] = (
    "APPLIES_TO", "ELIGIBLE_FOR", "REQUIRES", "GRANTS", "LIMITS", "EXCLUDES",
    "EXCEPTION_TO", "DEFINED_IN", "REFERENCES", "EFFECTIVE_DURING",
    "HAS_VALUE", "OWNED_BY", "APPROVED_BY", "SUPERSEDES",
)

_ENTITY_SET = frozenset(ENTITY_TYPES)
_RELATION_SET = frozenset(RELATION_TYPES)

# Near misses seen from instruction-tuned models: synonyms, the plural, and the
# generic word a model reaches for when the ontology term does not spring to
# mind. Keys are normalized (lowercase, underscores collapsed) before lookup.
_ENTITY_ALIASES: dict[str, str] = {
    "policies": "Policy", "rule": "Policy", "guideline": "Policy",
    "departments": "Department", "team": "Department", "org_unit": "Department",
    "business_unit": "Department", "function": "Department",
    "roles": "Role", "job_title": "Role", "title": "Role", "position": "Role",
    "person": "Role", "employee": "Role", "approver": "Role",
    "benefits": "Benefit", "perk": "Benefit", "coverage": "Benefit",
    "entitlement": "Benefit",
    "plans": "Plan", "tier": "Plan", "program": "Plan", "product": "Plan",
    "rates": "Rate", "price": "Rate", "pricing": "Rate", "discount": "Rate",
    "percentage": "Rate", "fee": "Rate",
    "vendors": "Vendor", "supplier": "Vendor", "provider": "Vendor",
    "counterparty": "Vendor", "partner": "Vendor", "company": "Vendor",
    "organization": "Vendor", "organisation": "Vendor",
    "systems": "System", "software": "System", "application": "System",
    "app": "System", "tool": "System", "platform": "System", "portal": "System",
    "obligations": "Obligation", "requirement": "Obligation", "duty": "Obligation",
    "restriction": "Obligation", "prohibition": "Obligation",
    "conditions": "Condition", "criteria": "Condition", "criterion": "Condition",
    "eligibility": "Condition", "prerequisite": "Condition", "trigger": "Condition",
    "periods": "Period", "date": "Period", "duration": "Period",
    "time_period": "Period", "timeframe": "Period", "term": "Period",
    "deadline": "Period", "schedule": "Period",
    "amounts": "Amount", "money": "Amount", "cost": "Amount", "limit": "Amount",
    "cap": "Amount", "currency": "Amount", "value": "Amount", "quantity": "Amount",
    "locations": "Location", "place": "Location", "city": "Location",
    "country": "Location", "region": "Location", "site": "Location",
    "processes": "Process", "procedure": "Process", "workflow": "Process",
    "step": "Process", "activity": "Process",
    "metrics": "Metric", "measure": "Metric", "kpi": "Metric", "sla": "Metric",
    "threshold": "Metric",
    "documents": "Document", "agreement": "Document", "contract": "Document",
    "form": "Document", "record": "Document",
}

_RELATION_ALIASES: dict[str, str] = {
    "applies": "APPLIES_TO", "applicable_to": "APPLIES_TO",
    "governs": "APPLIES_TO", "covers": "APPLIES_TO", "scope": "APPLIES_TO",
    "eligible": "ELIGIBLE_FOR", "is_eligible_for": "ELIGIBLE_FOR",
    "are_eligible_for": "ELIGIBLE_FOR", "entitled_to": "ELIGIBLE_FOR",
    "is_entitled_to": "ELIGIBLE_FOR", "qualifies_for": "ELIGIBLE_FOR",
    "require": "REQUIRES", "required": "REQUIRES", "must": "REQUIRES",
    "shall": "REQUIRES", "needs": "REQUIRES", "mandates": "REQUIRES",
    "grant": "GRANTS", "provides": "GRANTS", "gives": "GRANTS",
    "offers": "GRANTS", "allows": "GRANTS", "permits": "GRANTS",
    "limit": "LIMITS", "caps": "LIMITS", "capped_at": "LIMITS",
    "restricts": "LIMITS", "max": "LIMITS", "maximum": "LIMITS",
    "exclude": "EXCLUDES", "excluded": "EXCLUDES", "prohibits": "EXCLUDES",
    "forbids": "EXCLUDES", "not_covered_by": "EXCLUDES",
    "exception": "EXCEPTION_TO", "exempt_from": "EXCEPTION_TO",
    "waiver_of": "EXCEPTION_TO",
    "defined_by": "DEFINED_IN", "described_in": "DEFINED_IN",
    "documented_in": "DEFINED_IN", "specified_in": "DEFINED_IN",
    "stated_in": "DEFINED_IN",
    "reference": "REFERENCES", "refers_to": "REFERENCES",
    "mentions": "REFERENCES", "see_also": "REFERENCES", "cites": "REFERENCES",
    "effective": "EFFECTIVE_DURING", "valid_during": "EFFECTIVE_DURING",
    "effective_from": "EFFECTIVE_DURING", "in_effect_during": "EFFECTIVE_DURING",
    "value": "HAS_VALUE", "has_amount": "HAS_VALUE", "equals": "HAS_VALUE",
    "is": "HAS_VALUE", "costs": "HAS_VALUE", "priced_at": "HAS_VALUE",
    "owner": "OWNED_BY", "owned": "OWNED_BY", "belongs_to": "OWNED_BY",
    "managed_by": "OWNED_BY", "maintained_by": "OWNED_BY",
    "approval_by": "APPROVED_BY", "requires_approval_from": "APPROVED_BY",
    "authorized_by": "APPROVED_BY", "signed_off_by": "APPROVED_BY",
    "supersede": "SUPERSEDES", "replaces": "SUPERSEDES",
    "supersedes_document": "SUPERSEDES", "obsoletes": "SUPERSEDES",
}

# Verbs whose ontology counterpart points the other way. Mapping them would
# invert the edge, so they are rejected explicitly rather than falling through
# to a fuzzy match that might one day accept them.
_INVERSES: frozenset[str] = frozenset({
    "approves", "approve", "defines", "define", "required_by", "granted_by",
    "limited_by", "excluded_by", "owns", "own", "superseded_by",
    "applied_by", "referenced_by", "contains",
})


def _normalize(raw: str | None) -> str:
    if not raw:
        return ""
    cleaned = raw.strip().lower()
    for ch in (" ", "-", ".", "/"):
        cleaned = cleaned.replace(ch, "_")
    return "_".join(part for part in cleaned.split("_") if part)


def is_valid_entity_type(name: str) -> bool:
    """Exact membership. Case-sensitive on purpose -- the canonical casing is
    what gets written to Neo4j as the node label, and `policy` and `Policy`
    would become two labels."""
    return name in _ENTITY_SET


def is_valid_relation_type(name: str) -> bool:
    return name in _RELATION_SET


def coerce_entity_type(raw: str | None) -> str | None:
    """Canonical entity type for `raw`, or None if it is not recognisable.

    Returning None is a real outcome, not a failure to try harder: an entity
    the model could not fit into the ontology is dropped, and dropping it is
    cheaper to live with than a graph full of one-off labels.
    """
    key = _normalize(raw)
    if not key:
        return None
    exact = {t.lower(): t for t in ENTITY_TYPES}
    if key in exact:
        return exact[key]
    if key in _ENTITY_ALIASES:
        return _ENTITY_ALIASES[key]
    # Bare plural of a canonical type ("periods" is aliased, "roles" is too,
    # but this catches any that were missed).
    if key.endswith("s") and key[:-1] in exact:
        return exact[key[:-1]]
    return None


def coerce_relation_type(raw: str | None) -> str | None:
    """Canonical relation type for `raw`, or None. Never inverts an edge."""
    key = _normalize(raw)
    if not key or key in _INVERSES:
        return None
    exact = {t.lower(): t for t in RELATION_TYPES}
    if key in exact:
        return exact[key]
    return _RELATION_ALIASES.get(key)


# --------------------------------------------------------------------------
# Strict JSON Schema
# --------------------------------------------------------------------------
#
# Azure OpenAI's `response_format={"type": "json_schema", "strict": true}`
# imposes structural rules the schema must satisfy or the request 400s:
#   * every object sets `additionalProperties: false`
#   * every property of an object appears in that object's `required` list
#     (optionality is expressed with a nullable/empty value, not by omission)
#   * the root is an object
# `_strict_object` is the only way objects are built below so none of those can
# be forgotten in a later edit.

_SCHEMA_NAME = "graph_extraction"


def _strict_object(properties: dict[str, dict], description: str = "") -> dict:
    schema: dict = {
        "type": "object",
        "properties": properties,
        # Strict mode: required must be *every* key, not the semantically
        # mandatory subset.
        "required": list(properties),
        "additionalProperties": False,
    }
    if description:
        schema["description"] = description
    return schema


def build_extraction_schema() -> dict:
    """The `json_schema` value for `response_format`, ontology-constrained.

    The response is a list of per-unit results rather than a single flat list
    of relations because units are packed several to a call (see
    `extraction.units.pack_units`): without `unit_id` on each group there is no
    way to attribute a relation back to the section -- and therefore to the
    chunk, page, and document -- it came from, and a relation without
    provenance is one we refuse to write.
    """
    entity = _strict_object(
        {
            "name": {
                "type": "string",
                "description": "Entity name exactly as written in the source text.",
            },
            "type": {
                "type": "string",
                "enum": list(ENTITY_TYPES),
                "description": "Closed vocabulary; pick the closest fit.",
            },
            "description": {
                "type": "string",
                "description": "One short clause from the text, or an empty string.",
            },
        },
        "A thing the passage talks about.",
    )

    relation = _strict_object(
        {
            "subject": {"type": "string", "description": "Subject entity name."},
            "subject_type": {"type": "string", "enum": list(ENTITY_TYPES)},
            "predicate": {"type": "string", "enum": list(RELATION_TYPES)},
            "object": {"type": "string", "description": "Object entity name."},
            "object_type": {"type": "string", "enum": list(ENTITY_TYPES)},
            "confidence": {
                "type": "number",
                "description": "0.0-1.0 confidence that the text states this.",
            },
            "evidence_span": {
                "type": "string",
                "description": (
                    "Verbatim substring of the unit text supporting the relation. "
                    "Omit the relation entirely if no such span exists."
                ),
            },
        },
        "A typed edge, with the span of source text that justifies it.",
    )

    unit = _strict_object(
        {
            "unit_id": {
                "type": "string",
                "description": "Echo the id of the unit these findings came from.",
            },
            "entities": {"type": "array", "items": entity},
            "relations": {"type": "array", "items": relation},
        },
        "Findings for one extraction unit.",
    )

    root = _strict_object(
        {"units": {"type": "array", "items": unit}},
        "One entry per unit supplied in the request, in the same order.",
    )

    return {"name": _SCHEMA_NAME, "strict": True, "schema": root}


def response_format() -> dict:
    """Ready to hand to the chat completions call."""
    return {"type": "json_schema", "json_schema": build_extraction_schema()}
