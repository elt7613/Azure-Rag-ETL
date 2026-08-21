"""The closed vocabularies and the strict JSON schema built from them.

The schema test is not cosmetic: Azure rejects a `strict: true` json_schema
outright if a single nested object omits `additionalProperties: false` or
leaves one of its properties out of `required`. That failure arrives as a 400
at extraction time, on the 5-millionth document as readily as the first, so
the structural contract is asserted here instead.
"""
from __future__ import annotations

import pytest

from rag.extraction.ontology import (
    ENTITY_TYPES,
    RELATION_TYPES,
    build_extraction_schema,
    coerce_entity_type,
    coerce_relation_type,
    is_valid_entity_type,
    is_valid_relation_type,
)


def test_vocabularies_match_the_design_exactly():
    assert list(ENTITY_TYPES) == [
        "Policy", "Department", "Role", "Benefit", "Plan", "Rate", "Vendor",
        "System", "Obligation", "Condition", "Period", "Amount", "Location",
        "Process", "Metric", "Document",
    ]
    assert list(RELATION_TYPES) == [
        "APPLIES_TO", "ELIGIBLE_FOR", "REQUIRES", "GRANTS", "LIMITS",
        "EXCLUDES", "EXCEPTION_TO", "DEFINED_IN", "REFERENCES",
        "EFFECTIVE_DURING", "HAS_VALUE", "OWNED_BY", "APPROVED_BY",
        "SUPERSEDES",
    ]


def test_validation_is_exact_and_case_sensitive():
    assert is_valid_entity_type("Policy")
    assert not is_valid_entity_type("policy")
    assert not is_valid_entity_type("Elephant")
    assert is_valid_relation_type("APPLIES_TO")
    assert not is_valid_relation_type("applies_to")
    assert not is_valid_relation_type("EATS")


@pytest.mark.parametrize("raw,expected", [
    ("policy", "Policy"),
    ("  POLICY  ", "Policy"),
    ("policies", "Policy"),
    ("job title", "Role"),
    ("time_period", "Period"),
    ("money", "Amount"),
    ("software", "System"),
    ("supplier", "Vendor"),
])
def test_entity_type_coercion_pulls_near_misses_into_the_closed_set(raw, expected):
    assert coerce_entity_type(raw) == expected


@pytest.mark.parametrize("raw", ["Elephant", "", "   ", None, "Thing"])
def test_entity_type_coercion_rejects_rather_than_guesses(raw):
    assert coerce_entity_type(raw) is None


@pytest.mark.parametrize("raw,expected", [
    ("applies to", "APPLIES_TO"),
    ("APPLIES-TO", "APPLIES_TO"),
    ("is_eligible_for", "ELIGIBLE_FOR"),
    ("must", "REQUIRES"),
    ("provides", "GRANTS"),
    ("caps", "LIMITS"),
    ("refers to", "REFERENCES"),
    ("replaces", "SUPERSEDES"),
])
def test_relation_type_coercion_pulls_near_misses_into_the_closed_set(raw, expected):
    assert coerce_relation_type(raw) == expected


@pytest.mark.parametrize("raw", ["APPROVES", "DEFINES", "REQUIRED_BY", "EATS", "", None])
def test_relation_coercion_refuses_to_invert_or_invent_an_edge(raw):
    """An inverse ("APPROVES" for APPROVED_BY) would silently reverse the edge."""
    assert coerce_relation_type(raw) is None


# ---- strict-mode structural contract ----

def _objects(node, path="$"):
    """Yield (path, object_schema) for every object-typed node in the schema."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield path, node
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                for prop, sub in value.items():
                    yield from _objects(sub, f"{path}.properties.{prop}")
            elif key == "items":
                yield from _objects(value, f"{path}.items")
            elif isinstance(value, (dict, list)):
                yield from _objects(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _objects(value, f"{path}[{i}]")


def test_schema_envelope_is_shaped_for_response_format():
    envelope = build_extraction_schema()
    assert envelope["strict"] is True
    assert isinstance(envelope["name"], str) and envelope["name"]
    assert envelope["schema"]["type"] == "object"


def test_every_object_forbids_extra_properties_and_requires_all_of_them():
    schema = build_extraction_schema()["schema"]
    found = list(_objects(schema))
    assert len(found) >= 3  # root, unit, entity, relation
    for path, obj in found:
        assert obj.get("additionalProperties") is False, f"{path} allows extra properties"
        assert set(obj.get("required", [])) == set(obj.get("properties", {})), (
            f"{path}: required must list every property under strict mode"
        )


def test_schema_enum_constrains_output_to_the_ontology():
    schema = build_extraction_schema()["schema"]
    enums = {tuple(obj["properties"][p]["enum"])
             for _, obj in _objects(schema)
             for p in obj.get("properties", {})
             if "enum" in obj["properties"][p]}
    assert tuple(ENTITY_TYPES) in enums
    assert tuple(RELATION_TYPES) in enums


def test_schema_carries_a_per_unit_id_so_packed_units_stay_attributable():
    schema = build_extraction_schema()["schema"]
    unit = schema["properties"]["units"]["items"]
    assert "unit_id" in unit["properties"]
    assert "entities" in unit["properties"]
    assert "relations" in unit["properties"]


def test_relations_must_carry_evidence_and_confidence():
    """No edge without a citable source -- enforced at the schema level."""
    schema = build_extraction_schema()["schema"]
    unit = schema["properties"]["units"]["items"]
    relation = unit["properties"]["relations"]["items"]
    assert {"subject", "predicate", "object", "confidence", "evidence_span"} <= set(
        relation["properties"]
    )
