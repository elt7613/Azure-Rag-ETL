"""Department registry: the env list is the single source of truth."""
from __future__ import annotations

import pytest

from rag.config import Settings
from rag.departments import DepartmentRegistry


def _settings(**overrides) -> Settings:
    base = dict(
        departments=["HR", "finance", "IT"],
        azure_openai_endpoint="https://example.openai.azure.com/",
        azure_openai_key="k",
        azure_openai_embedding_deployment="text-embedding-3-small",
        azure_search_endpoint="https://example.search.windows.net",
        azure_search_key="k",
        azure_search_index="idx",
    )
    base.update(overrides)
    return Settings(**base)


def test_resolves_department_from_leading_path_segment():
    reg = DepartmentRegistry(_settings())
    assert reg.department_for_key("HR/LeavePolicy.pdf") == "HR"
    assert reg.department_for_key("finance/ExpensePolicy.pdf") == "finance"


def test_lookup_is_case_insensitive_but_returns_canonical_name():
    reg = DepartmentRegistry(_settings())
    assert reg.department_for_key("hr/LeavePolicy.pdf") == "HR"
    assert reg.department_for_key("It/VPNGuide.pdf") == "IT"


def test_department_sources_decouples_name_from_location():
    reg = DepartmentRegistry(
        _settings(department_sources={"HR": "people-ops", "IT": "engineering/it"})
    )
    assert reg.department_for_key("people-ops/LeavePolicy.pdf") == "HR"
    # The department's own name still resolves, so either layout works.
    assert reg.department_for_key("HR/LeavePolicy.pdf") == "HR"
    assert reg.source_prefixes() == ["people-ops", "finance", "engineering/it"]
    # A multi-segment prefix resolves on the whole prefix, not its first
    # segment -- "engineering/" alone belongs to nobody.
    assert reg.department_for_key("engineering/it/VPNGuide.pdf") == "IT"
    assert reg.lookup("engineering/Roadmap.pdf") is None


def test_adding_a_department_is_config_only():
    reg = DepartmentRegistry(_settings(departments=["HR", "finance", "IT", "legal"]))
    assert "legal" in reg.names()
    assert reg.is_ingestable("legal/NDA.docx")


def test_unknown_department_is_skipped_by_default():
    reg = DepartmentRegistry(_settings())
    assert reg.is_ingestable("marketing/Brochure.pdf") is False


def test_unknown_department_ingested_when_policy_says_so():
    reg = DepartmentRegistry(_settings(unknown_department_policy="ingest"))
    assert reg.is_ingestable("marketing/Brochure.pdf") is True
    assert reg.department_for_key("marketing/Brochure.pdf") == "marketing"


@pytest.mark.parametrize("scope", [None, []])
def test_access_denies_by_default(scope):
    reg = DepartmentRegistry(_settings())
    assert reg.is_allowed("HR/LeavePolicy.pdf", scope) is False


def test_access_scoped_to_allowed_departments():
    reg = DepartmentRegistry(_settings())
    assert reg.is_allowed("HR/LeavePolicy.pdf", ["HR"]) is True
    assert reg.is_allowed("HR/LeavePolicy.pdf", ["IT", "finance"]) is False
    assert reg.is_allowed("HR/LeavePolicy.pdf", ["hr"]) is True
