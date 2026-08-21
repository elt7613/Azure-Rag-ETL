import pytest

from rag.models import BlockType
from tests.conftest import azure_configured

pytestmark = pytest.mark.skipif(
    not azure_configured("azure_docint_endpoint", "azure_docint_key"),
    reason="Azure Document Intelligence not configured",
)


async def test_docint_recovers_tier_table():
    from rag.parsing.azure_docint import AzureDocIntParser

    data = open("source_data/sales/Pricing2026.pdf", "rb").read()
    doc = await AzureDocIntParser().parse(data, "sales/Pricing2026.pdf")
    tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
    assert tables, "expected at least one reconstructed table"
    grid = {r[0]: r[1] for t in tables if t.rows[0][0] == "Tier" for r in t.rows[1:]}
    assert grid.get("Professional") == "$65"
