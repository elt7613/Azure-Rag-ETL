import os
import pytest
from rag.config import Settings


def test_settings_reads_departments_from_env(monkeypatch):
    monkeypatch.setenv("DEPARTMENTS", '["HR","finance","IT","legal","sales"]')
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://x.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_KEY", "k")
    monkeypatch.setenv("AZURE_SEARCH_INDEX", "kb")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    s = Settings(_env_file=None)
    assert s.departments == ["HR", "finance", "IT", "legal", "sales"]
    assert s.embedding_dimensions == 1536


def test_settings_rejects_missing_required(monkeypatch):
    for k in ("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "AZURE_SEARCH_INDEX"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(Exception):
        Settings(_env_file=None)
