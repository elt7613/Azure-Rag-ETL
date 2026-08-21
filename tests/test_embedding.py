import pytest

from tests.conftest import azure_configured

pytestmark = pytest.mark.skipif(
    not azure_configured(
        "azure_openai_endpoint", "azure_openai_key", "azure_openai_embedding_deployment"
    ),
    reason="Azure OpenAI not configured",
)


async def test_embed_returns_configured_dimensions():
    from rag.config import get_settings
    from rag.embedding.azure_openai import AzureOpenAIEmbedder

    vectors = await AzureOpenAIEmbedder().embed(["hello world", "second text"])
    assert len(vectors) == 2
    assert len(vectors[0]) == get_settings().embedding_dimensions


async def test_batches_beyond_batch_size():
    from rag.embedding.azure_openai import AzureOpenAIEmbedder

    texts = [f"chunk number {i}" for i in range(70)]
    vectors = await AzureOpenAIEmbedder().embed(texts)
    assert len(vectors) == 70
