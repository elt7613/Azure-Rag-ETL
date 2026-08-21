"""Batched embeddings via Azure OpenAI."""
from __future__ import annotations

from openai import AsyncAzureOpenAI

from rag.config import get_settings


class AzureOpenAIEmbedder:
    def __init__(self) -> None:
        settings = get_settings()
        self._deployment = settings.azure_openai_embedding_deployment
        self._batch_size = settings.embedding_batch_size
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = await self._client.embeddings.create(
                model=self._deployment, input=batch
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
