"""OpenAI embedding provider."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from contexo.core.exceptions import EmbeddingError
from contexo.embeddings.base import EmbeddingProvider


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider using the API.

    Supports text-embedding-3-small and text-embedding-3-large models.
    """

    # Model dimensions
    DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimension: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI embedding provider.

        Args:
            model: The OpenAI embedding model to use
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            dimension: Optional dimension to truncate embeddings to
            **kwargs: Additional arguments passed to AsyncOpenAI client
        """
        self._model = model
        self._requested_dimension = dimension
        self._client_kwargs = kwargs
        self._client: AsyncOpenAI | None = None
        self._initialized = False

        # Set the actual dimension
        self._dimension = dimension or self.DIMENSIONS.get(model, 1536)

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self._initialized:
            return

        self._client = AsyncOpenAI(
            api_key=self._client_kwargs.pop("api_key", None), **self._client_kwargs
        )
        self._initialized = True

    async def close(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the provider has been initialized."""
        return self._initialized

    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Get the name of the embedding model."""
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            A vector embedding as a list of floats
        """
        if not self.is_initialized:
            raise EmbeddingError("Embedding provider not initialized")

        try:
            # Only pass dimensions if explicitly set (API doesn't accept None)
            params = {"model": self._model, "input": text}
            if self._requested_dimension is not None:
                params["dimensions"] = self._requested_dimension

            response = await self._client.embeddings.create(**params)  # type: ignore
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: The texts to embed

        Returns:
            A list of vector embeddings
        """
        if not self.is_initialized:
            raise EmbeddingError("Embedding provider not initialized")

        if not texts:
            return []

        try:
            # Only pass dimensions if explicitly set (API doesn't accept None)
            params = {"model": self._model, "input": texts}
            if self._requested_dimension is not None:
                params["dimensions"] = self._requested_dimension

            response = await self._client.embeddings.create(**params)  # type: ignore
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e
