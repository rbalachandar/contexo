"""Mock embedding provider for testing."""

from __future__ import annotations

import hashlib
from typing import Any

from contexo.embeddings.base import EmbeddingProvider


class MockEmbeddings(EmbeddingProvider):
    """Mock embedding provider for testing.

    Generates deterministic embeddings based on input text hash.
    Not suitable for production use.
    """

    def __init__(self, dimension: int = 384, model_name: str = "mock") -> None:
        """Initialize the mock embedding provider.

        Args:
            dimension: Dimension of the embedding vectors
            model_name: Name to report for the model
        """
        self._dimension = dimension
        self._model_name = model_name
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the mock provider."""
        self._initialized = True

    async def close(self) -> None:
        """Close the mock provider."""
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
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        """Generate a deterministic mock embedding.

        Args:
            text: The text to embed

        Returns:
            A mock embedding vector
        """
        # Generate a deterministic hash-based embedding
        hash_bytes = hashlib.sha256(text.encode()).digest()
        values = []
        for i in range(self._dimension):
            # Use different bytes for different dimensions
            byte_idx = i % len(hash_bytes)
            # Convert to float between -1 and 1
            value = (hash_bytes[byte_idx] / 127.5) - 1.0
            values.append(value)
        return values

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for multiple texts.

        Args:
            texts: The texts to embed

        Returns:
            A list of mock embedding vectors
        """
        return [await self.embed(text) for text in texts]
