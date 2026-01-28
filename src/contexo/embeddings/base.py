"""Base embedding provider protocol."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Protocol defining the interface for embedding providers.

    Embedding providers convert text into vector embeddings for
    semantic search and similarity comparisons.
    """

    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        ...

    @property
    def model_name(self) -> str:
        """Get the name of the embedding model."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            A vector embedding as a list of floats
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: The texts to embed

        Returns:
            A list of vector embeddings
        """
        ...

    async def initialize(self) -> None:
        """Initialize the embedding provider.

        This should be called before using the provider.
        """
        ...

    async def close(self) -> None:
        """Close the embedding provider and release resources."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if the embedding provider has been initialized."""
        ...
