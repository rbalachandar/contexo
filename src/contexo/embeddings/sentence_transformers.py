"""Sentence-transformers embedding provider for local embeddings."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from contexo.core.exceptions import EmbeddingError
from contexo.embeddings.base import EmbeddingProvider

# Cache for models to avoid reloading
_model_cache: dict[str, Any] = {}


@lru_cache(maxsize=1)
def _get_model(model_name: str, device: str | None):
    """Get or load a sentence-transformers model.

    Args:
        model_name: Name of the model to load
        device: Device to load the model on

    Returns:
        The loaded model
    """
    from sentence_transformers import SentenceTransformer

    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name, device=device)
    return _model_cache[model_name]


class SentenceTransformersEmbeddings(EmbeddingProvider):
    """Sentence-transformers embedding provider for local embeddings.

    This provider runs locally and doesn't require an API key.
    Models are downloaded from Hugging Face on first use.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the sentence-transformers embedding provider.

        Args:
            model_name: Name of the sentence-transformers model
                Popular choices:
                - "all-MiniLM-L6-v2": Fast, good quality (384 dim)
                - "all-mpnet-base-v2": High quality (768 dim)
                - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual
            device: Device to run on (e.g., "cpu", "cuda", "mps")
            **kwargs: Additional arguments (unused, for compatibility)
        """
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._dimension = 0
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the model."""
        if self._initialized:
            return

        try:
            self._model = _get_model(self._model_name, self._device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            self._initialized = True
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize sentence-transformers model: {e}") from e

    async def close(self) -> None:
        """Close the embedding provider."""
        # Keep the model cached for future use
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
        """Generate an embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            A vector embedding as a list of floats
        """
        if not self.is_initialized:
            raise EmbeddingError("Embedding provider not initialized")

        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
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
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

    @staticmethod
    def similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Similarity score from -1 to 1 (1 is identical)
        """
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
