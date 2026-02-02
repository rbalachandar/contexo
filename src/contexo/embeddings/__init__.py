"""Embedding provider implementations and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contexo.embeddings.base import EmbeddingProvider
from contexo.embeddings.mock import MockEmbeddings

if TYPE_CHECKING:
    from contexo.embeddings.openai import OpenAIEmbeddings as _OpenAIEmbeddings
    from contexo.embeddings.sentence_transformers import (
        SentenceTransformersEmbeddings as _SentenceTransformersEmbeddings,
    )

try:
    from contexo.embeddings.openai import OpenAIEmbeddings

    _openai_available = True
except ImportError:
    _openai_available = False

    class OpenAIEmbeddings:  # type: ignore[no-redef]
        """Placeholder when OpenAI is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "openai is required for OpenAI embeddings. Install with: pip install openai"
            )


try:
    from contexo.embeddings.sentence_transformers import (
        SentenceTransformersEmbeddings,
    )

    _sentence_transformers_available = True
except ImportError:
    _sentence_transformers_available = False

    class SentenceTransformersEmbeddings:  # type: ignore[no-redef]
        """Placeholder when sentence-transformers is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )


def create_embedding_provider(provider_type: str, **kwargs: Any) -> EmbeddingProvider:
    """Create an embedding provider by type name.

    Args:
        provider_type: The type of provider to create
            - "mock" or "test": MockEmbeddings
            - "openai": OpenAIEmbeddings
            - "sentence_transformers" or "sentence-transformers": SentenceTransformersEmbeddings
        **kwargs: Additional arguments passed to the provider constructor

    Returns:
        An embedding provider instance

    Raises:
        ValueError: If the provider type is unknown
    """
    provider_type = provider_type.lower().replace("-", "_")

    if provider_type in ("mock", "test"):
        # MockEmbeddings only uses dimension and model_name
        # Filter out None values to use defaults
        mock_kwargs = {
            k: v for k, v in kwargs.items() if k in ("dimension", "model_name") and v is not None
        }
        return MockEmbeddings(**mock_kwargs)

    if provider_type == "openai":
        if not _openai_available:
            raise ImportError(
                "openai is required for OpenAI embeddings. Install with: pip install openai"
            )
        # OpenAIEmbeddings uses 'model' parameter, but config uses 'model_name'
        # Also filter out 'device' which is for local models only
        openai_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("device", "batch_size")
        }
        if "model_name" in openai_kwargs and "model" not in openai_kwargs:
            openai_kwargs["model"] = openai_kwargs.pop("model_name")
        return OpenAIEmbeddings(**openai_kwargs)

    if provider_type in ("sentence_transformers", "sentencetransformers"):
        if not _sentence_transformers_available:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        return SentenceTransformersEmbeddings(**kwargs)

    raise ValueError(f"Unknown embedding provider type: {provider_type}")


__all__ = [
    "EmbeddingProvider",
    "MockEmbeddings",
    "OpenAIEmbeddings",
    "SentenceTransformersEmbeddings",
    "create_embedding_provider",
]
