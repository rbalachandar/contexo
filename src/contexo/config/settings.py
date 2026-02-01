"""Configuration dataclasses for Contexo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StorageConfig:
    """Configuration for storage backends."""

    backend_type: str = "in_memory"  # in_memory, sqlite, postgresql, redis, filesystem
    connection_string: str | None = None
    db_path: str = ":memory:"
    table_prefix: str = "contexo_"
    pool_size: int = 5
    max_overflow: int = 10


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding providers."""

    provider_type: str = "mock"  # mock, openai, cohere, sentence_transformers
    model_name: str = "mock"
    api_key: str | None = None
    dimension: int | None = None
    batch_size: int = 32
    device: str | None = None


@dataclass(frozen=True)
class WorkingMemoryConfig:
    """Configuration for working memory."""

    max_tokens: int = 4096
    max_entries: int | None = None
    strategy: str = "sliding_window"  # passthrough, sliding_window, summarization, importance
    token_counter: Callable[[str], int] | None = None

    # Strategy-specific settings
    summary_target_tokens: int = 200
    importance_recency_bias: float = 0.1

    # Section configuration
    sections: dict[str, dict[str, Any]] | None = None  # Section config for sectioned mode


@dataclass(frozen=True)
class ContexoConfig:
    """Main configuration for Contexo.

    Attributes:
        storage: Storage backend configuration
        embeddings: Embedding provider configuration
        working_memory: Working memory configuration
        auto_initialize: Whether to auto-initialize on creation
        enable_provenance: Whether to track provenance
        conversation_id: Default conversation ID for entries
        multi_agent: Enable multi-agent mode with agent_id and scope metadata
    """

    storage: StorageConfig = field(default_factory=StorageConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    working_memory: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    auto_initialize: bool = True
    enable_provenance: bool = False
    conversation_id: str | None = None
    multi_agent: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
