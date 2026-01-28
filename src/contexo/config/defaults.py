"""Default configuration presets for Contexo."""

from __future__ import annotations

from typing import Any, Dict

from contexo.config.settings import (
    ContexoConfig,
    EmbeddingConfig,
    StorageConfig,
    WorkingMemoryConfig,
)


def minimal_config() -> ContexoConfig:
    """Create a minimal configuration using in-memory storage only.

    This is the fastest option for testing and development.
    Data is lost when the program exits.

    Returns:
        A minimal ContexoConfig
    """
    return ContexoConfig(
        storage=StorageConfig(backend_type="in_memory"),
        embeddings=EmbeddingConfig(provider_type="mock"),
        working_memory=WorkingMemoryConfig(
            max_tokens=4096,
            strategy="sliding_window",
        ),
        auto_initialize=True,
        enable_provenance=False,
    )


def local_config(
    db_path: str = "./context.db",
    model_name: str = "all-MiniLM-L6-v2",
    max_tokens: int = 4096,
) -> ContexoConfig:
    """Create a local configuration with SQLite and local embeddings.

    This setup runs entirely locally without external API calls.

    Args:
        db_path: Path to the SQLite database file
        model_name: Name of the sentence-transformers model
        max_tokens: Maximum tokens in working memory

    Returns:
        A local ContexoConfig
    """
    return ContexoConfig(
        storage=StorageConfig(
            backend_type="sqlite",
            db_path=db_path,
        ),
        embeddings=EmbeddingConfig(
            provider_type="sentence_transformers",
            model_name=model_name,
        ),
        working_memory=WorkingMemoryConfig(
            max_tokens=max_tokens,
            strategy="sliding_window",
        ),
        auto_initialize=True,
        enable_provenance=True,
    )


def cloud_config(
    connection_string: str,
    api_key: str,
    model: str = "text-embedding-3-small",
    max_tokens: int = 8192,
    strategy: str = "summarization",
) -> ContexoConfig:
    """Create a cloud configuration with PostgreSQL and OpenAI.

    This setup uses cloud services for production applications.

    Args:
        connection_string: Database connection string
        api_key: OpenAI API key
        model: OpenAI embedding model name
        max_tokens: Maximum tokens in working memory
        strategy: Compaction strategy to use

    Returns:
        A cloud ContexoConfig
    """
    return ContexoConfig(
        storage=StorageConfig(
            backend_type="postgresql",
            connection_string=connection_string,
        ),
        embeddings=EmbeddingConfig(
            provider_type="openai",
            model_name=model,
            api_key=api_key,
        ),
        working_memory=WorkingMemoryConfig(
            max_tokens=max_tokens,
            strategy=strategy,
            summary_target_tokens=200,
        ),
        auto_initialize=True,
        enable_provenance=True,
    )


def development_config() -> ContexoConfig:
    """Create a development configuration.

    Similar to local_config but with provenance enabled and
    a smaller context window for faster iteration.

    Returns:
        A development ContexoConfig
    """
    return ContexoConfig(
        storage=StorageConfig(
            backend_type="sqlite",
            db_path="./dev_context.db",
        ),
        embeddings=EmbeddingConfig(
            provider_type="mock",
        ),
        working_memory=WorkingMemoryConfig(
            max_tokens=2048,
            strategy="sliding_window",
        ),
        auto_initialize=True,
        enable_provenance=True,
    )


def production_config(
    connection_string: str,
    api_key: str,
    embedding_model: str = "text-embedding-3-large",
    max_tokens: int = 16384,
) -> ContexoConfig:
    """Create a production-optimized configuration.

    Uses cloud services with larger context windows and
    provenance tracking enabled.

    Args:
        connection_string: Database connection string
        api_key: OpenAI API key
        embedding_model: OpenAI embedding model name
        max_tokens: Maximum tokens in working memory

    Returns:
        A production ContexoConfig
    """
    return ContexoConfig(
        storage=StorageConfig(
            backend_type="postgresql",
            connection_string=connection_string,
            pool_size=20,
            max_overflow=40,
        ),
        embeddings=EmbeddingConfig(
            provider_type="openai",
            model_name=embedding_model,
            api_key=api_key,
            batch_size=64,
        ),
        working_memory=WorkingMemoryConfig(
            max_tokens=max_tokens,
            strategy="importance",
            importance_recency_bias=0.2,
        ),
        auto_initialize=True,
        enable_provenance=True,
    )


def graphdb_config(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    database: str = "neo4j",
    model_name: str = "all-MiniLM-L6-v2",
    max_tokens: int = 8192,
) -> ContexoConfig:
    """Create a graph database configuration with Neo4j.

    This setup uses Neo4j for relationship-based memory storage,
    enabling powerful queries on connections between memories.

    Args:
        uri: Neo4j connection URI
        user: Database user
        password: Database password
        database: Database name
        model_name: Name of the sentence-transformers model
        max_tokens: Maximum tokens in working memory

    Returns:
        A graph database ContexoConfig
    """
    return ContexoConfig(
        storage=StorageConfig(
            backend_type="graphdb",
            connection_string=uri,
        ),
        embeddings=EmbeddingConfig(
            provider_type="sentence_transformers",
            model_name=model_name,
        ),
        working_memory=WorkingMemoryConfig(
            max_tokens=max_tokens,
            strategy="importance",
            importance_recency_bias=0.1,
        ),
        auto_initialize=True,
        enable_provenance=True,
    )


def chat_config(
    max_tokens: int = 8192,
    sections: dict[str, dict[str, Any]] | None = None,
) -> ContexoConfig:
    """Create a configuration optimized for chat applications.

    Provides pre-configured sections for chat applications:
    - system: System prompt (pinned, high priority)
    - user_profile: User preferences (pinned, high priority)
    - conversation: Chat messages (medium priority, largest allocation)
    - rag_context: Retrieved documents (medium priority)
    - tools: Function call outputs (low priority, first to evict)

    Args:
        max_tokens: Total context window size
        sections: Optional custom section configuration to override defaults

    Returns:
        A chat-optimized ContexoConfig
    """
    # Default section sizes for 8K context
    default_sections = {
        "system": {"max_tokens": 500, "priority": 1.0, "pinned": True},
        "user_profile": {"max_tokens": 300, "priority": 1.0, "pinned": True},
        "conversation": {"max_tokens": 5000, "priority": 0.5},
        "rag_context": {"max_tokens": 1200, "priority": 0.5},
        "tools": {"max_tokens": 500, "priority": 0.2},
    }

    # Scale for different context sizes
    if max_tokens <= 4096:
        default_sections = {
            "system": {"max_tokens": 300, "priority": 1.0, "pinned": True},
            "user_profile": {"max_tokens": 200, "priority": 1.0, "pinned": True},
            "conversation": {"max_tokens": 2800, "priority": 0.5},
            "rag_context": {"max_tokens": 500, "priority": 0.5},
            "tools": {"max_tokens": 200, "priority": 0.2},
        }
    elif max_tokens >= 16384:
        default_sections = {
            "system": {"max_tokens": 1000, "priority": 1.0, "pinned": True},
            "user_profile": {"max_tokens": 500, "priority": 1.0, "pinned": True},
            "conversation": {"max_tokens": 9000, "priority": 0.5},
            "rag_context": {"max_tokens": 3500, "priority": 0.5},
            "tools": {"max_tokens": 1500, "priority": 0.2},
        }

    return ContexoConfig(
        storage=StorageConfig(backend_type="in_memory"),
        embeddings=EmbeddingConfig(provider_type="mock"),
        working_memory=WorkingMemoryConfig(
            max_tokens=max_tokens,
            sections=sections or default_sections,
            strategy="sliding_window",
        ),
        auto_initialize=True,
        enable_provenance=True,
    )
