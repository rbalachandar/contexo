"""Configuration presets and settings for Contexo."""

from contexo.config.defaults import (
    chat_config,
    cloud_config,
    development_config,
    graphdb_config,
    local_config,
    minimal_config,
    production_config,
)
from contexo.config.settings import (
    ContexoConfig,
    EmbeddingConfig,
    RetrievalConfig,
    StorageConfig,
    WorkingMemoryConfig,
)

__all__ = [
    # Presets
    "minimal_config",
    "local_config",
    "cloud_config",
    "development_config",
    "production_config",
    "graphdb_config",
    "chat_config",
    # Settings
    "ContexoConfig",
    "StorageConfig",
    "EmbeddingConfig",
    "WorkingMemoryConfig",
    "RetrievalConfig",
]
