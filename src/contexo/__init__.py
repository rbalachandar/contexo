"""Contexo - LLM Context Management Library.

This library provides a two-tier memory hierarchy for managing LLM context:
- Short-term Memory (Working Memory): First-level cache with compaction
- Long-term Memory (Persistent Storage): Second-level storage with search

Example:
    ```python
    from contexo import Contexo, minimal_config

    ctx = Contexo(config=minimal_config())
    await ctx.initialize()

    await ctx.add_message("Hello, world!", role="user")
    context = await ctx.get_context()

    await ctx.close()
    ```
"""

from __future__ import annotations

# Main exports
from contexo.config.defaults import (
    chat_config,
    cloud_config,
    development_config,
    graphdb_config,
    local_config,
    minimal_config,
    production_config,
)
from contexo.config.settings import ContexoConfig
from contexo.contexo import Contexo
from contexo.core.memory import EntryType, MemoryEntry

__all__ = [
    "Contexo",
    "ContexoConfig",
    "MemoryEntry",
    "EntryType",
    "minimal_config",
    "local_config",
    "cloud_config",
    "development_config",
    "production_config",
    "graphdb_config",
    "chat_config",
]
