"""Storage backend implementations and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contexo.storage.base import SearchQuery, SearchResult, StorageBackend
from contexo.storage.in_memory import InMemoryStorage

if TYPE_CHECKING:
    from contexo.storage.graphdb import GraphDBStorage as _GraphDBStorage
    from contexo.storage.sqlite import SQLiteStorage as _SQLiteStorage

try:
    from contexo.storage.sqlite import SQLiteStorage

    _sqlite_available = True
except ImportError:
    _sqlite_available = False

    class SQLiteStorage:  # type: ignore[no-redef]
        """Placeholder when SQLite is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "aiosqlite is required for SQLite storage. Install with: pip install aiosqlite"
            )


try:
    from contexo.storage.graphdb import GraphDBStorage

    _graphdb_available = True
except ImportError:
    _graphdb_available = False

    class GraphDBStorage:  # type: ignore[no-redef]
        """Placeholder when GraphDB is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "neo4j is required for GraphDB storage. Install with: pip install neo4j"
            )


def create_storage(backend_type: str, **kwargs: Any) -> StorageBackend:
    """Create a storage backend by type name.

    Args:
        backend_type: The type of backend to create
            - "in_memory" or "memory": InMemoryStorage
            - "sqlite": SQLiteStorage
            - "graphdb" or "neo4j": GraphDBStorage (Neo4j)
        **kwargs: Additional arguments passed to the backend constructor

    Returns:
        A storage backend instance

    Raises:
        ValueError: If the backend type is unknown
    """
    backend_type = backend_type.lower().replace("-", "_")

    if backend_type in ("in_memory", "memory"):
        # InMemoryStorage doesn't take any kwargs
        return InMemoryStorage()

    if backend_type == "sqlite":
        if not _sqlite_available:
            raise ImportError(
                "aiosqlite is required for SQLite storage. Install with: pip install aiosqlite"
            )
        # SQLiteStorage expects db_path
        return SQLiteStorage(db_path=kwargs.get("db_path", ":memory:"))

    if backend_type in ("graphdb", "neo4j"):
        if not _graphdb_available:
            raise ImportError(
                "neo4j is required for GraphDB storage. Install with: pip install neo4j"
            )
        # GraphDBStorage expects uri, user, password, database
        return GraphDBStorage(
            uri=kwargs.get("uri", "bolt://localhost:7687"),
            user=kwargs.get("user", "neo4j"),
            password=kwargs.get("password", "password"),
            database=kwargs.get("database", "neo4j"),
        )

    raise ValueError(f"Unknown storage backend type: {backend_type}")


__all__ = [
    "SearchQuery",
    "SearchResult",
    "StorageBackend",
    "InMemoryStorage",
    "SQLiteStorage",
    "GraphDBStorage",
    "create_storage",
]
