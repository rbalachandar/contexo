"""Base storage backend protocol and related types."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from contexo.core.memory import EntryType, MemoryEntry


@dataclass
class SearchResult:
    """A result from a storage search query.

    Attributes:
        entry: The matching memory entry
        score: Relevance score (higher is more relevant)
        metadata: Additional metadata about the search result
    """

    entry: MemoryEntry
    score: float
    metadata: dict[str, Any] | None = None


@dataclass
class SearchQuery:
    """A query for searching stored entries.

    Attributes:
        query: The search query text
        limit: Maximum number of results to return
        entry_type: Filter by entry type
        conversation_id: Filter by conversation ID
        parent_id: Filter by parent ID
        min_score: Minimum relevance score threshold
        metadata_filter: Additional metadata filters as key-value pairs
    """

    query: str
    limit: int = 10
    entry_type: EntryType | None = None
    conversation_id: str | None = None
    parent_id: str | None = None
    min_score: float = 0.0
    metadata_filter: dict[str, Any] | None = None


class StorageBackend(Protocol):
    """Protocol defining the interface for storage backends.

    Storage backends provide persistent storage for memory entries,
    with support for CRUD operations and semantic search.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend.

        This should be called before using the backend.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend and release resources."""
        ...

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the storage backend has been initialized."""
        ...

    @abstractmethod
    async def save(self, entry: MemoryEntry) -> MemoryEntry:
        """Save a memory entry.

        Args:
            entry: The entry to save

        Returns:
            The saved entry (possibly with generated ID or embedding)
        """
        ...

    @abstractmethod
    async def load(self, entry_id: str) -> MemoryEntry | None:
        """Load a memory entry by ID.

        Args:
            entry_id: The ID of the entry to load

        Returns:
            The entry if found, None otherwise
        """
        ...

    @abstractmethod
    async def update(self, entry: MemoryEntry) -> MemoryEntry:
        """Update an existing memory entry.

        Args:
            entry: The entry to update (must have an existing ID)

        Returns:
            The updated entry
        """
        ...

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID.

        Args:
            entry_id: The ID of the entry to delete

        Returns:
            True if the entry was deleted, False if it didn't exist
        """
        ...

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for entries matching a query.

        Args:
            query: The search query

        Returns:
            List of search results sorted by relevance
        """
        ...

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """List all available collections.

        Collections can be used to group entries, e.g., by conversation ID.

        Returns:
            List of collection names/IDs
        """
        ...

    @abstractmethod
    async def count(
        self,
        collection: str | None = None,
        entry_type: EntryType | None = None,
    ) -> int:
        """Count entries in storage.

        Args:
            collection: Filter by collection (e.g., conversation ID)
            entry_type: Filter by entry type

        Returns:
            The count of matching entries
        """
        ...

    @abstractmethod
    async def list_entries(
        self,
        collection: str | None = None,
        entry_type: EntryType | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List entries from storage.

        Args:
            collection: Filter by collection
            entry_type: Filter by entry type
            limit: Maximum number of entries to return
            offset: Number of entries to skip

        Returns:
            List of entries
        """
        ...

    @abstractmethod
    async def clear_collection(self, collection: str) -> None:
        """Clear all entries from a collection.

        Args:
            collection: The collection to clear
        """
        ...

    @abstractmethod
    async def delete_collection(self, collection: str) -> None:
        """Delete a collection and all its entries.

        Args:
            collection: The collection to delete
        """
        ...

    @property
    @abstractmethod
    def supports_semantic_search(self) -> bool:
        """Check if this backend supports semantic search."""
        ...

    @property
    @abstractmethod
    def supports_fts(self) -> bool:
        """Check if this backend supports full-text search."""
        ...
