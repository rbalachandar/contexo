"""In-memory storage backend for testing and development."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from contexo.core.exceptions import StorageError
from contexo.core.memory import EntryType, MemoryEntry
from contexo.storage.base import (
    SearchQuery,
    SearchResult,
    StorageBackend,
)


class InMemoryStorage(StorageBackend):
    """In-memory storage backend for testing and development.

    This backend stores all entries in memory and provides basic
    search functionality. Data is lost when the backend is closed.
    """

    def __init__(self) -> None:
        """Initialize the in-memory storage."""
        self._lock = asyncio.Lock()
        self._entries: dict[str, MemoryEntry] = {}
        self._entries_by_collection: dict[str, list[str]] = defaultdict(list)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the storage backend."""
        async with self._lock:
            if self._initialized:
                return
            self._entries.clear()
            self._entries_by_collection.clear()
            self._initialized = True

    async def close(self) -> None:
        """Close the storage backend and clear all data."""
        async with self._lock:
            self._entries.clear()
            self._entries_by_collection.clear()
            self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the storage backend has been initialized."""
        return self._initialized

    async def save(self, entry: MemoryEntry) -> MemoryEntry:
        """Save a memory entry.

        Args:
            entry: The entry to save

        Returns:
            The saved entry
        """
        async with self._lock:
            if not self._initialized:
                raise StorageError("Storage backend not initialized")

            self._entries[entry.id] = entry

            # Track by collection (conversation_id)
            if entry.conversation_id:
                if entry.id not in self._entries_by_collection[entry.conversation_id]:
                    self._entries_by_collection[entry.conversation_id].append(entry.id)

            return entry

    async def load(self, entry_id: str) -> MemoryEntry | None:
        """Load a memory entry by ID.

        Args:
            entry_id: The ID of the entry to load

        Returns:
            The entry if found, None otherwise
        """
        async with self._lock:
            return self._entries.get(entry_id)

    async def update(self, entry: MemoryEntry) -> MemoryEntry:
        """Update an existing memory entry.

        Args:
            entry: The entry to update

        Returns:
            The updated entry
        """
        async with self._lock:
            if not self._initialized:
                raise StorageError("Storage backend not initialized")

            if entry.id not in self._entries:
                raise StorageError(f"Entry not found: {entry.id}")

            self._entries[entry.id] = entry
            return entry

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID.

        Args:
            entry_id: The ID of the entry to delete

        Returns:
            True if the entry was deleted, False if it didn't exist
        """
        async with self._lock:
            if entry_id in self._entries:
                entry = self._entries.pop(entry_id)

                # Remove from collection tracking
                if entry.conversation_id:
                    collection = self._entries_by_collection[entry.conversation_id]
                    if entry_id in collection:
                        collection.remove(entry_id)

                return True
            return False

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for entries matching a query.

        This performs a simple substring search on entry content.
        For semantic search, use a backend that supports embeddings.

        Args:
            query: The search query

        Returns:
            List of search results sorted by relevance
        """
        async with self._lock:
            query_lower = query.query.lower()
            results: list[SearchResult] = []

            for entry in self._entries.values():
                # Apply filters
                if query.entry_type is not None and entry.entry_type != query.entry_type:
                    continue
                if query.conversation_id is not None and entry.conversation_id != query.conversation_id:
                    continue
                if query.parent_id is not None and entry.parent_id != query.parent_id:
                    continue

                # Check metadata filters
                if query.metadata_filter:
                    match = True
                    for key, value in query.metadata_filter.items():
                        if entry.metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue

                # Simple relevance scoring based on substring match
                content_lower = entry.content.lower()
                score = 0.0

                if query_lower in content_lower:
                    # Score based on position and match quality
                    if content_lower.startswith(query_lower):
                        score = 1.0
                    elif content_lower.endswith(query_lower):
                        score = 0.9
                    else:
                        score = 0.7

                    # Boost by importance
                    score = score * (0.5 + entry.importance_score * 0.5)

                if score >= query.min_score:
                    results.append(SearchResult(entry=entry, score=score))

            # Sort by score descending
            results.sort(key=lambda r: r.score, reverse=True)

            # Apply limit
            if query.limit:
                results = results[: query.limit]

            return results

    async def list_collections(self) -> list[str]:
        """List all available collections.

        Returns:
            List of collection names/IDs
        """
        async with self._lock:
            return list(self._entries_by_collection.keys())

    async def count(
        self,
        collection: str | None = None,
        entry_type: EntryType | None = None,
    ) -> int:
        """Count entries in storage.

        Args:
            collection: Filter by collection
            entry_type: Filter by entry type

        Returns:
            The count of matching entries
        """
        async with self._lock:
            if collection is None:
                entries = self._entries.values()
            else:
                entry_ids = self._entries_by_collection.get(collection, [])
                entries = [self._entries[eid] for eid in entry_ids if eid in self._entries]

            if entry_type is not None:
                entries = [e for e in entries if e.entry_type == entry_type]

            return len(entries)

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
        async with self._lock:
            if collection is None:
                entries = list(self._entries.values())
            else:
                entry_ids = self._entries_by_collection.get(collection, [])
                entries = [self._entries[eid] for eid in entry_ids if eid in self._entries]

            if entry_type is not None:
                entries = [e for e in entries if e.entry_type == entry_type]

            # Sort by timestamp
            entries.sort(key=lambda e: e.timestamp)

            # Apply offset and limit
            if offset:
                entries = entries[offset:]
            if limit:
                entries = entries[:limit]

            return entries

    async def clear_collection(self, collection: str) -> None:
        """Clear all entries from a collection.

        Args:
            collection: The collection to clear
        """
        async with self._lock:
            entry_ids = self._entries_by_collection.get(collection, [])
            for entry_id in entry_ids:
                if entry_id in self._entries:
                    del self._entries[entry_id]
            self._entries_by_collection[collection].clear()

    async def delete_collection(self, collection: str) -> None:
        """Delete a collection and all its entries.

        Args:
            collection: The collection to delete
        """
        async with self._lock:
            await self.clear_collection(collection)
            del self._entries_by_collection[collection]

    @property
    def supports_semantic_search(self) -> bool:
        """Check if this backend supports semantic search."""
        return False

    @property
    def supports_fts(self) -> bool:
        """Check if this backend supports full-text search."""
        return False
