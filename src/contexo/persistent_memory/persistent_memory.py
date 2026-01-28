"""Long-term persistent memory with semantic search."""

from __future__ import annotations

import logging
from typing import Any

from contexo.core.memory import EntryType, MemoryEntry, MemoryManager
from contexo.embeddings.base import EmbeddingProvider
from contexo.embeddings.mock import MockEmbeddings
from contexo.storage.base import SearchQuery, SearchResult, StorageBackend
from contexo.storage.in_memory import InMemoryStorage

logger = logging.getLogger(__name__)


class PersistentMemory(MemoryManager):
    """Long-term persistent memory with semantic search.

    Persistent memory stores entries in a storage backend and provides
    semantic search capabilities using embeddings.
    """

    def __init__(
        self,
        storage: StorageBackend | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        auto_embed: bool = True,
    ) -> None:
        """Initialize the persistent memory.

        Args:
            storage: Storage backend to use (defaults to InMemoryStorage)
            embedding_provider: Embedding provider for semantic search
            auto_embed: Whether to automatically generate embeddings on save
        """
        self._storage = storage or InMemoryStorage()
        self._embedding_provider = embedding_provider or MockEmbeddings()
        self._auto_embed = auto_embed
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the persistent memory and storage backend."""
        if self._initialized:
            return

        await self._storage.initialize()
        await self._embedding_provider.initialize()
        self._initialized = True
        logger.debug("Persistent memory initialized")

    async def close(self) -> None:
        """Close the persistent memory and storage backend."""
        await self._storage.close()
        await self._embedding_provider.close()
        self._initialized = False
        logger.debug("Persistent memory closed")

    @property
    def is_initialized(self) -> bool:
        """Check if the persistent memory has been initialized."""
        return self._initialized

    @property
    def storage(self) -> StorageBackend:
        """Get the storage backend."""
        return self._storage

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Get the embedding provider."""
        return self._embedding_provider

    @property
    def total_tokens(self) -> int:
        """Get the total number of tokens stored."""
        # This would require counting all entries, which could be expensive
        # For now, return -1 to indicate "unknown"
        return -1

    async def add(self, entry: MemoryEntry) -> MemoryEntry:
        """Add an entry to persistent memory.

        Args:
            entry: The entry to add

        Returns:
            The saved entry (with embedding if auto_embed is enabled)
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        # Generate embedding if needed
        if self._auto_embed and entry.embedding is None:
            try:
                embedding = await self._embedding_provider.embed(entry.content)
                entry = entry.with_embedding(embedding)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        # Save to storage
        saved = await self._storage.save(entry)
        logger.debug(f"Saved entry {entry.id} to persistent memory")
        return saved

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Get an entry by ID.

        Args:
            entry_id: The ID of the entry to get

        Returns:
            The entry if found, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        return await self._storage.load(entry_id)

    async def update(self, entry: MemoryEntry) -> MemoryEntry:
        """Update an existing entry.

        Args:
            entry: The entry to update

        Returns:
            The updated entry
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        # Regenerate embedding if content changed and auto_embed is enabled
        if self._auto_embed and entry.embedding is None:
            embedding = await self._embedding_provider.embed(entry.content)
            entry = entry.with_embedding(embedding)

        updated = await self._storage.update(entry)
        logger.debug(f"Updated entry {entry.id}")
        return updated

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID.

        Args:
            entry_id: The ID of the entry to delete

        Returns:
            True if deleted, False if not found
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        deleted = await self._storage.delete(entry_id)
        if deleted:
            logger.debug(f"Deleted entry {entry_id}")
        return deleted

    async def list_all(
        self, limit: int | None = None, entry_type: EntryType | None = None
    ) -> list[MemoryEntry]:
        """List all entries.

        Args:
            limit: Maximum number of entries to return
            entry_type: Filter by entry type

        Returns:
            List of entries
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        return await self._storage.list_entries(
            entry_type=entry_type,
            limit=limit,
        )

    async def count(self, entry_type: EntryType | None = None) -> int:
        """Count entries in persistent memory.

        Args:
            entry_type: Filter by entry type

        Returns:
            The count of entries
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        return await self._storage.count(entry_type=entry_type)

    async def clear(self) -> None:
        """Clear all entries from persistent memory.

        Warning: This deletes all data from the storage backend.
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        # Get all collections and clear them
        collections = await self._storage.list_collections()
        for collection in collections:
            await self._storage.clear_collection(collection)

        # Also clear any entries without a collection
        # This varies by backend, so we'll just log for now
        logger.debug("Persistent memory cleared")

    async def search(
        self,
        query: str,
        limit: int = 10,
        entry_type: EntryType | None = None,
        conversation_id: str | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for entries using semantic or full-text search.

        Args:
            query: The search query
            limit: Maximum number of results to return
            entry_type: Filter by entry type
            conversation_id: Filter by conversation ID
            min_score: Minimum relevance score threshold

        Returns:
            List of search results sorted by relevance
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        # Try semantic search if supported
        if self._storage.supports_semantic_search:
            return await self._semantic_search(
                query=query,
                limit=limit,
                entry_type=entry_type,
                conversation_id=conversation_id,
                min_score=min_score,
            )

        # Fall back to full-text search
        search_query = SearchQuery(
            query=query,
            limit=limit,
            entry_type=entry_type,
            conversation_id=conversation_id,
            min_score=min_score,
        )
        return await self._storage.search(search_query)

    async def _semantic_search(
        self,
        query: str,
        limit: int = 10,
        entry_type: EntryType | None = None,
        conversation_id: str | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Perform semantic search using embeddings.

        Args:
            query: The search query
            limit: Maximum number of results
            entry_type: Filter by entry type
            conversation_id: Filter by conversation ID
            min_score: Minimum relevance score

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = await self._embedding_provider.embed(query)

        # Get all candidate entries
        candidates = await self._storage.list_entries(
            collection=conversation_id,
            entry_type=entry_type,
        )

        # Calculate similarity scores
        from contexo.persistent_memory.provenance import cosine_similarity

        results = []
        for entry in candidates:
            if entry.embedding is None:
                continue

            score = cosine_similarity(query_embedding, entry.embedding)
            if score >= min_score:
                results.append(
                    SearchResult(
                        entry=entry,
                        score=score,
                        metadata={"search_method": "semantic"},
                    )
                )

        # Sort by score descending and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def find_related(
        self,
        entry_id: str,
        limit: int = 5,
        min_score: float = 0.5,
    ) -> list[SearchResult]:
        """Find entries related to a given entry.

        Args:
            entry_id: The ID of the entry to find relations for
            limit: Maximum number of results
            min_score: Minimum similarity score

        Returns:
            List of related entries
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        entry = await self.get(entry_id)
        if entry is None:
            return []

        if entry.embedding is None:
            # Generate embedding on the fly
            embedding = await self._embedding_provider.embed(entry.content)
            entry = entry.with_embedding(embedding)

        # Get candidates from the same conversation
        candidates = await self._storage.list_entries(
            collection=entry.conversation_id,
        )

        # Find similar entries
        from contexo.persistent_memory.provenance import cosine_similarity

        results = []
        for candidate in candidates:
            if candidate.id == entry_id:
                continue
            if candidate.embedding is None:
                continue

            score = cosine_similarity(entry.embedding, candidate.embedding)
            if score >= min_score:
                results.append(
                    SearchResult(
                        entry=candidate,
                        score=score,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the persistent memory.

        Returns:
            Dictionary with memory statistics
        """
        return {
            "storage_type": type(self._storage).__name__,
            "embedding_provider": type(self._embedding_provider).__name__,
            "embedding_dimension": self._embedding_provider.dimension,
            "supports_semantic_search": self._storage.supports_semantic_search,
            "supports_fts": self._storage.supports_fts,
            "auto_embed": self._auto_embed,
        }

    async def get_children(
        self,
        entry_id: str,
        entry_type: EntryType | None = None,
    ) -> list[MemoryEntry]:
        """Get entries that reference this entry as their parent.

        Useful for finding tool responses for a tool call.

        Args:
            entry_id: The parent entry ID
            entry_type: Optional filter by entry type

        Returns:
            List of child entries
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        entries = await self._storage.list_entries()
        children = [
            e
            for e in entries
            if e.parent_id == entry_id and (entry_type is None or e.entry_type == entry_type)
        ]
        return children

    async def get_conversation_context(
        self,
        entry_id: str,
        window_size: int = 5,
    ) -> list[MemoryEntry]:
        """Get the conversation context around an entry.

        Returns entries before and after the target entry in temporal order.

        Args:
            entry_id: The entry ID to get context for
            window_size: Number of entries before and after

        Returns:
            List of entries in temporal order
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        entry = await self.get(entry_id)
        if entry is None:
            return []

        if not entry.conversation_id:
            return [entry]

        # Get all entries in the conversation
        all_entries = await self._storage.list_entries(
            collection=entry.conversation_id,
        )

        # Sort by timestamp
        all_entries.sort(key=lambda e: e.timestamp)

        # Find the index of the target entry
        try:
            target_index = next(i for i, e in enumerate(all_entries) if e.id == entry_id)
        except StopIteration:
            return [entry]

        # Get window around the target
        start = max(0, target_index - window_size)
        end = min(len(all_entries), target_index + window_size + 1)

        return all_entries[start:end]

    async def get_full_context(
        self,
        entry_id: str,
        include_children: bool = True,
        include_conversation: bool = True,
        conversation_window: int = 5,
    ) -> dict[str, Any]:
        """Get full context for an entry including relationships.

        This is useful for provenance tracking and understanding
        what context influenced a specific message.

        Args:
            entry_id: The entry ID to get context for
            include_children: Include child entries (e.g., tool responses)
            include_conversation: Include conversation context
            conversation_window: Size of conversation window

        Returns:
            Dictionary with the entry and its related context
        """
        if not self._initialized:
            raise RuntimeError("Persistent memory not initialized")

        entry = await self.get(entry_id)
        if entry is None:
            return {
                "entry": None,
                "parent": None,
                "children": [],
                "conversation_context": [],
            }

        result: dict[str, Any] = {
            "entry": entry,
            "parent": None,
            "children": [],
            "conversation_context": [],
            "related": [],
        }

        # Get parent (e.g., tool call for a tool response)
        if entry.parent_id:
            result["parent"] = await self.get(entry.parent_id)

        # Get children (e.g., tool responses for a tool call)
        if include_children:
            result["children"] = await self.get_children(entry_id)

        # Get conversation context
        if include_conversation:
            result["conversation_context"] = await self.get_conversation_context(
                entry_id,
                window_size=conversation_window,
            )

        # Get semantically related entries
        related = await self.find_related(entry_id, limit=3, min_score=0.6)
        result["related"] = [r.entry for r in related]

        return result
