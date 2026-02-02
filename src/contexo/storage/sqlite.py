"""SQLite storage backend with FTS5 full-text search support."""

from __future__ import annotations

import json
import logging
import pickle
import re
from typing import Any

import aiosqlite

from contexo.core.exceptions import StorageError
from contexo.core.memory import EntryType, MemoryEntry
from contexo.storage.base import (
    SearchQuery,
    SearchResult,
    StorageBackend,
)

logger = logging.getLogger(__name__)


def _escape_fts5(query: str) -> str:
    """Escape query for FTS5 MATCH syntax.

    FTS5 treats certain characters as operators. We remove them
    and use token-based search (not phrase search) for better recall.

    WARNING: Uses string interpolation for MATCH because FTS5 doesn't
    support parameter binding for MATCH queries. The query is sanitized
    by removing all non-alphanumeric characters except spaces and underscores.

    Args:
        query: The raw search query

    Returns:
        Sanitized query string safe for FTS5 MATCH, or first 50 chars
        if sanitization results in empty string
    """
    # Remove all non-alphanumeric characters except basic safe punctuation
    # Keep only: letters, numbers, spaces, underscore
    # Remove colons to avoid column:term syntax interpretation
    cleaned = re.sub(r'[^\w\s_]', ' ', query)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Fallback if query becomes empty
    if not cleaned:
        cleaned = query[:50]  # Use first 50 chars as fallback
    return cleaned


class SQLiteStorage(StorageBackend):
    """SQLite storage backend with FTS5 full-text search support.

    This backend stores entries in a SQLite database with full-text
    search capabilities using FTS5.
    """

    # Default multiplier for FTS fetch (extra candidates for min_score filtering)
    DEFAULT_FTS_FETCH_MULTIPLIER: int = 5

    def __init__(self, db_path: str = ":memory:", fts_fetch_multiplier: int | None = None) -> None:
        """Initialize the SQLite storage backend.

        Args:
            db_path: Path to the SQLite database file. Use ":memory:" for in-memory.
            fts_fetch_multiplier: Multiplier for FTS candidate retrieval (default: 5).
                Higher values (e.g., 10) help with min_score filtering but are slower.
                Production: use 5, Benchmarks: use 10.
        """
        self.db_path = db_path
        self.fts_fetch_multiplier = fts_fetch_multiplier or self.DEFAULT_FTS_FETCH_MULTIPLIER
        self._conn: aiosqlite.Connection | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the storage backend and create tables."""
        if self._initialized:
            return

        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row

        # Enable foreign keys
        await self._conn.execute("PRAGMA foreign_keys = ON")

        # Create main entries table
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                entry_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                importance_score REAL NOT NULL DEFAULT 0.5,
                embedding BLOB,
                parent_id TEXT,
                conversation_id TEXT,
                FOREIGN KEY (parent_id) REFERENCES entries(id) ON DELETE SET NULL
            )
            """)

        # Create FTS5 virtual table for full-text search
        await self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                docid, id, content, metadata
            )
            """)

        # Create triggers to keep FTS index updated
        await self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
                INSERT INTO entries_fts(docid, id, content, metadata)
                VALUES (new.rowid, new.id, new.content, new.metadata);
            END
            """)

        await self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
                DELETE FROM entries_fts WHERE docid = old.rowid;
            END
            """)

        await self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
                UPDATE entries_fts SET content = new.content, metadata = new.metadata
                WHERE docid = new.rowid;
            END
            """)

        # Create indexes for common queries
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_conversation ON entries(conversation_id)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_parent ON entries(parent_id)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_timestamp ON entries(timestamp)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_type ON entries(entry_type)"
        )

        await self._conn.commit()
        self._initialized = True

    async def close(self) -> None:
        """Close the storage backend."""
        if self._conn:
            await self._conn.close()
            self._conn = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the storage backend has been initialized."""
        return self._initialized and self._conn is not None

    async def save(self, entry: MemoryEntry) -> MemoryEntry:
        """Save a memory entry.

        Args:
            entry: The entry to save

        Returns:
            The saved entry
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        embedding_blob = None
        if entry.embedding:
            embedding_blob = pickle.dumps(entry.embedding)

        metadata_json = json.dumps(entry.metadata) if entry.metadata else None

        await self._conn.execute(  # type: ignore
            """
            INSERT OR REPLACE INTO entries
            (id, entry_type, content, metadata, timestamp, token_count,
             importance_score, embedding, parent_id, conversation_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.entry_type.value,
                entry.content,
                metadata_json,
                entry.timestamp,
                entry.token_count,
                entry.importance_score,
                embedding_blob,
                entry.parent_id,
                entry.conversation_id,
            ),
        )
        await self._conn.commit()

        return entry

    async def load(self, entry_id: str) -> MemoryEntry | None:
        """Load a memory entry by ID.

        Args:
            entry_id: The ID of the entry to load

        Returns:
            The entry if found, None otherwise
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        cursor = await self._conn.execute(  # type: ignore
            "SELECT * FROM entries WHERE id = ?", (entry_id,)
        )
        row = await cursor.fetchone()

        if row is None:
            return None

        return self._row_to_entry(row)

    async def update(self, entry: MemoryEntry) -> MemoryEntry:
        """Update an existing memory entry.

        Args:
            entry: The entry to update

        Returns:
            The updated entry
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        # Check if entry exists
        existing = await self.load(entry.id)
        if existing is None:
            raise StorageError(f"Entry not found: {entry.id}")

        await self.save(entry)
        return entry

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID.

        Args:
            entry_id: The ID of the entry to delete

        Returns:
            True if the entry was deleted, False if it didn't exist
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        cursor = await self._conn.execute(  # type: ignore
            "DELETE FROM entries WHERE id = ?", (entry_id,)
        )
        await self._conn.commit()

        return cursor.rowcount > 0

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for entries using FTS5 full-text search.

        Args:
            query: The search query

        Returns:
            List of search results sorted by relevance
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        # Escape query for FTS5 MATCH syntax
        fts_query = _escape_fts5(query.query)

        # Use string interpolation for MATCH (FTS5 doesn't support param binding)
        # Safe because _escape_fts5() removes all special characters except spaces/underscores
        # Use subquery approach instead of JOIN for better FTS5 compatibility
        where_clauses = []
        params: list[Any] = []

        if query.entry_type is not None:
            where_clauses.append("e.entry_type = ?")
            params.append(query.entry_type.value)

        if query.conversation_id is not None:
            where_clauses.append("e.conversation_id = ?")
            params.append(query.conversation_id)

        if query.parent_id is not None:
            where_clauses.append("e.parent_id = ?")
            params.append(query.parent_id)

        where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""

        # Get matching IDs with BM25 scores first, then join with entries
        # Get extra candidates since we filter by min_score after scoring
        candidate_limit = query.limit * self.fts_fetch_multiplier

        sql = f"""
            SELECT e.*, fts.rank
            FROM entries e
            INNER JOIN (
                SELECT id, bm25(entries_fts) as rank
                FROM entries_fts
                WHERE entries_fts MATCH '{fts_query}'
                LIMIT ?
            ) fts ON e.id = fts.id
            WHERE 1=1 {where_sql}
            ORDER BY fts.rank
        """
        params.insert(0, candidate_limit)

        cursor = await self._conn.execute(sql, params)  # type: ignore
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            entry = self._row_to_entry(row)
            # Convert BM25 score (negative) to positive relevance score
            bm25_score = row["rank"]
            relevance = 1.0 / (1.0 + abs(bm25_score))
            # Apply importance boost
            relevance = relevance * (0.5 + entry.importance_score * 0.5)
            results.append(SearchResult(entry=entry, score=relevance))

        # Apply minimum score filter
        if query.min_score > 0:
            results = [r for r in results if r.score >= query.min_score]

        # Return at most query.limit results, after filtering
        return results[:query.limit]

    async def list_collections(self) -> list[str]:
        """List all available conversation IDs as collections.

        Returns:
            List of conversation IDs
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        cursor = await self._conn.execute(  # type: ignore
            "SELECT DISTINCT conversation_id FROM entries WHERE conversation_id IS NOT NULL"
        )
        rows = await cursor.fetchall()

        return [row[0] for row in rows]

    async def count(
        self,
        collection: str | None = None,
        entry_type: EntryType | None = None,
    ) -> int:
        """Count entries in storage.

        Args:
            collection: Filter by conversation ID
            entry_type: Filter by entry type

        Returns:
            The count of matching entries
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        sql = "SELECT COUNT(*) FROM entries WHERE 1=1"
        params: list[Any] = []

        if collection is not None:
            sql += " AND conversation_id = ?"
            params.append(collection)

        if entry_type is not None:
            sql += " AND entry_type = ?"
            params.append(entry_type.value)

        cursor = await self._conn.execute(sql, params)  # type: ignore
        row = await cursor.fetchone()

        return row[0] if row else 0

    async def list_entries(
        self,
        collection: str | None = None,
        entry_type: EntryType | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List entries from storage.

        Args:
            collection: Filter by conversation ID
            entry_type: Filter by entry type
            limit: Maximum number of entries to return
            offset: Number of entries to skip

        Returns:
            List of entries
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        sql = "SELECT * FROM entries WHERE 1=1"
        params: list[Any] = []

        if collection is not None:
            sql += " AND conversation_id = ?"
            params.append(collection)

        if entry_type is not None:
            sql += " AND entry_type = ?"
            params.append(entry_type.value)

        sql += " ORDER BY timestamp"

        if offset:
            sql += " OFFSET ?"
            params.append(offset)

        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        cursor = await self._conn.execute(sql, params)  # type: ignore
        rows = await cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def clear_collection(self, collection: str) -> None:
        """Clear all entries from a conversation.

        Args:
            collection: The conversation ID to clear
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        await self._conn.execute(  # type: ignore
            "DELETE FROM entries WHERE conversation_id = ?", (collection,)
        )
        await self._conn.commit()

    async def delete_collection(self, collection: str) -> None:
        """Delete a conversation and all its entries.

        Args:
            collection: The conversation ID to delete
        """
        await self.clear_collection(collection)

    def _row_to_entry(self, row: aiosqlite.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry.

        Args:
            row: The database row

        Returns:
            A MemoryEntry instance
        """
        embedding = None
        if row["embedding"]:
            try:
                embedding = pickle.loads(row["embedding"])
            except (pickle.PickleError, TypeError) as e:
                logger.warning(f"Failed to deserialize embedding for entry {row.get('id', 'unknown')}: {e}")

        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to deserialize metadata for entry {row.get('id', 'unknown')}: {e}")

        return MemoryEntry(
            id=row["id"],
            entry_type=EntryType(row["entry_type"]),
            content=row["content"],
            metadata=metadata,
            timestamp=row["timestamp"],
            token_count=row["token_count"],
            importance_score=row["importance_score"],
            embedding=embedding,
            parent_id=row["parent_id"],
            conversation_id=row["conversation_id"],
        )

    @property
    def supports_semantic_search(self) -> bool:
        """Check if this backend supports semantic search."""
        return False  # FTS5 is lexical, not semantic

    @property
    def supports_fts(self) -> bool:
        """Check if this backend supports full-text search."""
        return True
