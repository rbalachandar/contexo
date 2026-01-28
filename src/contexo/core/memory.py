"""Core data structures for memory management."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

from contexo.core.exceptions import ValidationError


class EntryType(Enum):
    """Types of memory entries."""

    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    SYSTEM = "system"
    SUMMARIZED = "summarized"


@dataclass(frozen=True)
class MemoryEntry:
    """A single entry in memory.

    Attributes:
        id: Unique identifier for this entry
        entry_type: The type of entry (message, tool call, etc.)
        content: The main content of the entry
        metadata: Additional metadata associated with the entry
        timestamp: Unix timestamp when the entry was created
        token_count: Estimated number of tokens in the entry
        importance_score: Score from 0.0 to 1.0 indicating importance
        embedding: Vector embedding for semantic search (optional)
        parent_id: ID of the parent entry (e.g., tool_call -> tool_response)
        conversation_id: ID of the conversation this entry belongs to
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    entry_type: EntryType = EntryType.MESSAGE
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    importance_score: float = 0.5
    embedding: list[float] | None = None
    parent_id: str | None = None
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the memory entry after creation."""
        if not 0.0 <= self.importance_score <= 1.0:
            raise ValidationError(
                f"Importance score must be between 0.0 and 1.0, got {self.importance_score}"
            )
        if self.token_count < 0:
            raise ValidationError(f"Token count must be non-negative, got {self.token_count}")
        if self.embedding is not None and len(self.embedding) == 0:
            raise ValidationError("Embedding must be None or non-empty")

    def with_embedding(self, embedding: list[float]) -> "MemoryEntry":
        """Return a new MemoryEntry with the given embedding.

        Since MemoryEntry is frozen, we need to create a new instance.
        """
        # Create a new dict of fields, preserving everything except embedding
        fields = {
            "id": self.id,
            "entry_type": self.entry_type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "importance_score": self.importance_score,
            "embedding": embedding,
            "parent_id": self.parent_id,
            "conversation_id": self.conversation_id,
        }
        return MemoryEntry(**fields)

    def with_importance(self, importance: float) -> "MemoryEntry":
        """Return a new MemoryEntry with the given importance score."""
        fields = {
            "id": self.id,
            "entry_type": self.entry_type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "importance_score": importance,
            "embedding": self.embedding,
            "parent_id": self.parent_id,
            "conversation_id": self.conversation_id,
        }
        return MemoryEntry(**fields)


class MemoryManager(Protocol):
    """Protocol defining the interface for memory management.

    Both WorkingMemory and PersistentMemory implement this protocol.
    """

    async def add(self, entry: MemoryEntry) -> MemoryEntry:
        """Add an entry to memory.

        Args:
            entry: The entry to add

        Returns:
            The added entry (potentially modified with ID, embedding, etc.)
        """
        ...

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve an entry by ID.

        Args:
            entry_id: The ID of the entry to retrieve

        Returns:
            The entry if found, None otherwise
        """
        ...

    async def update(self, entry: MemoryEntry) -> MemoryEntry:
        """Update an existing entry.

        Args:
            entry: The entry to update (must have an existing ID)

        Returns:
            The updated entry
        """
        ...

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID.

        Args:
            entry_id: The ID of the entry to delete

        Returns:
            True if the entry was deleted, False if it didn't exist
        """
        ...

    async def list_all(
        self, limit: int | None = None, entry_type: EntryType | None = None
    ) -> list[MemoryEntry]:
        """List all entries, optionally filtered.

        Args:
            limit: Maximum number of entries to return
            entry_type: Filter by entry type

        Returns:
            List of entries
        """
        ...

    async def count(self, entry_type: EntryType | None = None) -> int:
        """Count the number of entries.

        Args:
            entry_type: Filter by entry type

        Returns:
            The count of entries
        """
        ...

    async def clear(self) -> None:
        """Clear all entries from memory."""
        ...

    async def initialize(self) -> None:
        """Initialize the memory manager.

        This should be called before using the memory manager.
        """
        ...

    async def close(self) -> None:
        """Close the memory manager and release resources."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if the memory manager has been initialized."""
        ...

    @property
    def total_tokens(self) -> int:
        """Get the total number of tokens stored."""
        ...
