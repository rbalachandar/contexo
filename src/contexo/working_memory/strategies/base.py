"""Base compaction strategy protocol."""

from __future__ import annotations

from typing import Protocol

from contexo.core.memory import MemoryEntry


class CompactionStrategy(Protocol):
    """Protocol defining the interface for context compaction strategies.

    Compaction strategies determine which entries to evict from working
    memory when the context window is full.
    """

    async def select_for_eviction(
        self, entries: list[MemoryEntry], target_count: int
    ) -> list[MemoryEntry]:
        """Select entries to evict from the context window.

        Args:
            entries: Current entries in the context window
            target_count: Number of entries to select for eviction

        Returns:
            List of entries selected for eviction
        """
        ...

    async def compact(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Compact a list of entries into a summarized form.

        This method is called after selecting entries for eviction.
        It can combine multiple entries into a single summarized entry.

        Args:
            entries: Entries to compact

        Returns:
            List of new entries (typically a single summary entry)
        """
        ...

    @property
    def name(self) -> str:
        """Get the name of this strategy."""
        ...


class PassthroughStrategy(CompactionStrategy):
    """A passthrough strategy that doesn't compact anything.

    This is useful for testing or when you want full control over
    context window management.
    """

    @property
    def name(self) -> str:
        """Get the name of this strategy."""
        return "passthrough"

    async def select_for_eviction(
        self, entries: list[MemoryEntry], target_count: int
    ) -> list[MemoryEntry]:
        """Select oldest entries for eviction.

        Args:
            entries: Current entries in the context window
            target_count: Number of entries to select

        Returns:
            List of oldest entries
        """
        return entries[:target_count]

    async def compact(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Return entries as-is without compacting.

        Args:
            entries: Entries to compact

        Returns:
            The same entries (no compaction)
        """
        return entries
