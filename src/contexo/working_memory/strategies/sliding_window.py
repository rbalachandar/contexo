"""Sliding window compaction strategy - FIFO eviction."""

from __future__ import annotations

from contexo.core.memory import MemoryEntry
from contexo.working_memory.strategies.base import PassthroughStrategy


class SlidingWindowStrategy(PassthroughStrategy):
    """Sliding window strategy using FIFO eviction.

    This strategy evicts the oldest entries first without summarization.
    It's simple, predictable, and preserves the most recent context.
    """

    @property
    def name(self) -> str:
        """Get the name of this strategy."""
        return "sliding_window"

    async def select_for_eviction(
        self, entries: list[MemoryEntry], target_count: int
    ) -> list[MemoryEntry]:
        """Select the oldest entries for eviction (FIFO).

        Args:
            entries: Current entries in the context window (sorted by timestamp)
            target_count: Number of entries to select

        Returns:
            List of oldest entries
        """
        # Entries should already be sorted by timestamp, so take from the start
        return entries[:target_count]
