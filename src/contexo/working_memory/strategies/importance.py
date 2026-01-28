"""Importance-based compaction strategy - evict least important entries."""

from __future__ import annotations

from typing import Any

from contexo.core.memory import MemoryEntry
from contexo.working_memory.strategies.base import PassthroughStrategy


class ImportanceStrategy(PassthroughStrategy):
    """Importance-based strategy that evicts least important entries first.

    Entries are scored by importance (0.0 to 1.0) and those with the
    lowest scores are evicted first. This allows important information
    to be retained longer.
    """

    def __init__(self, score_fn: Any | None = None, recency_bias: float = 0.1) -> None:
        """Initialize the importance strategy.

        Args:
            score_fn: Optional function to calculate importance scores.
                If None, uses the entry's existing importance_score.
                Signature: (entry: MemoryEntry) -> float
            recency_bias: Weight for recency in scoring (0.0 to 1.0).
                Higher values prefer more recent entries.
        """
        self._score_fn = score_fn
        self._recency_bias = recency_bias

    @property
    def name(self) -> str:
        """Get the name of this strategy."""
        return "importance"

    async def select_for_eviction(
        self, entries: list[MemoryEntry], target_count: int
    ) -> list[MemoryEntry]:
        """Select the least important entries for eviction.

        Args:
            entries: Current entries in the context window
            target_count: Number of entries to select

        Returns:
            List of least important entries
        """
        if target_count >= len(entries):
            return entries

        # Calculate combined scores
        scored_entries = []
        max_timestamp = max((e.timestamp for e in entries), default=0)
        min_timestamp = min((e.timestamp for e in entries), default=0)
        timestamp_range = max_timestamp - min_timestamp if max_timestamp > min_timestamp else 1.0

        for entry in entries:
            # Get base importance score
            if self._score_fn is not None:
                importance = await self._call_score_fn(entry)
            else:
                importance = entry.importance_score

            # Apply recency bias (newer entries get boosted)
            if self._recency_bias > 0 and timestamp_range > 0:
                recency_score = (entry.timestamp - min_timestamp) / timestamp_range
                combined_score = (
                    importance * (1.0 - self._recency_bias) + recency_score * self._recency_bias
                )
            else:
                combined_score = importance

            scored_entries.append((combined_score, entry))

        # Sort by score ascending (lowest first) and select for eviction
        scored_entries.sort(key=lambda x: x[0])
        return [entry for _, entry in scored_entries[:target_count]]

    async def _call_score_fn(self, entry: MemoryEntry) -> float:
        """Call the custom scoring function if it's async.

        Args:
            entry: Entry to score

        Returns:
            Importance score
        """
        import inspect

        if self._score_fn is None:
            return entry.importance_score

        if inspect.iscoroutinefunction(self._score_fn):
            return await self._score_fn(entry)
        else:
            return self._score_fn(entry)  # type: ignore
