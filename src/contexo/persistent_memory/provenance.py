"""Provenance tracking for memory operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceEvent:
    """A provenance event tracking a memory operation.

    Attributes:
        timestamp: When the event occurred
        event_type: Type of event (add, delete, compact, search, etc.)
        entry_ids: IDs of entries involved in the event
        metadata: Additional event metadata
        context_snapshot: Optional snapshot of context at event time
    """

    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    event_type: str = "unknown"
    entry_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    context_snapshot: dict[str, Any] | None = None


@dataclass
class ProvenanceTrail:
    """A provenance trail for a single decision or operation.

    Tracks what context was used and what operations were performed.

    Attributes:
        trail_id: Unique identifier for this trail
        start_time: When the trail started
        end_time: When the trail ended
        events: List of events in the trail
        final_context: The final context state
        metadata: Additional metadata about the trail
    """

    trail_id: str
    start_time: float
    end_time: float | None = None
    events: list[ProvenanceEvent] = field(default_factory=list)
    final_context: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: ProvenanceEvent) -> None:
        """Add an event to the trail.

        Args:
            event: The event to add
        """
        self.events.append(event)

    def get_entry_ids(self) -> set[str]:
        """Get all unique entry IDs referenced in the trail.

        Returns:
            Set of entry IDs
        """
        ids: set[str] = set()
        for event in self.events:
            ids.update(event.entry_ids)
        return ids

    def get_entries_by_type(self, event_type: str) -> list[ProvenanceEvent]:
        """Get all events of a specific type.

        Args:
            event_type: The type of event to filter by

        Returns:
            List of matching events
        """
        return [e for e in self.events if e.event_type == event_type]

    def to_dict(self) -> dict[str, Any]:
        """Convert the trail to a dictionary.

        Returns:
            Dictionary representation of the trail
        """
        return {
            "trail_id": self.trail_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "event_count": len(self.events),
            "events": [
                {
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "entry_ids": e.entry_ids,
                    "metadata": e.metadata,
                }
                for e in self.events
            ],
            "final_context": self.final_context,
            "metadata": self.metadata,
        }


class ProvenanceTracker:
    """Tracks provenance for memory operations.

    This class provides methods to record and query provenance information,
    allowing you to understand what context influenced decisions.
    """

    def __init__(self) -> None:
        """Initialize the provenance tracker."""
        self._trails: dict[str, ProvenanceTrail] = {}
        self._current_trail_id: str | None = None
        self._entry_to_trails: dict[str, set[str]] = {}

    def start_trail(
        self, trail_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> str:
        """Start a new provenance trail.

        Args:
            trail_id: Optional trail ID (auto-generated if not provided)
            metadata: Optional metadata for the trail

        Returns:
            The trail ID
        """
        import time
        import uuid

        if trail_id is None:
            trail_id = str(uuid.uuid4())

        trail = ProvenanceTrail(
            trail_id=trail_id,
            start_time=time.time(),
            metadata=metadata or {},
        )

        self._trails[trail_id] = trail
        self._current_trail_id = trail_id

        logger.debug(f"Started provenance trail: {trail_id}")
        return trail_id

    def end_trail(
        self, trail_id: str | None = None, final_context: dict[str, Any] | None = None
    ) -> ProvenanceTrail | None:
        """End a provenance trail.

        Args:
            trail_id: The trail ID to end (uses current if not provided)
            final_context: Optional final context state

        Returns:
            The ended trail, or None if not found
        """
        import time

        if trail_id is None:
            trail_id = self._current_trail_id

        if trail_id is None or trail_id not in self._trails:
            return None

        trail = self._trails[trail_id]
        trail.end_time = time.time()
        trail.final_context = final_context

        if self._current_trail_id == trail_id:
            self._current_trail_id = None

        logger.debug(f"Ended provenance trail: {trail_id}")
        return trail

    def record_event(
        self,
        event_type: str,
        entry_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        context_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Record a provenance event.

        Args:
            event_type: Type of event (add, delete, compact, search, etc.)
            entry_ids: IDs of entries involved in the event
            metadata: Additional event metadata
            context_snapshot: Optional context snapshot
        """
        if self._current_trail_id is None:
            return

        event = ProvenanceEvent(
            event_type=event_type,
            entry_ids=entry_ids or [],
            metadata=metadata or {},
            context_snapshot=context_snapshot,
        )

        trail = self._trails.get(self._current_trail_id)
        if trail is None:
            return

        trail.add_event(event)

        # Update entry to trails mapping
        for entry_id in event.entry_ids:
            if entry_id not in self._entry_to_trails:
                self._entry_to_trails[entry_id] = set()
            self._entry_to_trails[entry_id].add(self._current_trail_id)

    def get_trail(self, trail_id: str) -> ProvenanceTrail | None:
        """Get a provenance trail by ID.

        Args:
            trail_id: The trail ID

        Returns:
            The trail, or None if not found
        """
        return self._trails.get(trail_id)

    def get_trails_for_entry(self, entry_id: str) -> list[ProvenanceTrail]:
        """Get all trails that involve a specific entry.

        Args:
            entry_id: The entry ID

        Returns:
            List of trails involving the entry
        """
        trail_ids = self._entry_to_trails.get(entry_id, set())
        return [self._trails[tid] for tid in trail_ids if tid in self._trails]

    def list_trails(self, limit: int | None = None) -> list[ProvenanceTrail]:
        """List all trails.

        Args:
            limit: Maximum number of trails to return

        Returns:
            List of trails
        """
        trails = list(self._trails.values())
        trails.sort(key=lambda t: t.start_time, reverse=True)
        if limit:
            trails = trails[:limit]
        return trails

    def clear(self) -> None:
        """Clear all trails."""
        self._trails.clear()
        self._entry_to_trails.clear()
        self._current_trail_id = None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Similarity score from -1 to 1 (1 is identical)
    """
    import math

    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}")

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)
