"""Context window management for token and entry limits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from contexo.core.exceptions import TokenLimitError
from contexo.core.memory import EntryType, MemoryEntry


@dataclass
class ContextWindow:
    """Manages a sliding context window with token and entry limits.

    The context window maintains a list of entries and ensures that:
    - Total tokens do not exceed max_tokens
    - Total entries do not exceed max_entries (if set)

    When limits are exceeded, entries can be evicted using a selector function.
    """

    entries: list[MemoryEntry] = field(default_factory=list)
    max_tokens: int = 4096
    max_entries: int | None = None
    token_counter: Callable[[str], int] = lambda s: len(s.split())

    def __post_init__(self) -> None:
        """Validate the context window configuration."""
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.max_entries is not None and self.max_entries <= 0:
            raise ValueError(f"max_entries must be positive or None, got {self.max_entries}")

    @property
    def total_tokens(self) -> int:
        """Get the total number of tokens in the window."""
        return sum(entry.token_count for entry in self.entries)

    @property
    def total_entries(self) -> int:
        """Get the total number of entries in the window."""
        return len(self.entries)

    @property
    def is_full(self) -> bool:
        """Check if the context window is at capacity."""
        token_full = self.total_tokens >= self.max_tokens
        entry_full = self.max_entries is not None and self.total_entries >= self.max_entries
        return token_full or entry_full

    @property
    def remaining_tokens(self) -> int:
        """Get the remaining token capacity."""
        return max(0, self.max_tokens - self.total_tokens)

    @property
    def remaining_entries(self) -> int:
        """Get the remaining entry capacity."""
        if self.max_entries is None:
            return float("inf")
        return max(0, self.max_entries - self.total_entries)

    @property
    def utilization(self) -> float:
        """Get the window utilization as a fraction (0.0 to 1.0)."""
        return min(1.0, self.total_tokens / self.max_tokens)

    def can_fit(self, entry: MemoryEntry) -> bool:
        """Check if an entry can fit in the window.

        Args:
            entry: The entry to check

        Returns:
            True if the entry can fit without exceeding limits
        """
        token_ok = self.total_tokens + entry.token_count < self.max_tokens
        entry_ok = self.max_entries is None or self.total_entries + 1 <= self.max_entries
        return token_ok and entry_ok

    def add(self, entry: MemoryEntry) -> None:
        """Add an entry to the window.

        Args:
            entry: The entry to add

        Raises:
            TokenLimitError: If the entry would exceed limits
        """
        if not self.can_fit(entry):
            raise TokenLimitError(
                f"Entry would exceed limits: {entry.token_count} tokens, "
                f"{self.remaining_tokens} remaining"
            )
        self.entries.append(entry)

    def add_if_fits(self, entry: MemoryEntry) -> bool:
        """Add an entry only if it fits.

        Args:
            entry: The entry to add

        Returns:
            True if the entry was added, False otherwise
        """
        if self.can_fit(entry):
            self.entries.append(entry)
            return True
        return False

    def remove(self, entry_id: str) -> MemoryEntry | None:
        """Remove an entry by ID.

        Args:
            entry_id: The ID of the entry to remove

        Returns:
            The removed entry, or None if not found
        """
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                return self.entries.pop(i)
        return None

    def remove_oldest(self, count: int = 1) -> list[MemoryEntry]:
        """Remove the oldest entries.

        Args:
            count: Number of entries to remove

        Returns:
            List of removed entries
        """
        count = min(count, len(self.entries))
        removed = self.entries[:count]
        self.entries = self.entries[count:]
        return removed

    def remove_by_selector(
        self, selector: Callable[[list[MemoryEntry]], list[MemoryEntry]]
    ) -> list[MemoryEntry]:
        """Remove entries selected by a selector function.

        The selector receives the current entries and should return
        the entries to remove.

        Args:
            selector: Function that selects entries to remove

        Returns:
            List of removed entries
        """
        to_remove = selector(self.entries)
        to_remove_ids = {e.id for e in to_remove}
        self.entries = [e for e in self.entries if e.id not in to_remove_ids]
        return to_remove

    def get_entries(
        self, entry_type: EntryType | None = None, limit: int | None = None
    ) -> list[MemoryEntry]:
        """Get entries from the window, optionally filtered.

        Args:
            entry_type: Filter by entry type
            limit: Maximum number of entries to return

        Returns:
            List of entries
        """
        result = self.entries
        if entry_type is not None:
            result = [e for e in result if e.entry_type == entry_type]
        if limit is not None:
            result = result[:limit]
        return result

    def get_entry_text(self, separator: str = "\n") -> str:
        """Get all entry content joined by a separator.

        Args:
            separator: String to join entries with

        Returns:
            Joined content string
        """
        return separator.join(entry.content for entry in self.entries)

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        return self.token_counter(text)

    def clear(self) -> None:
        """Clear all entries from the window."""
        self.entries.clear()

    def clone(self) -> ContextWindow:
        """Create a copy of this context window.

        Returns:
            A new ContextWindow with the same entries and configuration
        """
        return ContextWindow(
            entries=self.entries.copy(),
            max_tokens=self.max_tokens,
            max_entries=self.max_entries,
            token_counter=self.token_counter,
        )

    def __len__(self) -> int:
        """Get the number of entries in the window."""
        return len(self.entries)

    def __contains__(self, entry_id: str) -> bool:
        """Check if an entry ID is in the window.

        Args:
            entry_id: The entry ID to check

        Returns:
            True if the entry is in the window
        """
        return any(e.id == entry_id for e in self.entries)
