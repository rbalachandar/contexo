"""Working memory implementation with auto-compaction and sections."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from contexo.core.context import ContextWindow
from contexo.core.exceptions import CompactionError, TokenLimitError
from contexo.core.memory import EntryType, MemoryEntry, MemoryManager
from contexo.working_memory.strategies.base import CompactionStrategy, PassthroughStrategy

logger = logging.getLogger(__name__)


@dataclass
class SectionConfig:
    """Configuration for a memory section.

    Attributes:
        max_tokens: Maximum tokens for this section
        priority: Priority level (high=1.0, medium=0.5, low=0.2)
        pinned: If True, entries in this section are never evicted
        name: Section name
    """

    max_tokens: int
    priority: float = 0.5
    pinned: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"section_{id(self)}"


@dataclass
class SectionStats:
    """Statistics for a memory section.

    Attributes:
        name: Section name
        entry_count: Number of entries in the section
        token_count: Total tokens used by the section
        max_tokens: Maximum tokens allocated to the section
        utilization: Utilization as a fraction (0.0 to 1.0)
        priority: Priority level
        pinned: Whether the section is pinned
    """

    name: str
    entry_count: int
    token_count: int
    max_tokens: int
    utilization: float
    priority: float
    pinned: bool


class WorkingMemory(MemoryManager):
    """Short-term working memory with auto-compaction and optional sections.

    Working memory maintains a sliding context window that automatically
    compacts when full using a configured strategy.

    With sections enabled, different types of entries can be grouped
    and managed with separate token limits and eviction priorities.

    Example without sections (flat mode):
        ```python
        memory = WorkingMemory(max_tokens=4096)
        ```

    Example with sections:
        ```python
        memory = WorkingMemory(
            max_tokens=8192,
            sections={
                "system": {"max_tokens": 500, "pinned": True},
                "conversation": {"max_tokens": 5000},
                "tools": {"max_tokens": 1500, "priority": "low"},
            }
        )
        await memory.add(entry, section="conversation")
        ```
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        max_entries: int | None = None,
        strategy: CompactionStrategy | None = None,
        token_counter: Any = None,
        sections: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the working memory.

        Args:
            max_tokens: Maximum number of tokens to store
            max_entries: Maximum number of entries to store (optional)
            strategy: Compaction strategy to use when full
            token_counter: Function to count tokens in text
            sections: Optional section configuration dict
                Format: {"section_name": {"max_tokens": int, "priority": float, "pinned": bool}}
        """
        self._context_window = ContextWindow(
            max_tokens=max_tokens,
            max_entries=max_entries,
            token_counter=token_counter or self._default_token_counter,
        )
        self._strategy = strategy or PassthroughStrategy()
        self._initialized = False

        # Initialize sections
        self._sections: dict[str, SectionConfig] = {}
        self._entry_sections: dict[str, str] = {}  # entry_id -> section_name
        self._sectioned_mode = sections is not None

        if sections:
            for name, config in sections.items():
                self._sections[name] = SectionConfig(
                    name=name,
                    max_tokens=config.get("max_tokens", max_tokens),
                    priority=config.get("priority", 0.5),
                    pinned=config.get("pinned", False),
                )
            logger.info(f"Working memory initialized with {len(sections)} sections")
        else:
            # Create a default section for backward compatibility
            self._sections["default"] = SectionConfig(
                name="default",
                max_tokens=max_tokens,
                priority=0.5,
                pinned=False,
            )

    def _default_token_counter(self, text: str) -> int:
        """Default token counter using word count approximation.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        return max(1, len(text) // 4)

    async def initialize(self) -> None:
        """Initialize the working memory."""
        self._initialized = True
        logger.debug("Working memory initialized")

    async def close(self) -> None:
        """Close the working memory and clear entries."""
        self._context_window.clear()
        self._entry_sections.clear()
        self._initialized = False
        logger.debug("Working memory closed")

    @property
    def is_initialized(self) -> bool:
        """Check if the working memory has been initialized."""
        return self._initialized

    @property
    def context_window(self) -> ContextWindow:
        """Get the context window."""
        return self._context_window

    @property
    def strategy(self) -> CompactionStrategy:
        """Get the compaction strategy."""
        return self._strategy

    def set_strategy(self, strategy: CompactionStrategy) -> None:
        """Set a new compaction strategy.

        Args:
            strategy: The new strategy to use
        """
        self._strategy = strategy

    @property
    def total_tokens(self) -> int:
        """Get the total number of tokens in working memory."""
        return self._context_window.total_tokens

    @property
    def utilization(self) -> float:
        """Get the current utilization (0.0 to 1.0)."""
        return self._context_window.utilization

    @property
    def sectioned_mode(self) -> bool:
        """Check if sections are enabled."""
        return self._sectioned_mode

    async def add(
        self,
        entry: MemoryEntry,
        section: str | None = None,
    ) -> MemoryEntry:
        """Add an entry to working memory.

        If sections are enabled, the entry is added to the specified section.
        If the entry doesn't fit, compaction will be triggered.

        Args:
            entry: The entry to add
            section: Section name (ignored if sections not enabled)

        Returns:
            The added entry (with estimated token count if not set)
        """
        if not self._initialized:
            raise RuntimeError("Working memory not initialized")

        # Determine which section to use
        target_section = self._get_target_section(section, entry)

        # Estimate token count if not provided
        if entry.token_count == 0:
            # Prepare metadata with section information
            metadata = dict(entry.metadata) if entry.metadata else {}
            if target_section != "default":
                metadata["section"] = target_section

            entry = MemoryEntry(
                id=entry.id,
                entry_type=entry.entry_type,
                content=entry.content,
                metadata=metadata,
                timestamp=entry.timestamp,
                token_count=self._context_window.estimate_tokens(entry.content),
                importance_score=entry.importance_score,
                embedding=entry.embedding,
                parent_id=entry.parent_id,
                conversation_id=entry.conversation_id,
            )

        # Check section limit (if in sectioned mode)
        if self._sectioned_mode:
            section_usage = self._get_section_usage(target_section)
            section_config = self._sections[target_section]

            if section_usage >= section_config.max_tokens:
                # Section is full - need to compact this section
                await self._compact_section(target_section, entry.token_count)

        # If entry fits in total window, add it directly
        if self._context_window.can_fit(entry):
            self._context_window.add(entry)
            self._entry_sections[entry.id] = target_section
            logger.debug(
                f"Added entry {entry.id} to section {target_section} ({entry.token_count} tokens)"
            )
            return entry

        # Entry doesn't fit in total window - trigger compaction
        await self._compact_and_add(entry, target_section)
        return entry

    def _get_target_section(self, section: str | None, entry: MemoryEntry) -> str:
        """Determine the target section for an entry.

        Args:
            section: Explicitly specified section
            entry: The entry to categorize

        Returns:
            Section name to use
        """
        # If section is explicitly provided, use it
        if section:
            if section not in self._sections:
                # Create section on the fly
                self._sections[section] = SectionConfig(
                    name=section,
                    max_tokens=self._context_window.max_tokens,
                    priority=0.5,
                    pinned=False,
                )
            return section

        # Auto-determine section based on entry type
        if self._sectioned_mode:
            entry_type = entry.entry_type.value

            # Map entry types to sections
            section_mapping = {
                "system": "system",
                "tool_call": "tools",
                "tool_response": "tools",
                "summarized": "conversation",
            }

            # Check if a section exists for this entry type
            for type_name, section_name in section_mapping.items():
                if entry_type == type_name and section_name in self._sections:
                    return section_name

            # Default to conversation section if it exists
            if "conversation" in self._sections:
                return "conversation"

        return "default"

    def _get_section_usage(self, section_name: str) -> int:
        """Get the current token usage for a section.

        Args:
            section_name: The section to check

        Returns:
            Current token count for the section
        """
        usage = 0
        for entry in self._context_window.entries:
            entry_section = self._entry_sections.get(entry.id, "default")
            if entry_section == section_name:
                usage += entry.token_count
        return usage

    async def _compact_section(
        self,
        section_name: str,
        required_tokens: int,
    ) -> None:
        """Compact a specific section to free up space.

        Args:
            section_name: Section to compact
            required_tokens: Tokens to free up
        """
        section_config = self._sections[section_name]

        # Can't evict from pinned sections
        if section_config.pinned:
            raise TokenLimitError(
                f"Cannot evict from pinned section '{section_name}'. "
                f"Need {required_tokens} tokens but section is full."
            )

        # Get entries in this section
        section_entries = [
            e
            for e in self._context_window.entries
            if self._entry_sections.get(e.id, "default") == section_name
        ]

        # Evict oldest entries from this section
        to_evict = section_entries[: required_tokens // 50 + 1]  # Rough estimate

        for entry in to_evict:
            self._context_window.remove(entry.id)
            self._entry_sections.pop(entry.id, None)
            logger.debug(f"Evicted entry {entry.id} from section {section_name}")

    async def _compact_and_add(
        self,
        new_entry: MemoryEntry,
        target_section: str = "default",
    ) -> None:
        """Compact the context window and add the new entry.

        Args:
            new_entry: The entry to add after compaction
            target_section: Section where the entry should go

        Raises:
            TokenLimitError: If compaction cannot free enough space
        """
        logger.debug(
            f"Compaction triggered: need {new_entry.token_count} tokens, "
            f"{self._context_window.remaining_tokens} remaining, "
            f"target section: {target_section}"
        )

        max_attempts = 10
        attempt = 0

        while not self._context_window.can_fit(new_entry) and attempt < max_attempts:
            # Get entries that can be evicted (not pinned)
            evictable_entries = [
                e
                for e in self._context_window.entries
                if not self._sections.get(
                    self._entry_sections.get(e.id, "default"),
                    self._sections["default"],
                ).pinned
            ]

            if not evictable_entries:
                break

            # Select entries to evict
            entries_to_evict_count = max(1, len(evictable_entries) // 4)
            to_evict = await self._strategy.select_for_eviction(
                evictable_entries, entries_to_evict_count
            )

            if not to_evict:
                break

            # Evict the entries
            for entry in to_evict:
                self._context_window.remove(entry.id)
                entry_section = self._entry_sections.pop(entry.id, None)
                logger.debug(f"Evicted entry {entry.id} from section {entry_section}")

            # If strategy supports compaction, create a summary
            if hasattr(self._strategy, "compact"):
                try:
                    compacted = await self._strategy.compact(to_evict)
                    for summary_entry in compacted:
                        if self._context_window.can_fit(summary_entry):
                            self._context_window.add(summary_entry)
                            self._entry_sections[summary_entry.id] = target_section
                            logger.debug(f"Added summary entry {summary_entry.id}")
                except CompactionError as e:
                    logger.warning(f"Compaction failed: {e}")

            attempt += 1

        # Try to add the entry again
        if not self._context_window.can_fit(new_entry):
            raise TokenLimitError(
                f"Cannot fit entry ({new_entry.token_count} tokens) even after compaction. "
                f"Current: {self._context_window.total_tokens}/{self._context_window.max_tokens} tokens"
            )

        self._context_window.add(new_entry)
        self._entry_sections[new_entry.id] = target_section
        logger.debug(f"Added entry {new_entry.id} to section {target_section} after compaction")

    # ==================== Section Management ====================

    def add_section(
        self,
        name: str,
        max_tokens: int,
        priority: float = 0.5,
        pinned: bool = False,
    ) -> None:
        """Add a new section to working memory.

        Args:
            name: Section name
            max_tokens: Maximum tokens for this section
            priority: Priority level (1.0=high, 0.5=medium, 0.2=low)
            pinned: If True, entries are never evicted from this section
        """
        self._sections[name] = SectionConfig(
            name=name,
            max_tokens=max_tokens,
            priority=priority,
            pinned=pinned,
        )
        self._sectioned_mode = True
        logger.info(
            f"Added section '{name}' (max_tokens={max_tokens}, priority={priority}, pinned={pinned})"
        )

    def update_section(
        self,
        name: str,
        max_tokens: int | None = None,
        priority: float | None = None,
        pinned: bool | None = None,
    ) -> None:
        """Update an existing section's configuration.

        Args:
            name: Section name to update
            max_tokens: New max tokens (optional)
            priority: New priority (optional)
            pinned: New pinned state (optional)
        """
        if name not in self._sections:
            raise KeyError(f"Section not found: {name}")

        section = self._sections[name]

        if max_tokens is not None:
            section.max_tokens = max_tokens
        if priority is not None:
            section.priority = priority
        if pinned is not None:
            section.pinned = pinned

        logger.info(f"Updated section '{name}'")

    def remove_section(self, name: str) -> None:
        """Remove a section and move its entries to the default section.

        Args:
            name: Section name to remove
        """
        if name not in self._sections:
            raise KeyError(f"Section not found: {name}")

        if name == "default":
            raise ValueError("Cannot remove default section")

        # Move entries to default section
        for entry_id, section in list(self._entry_sections.items()):
            if section == name:
                self._entry_sections[entry_id] = "default"

        del self._sections[name]
        logger.info(f"Removed section '{name}'")

    def get_section_names(self) -> list[str]:
        """Get list of all section names.

        Returns:
            List of section names
        """
        return list(self._sections.keys())

    def get_section_stats(self, name: str | None = None) -> SectionStats | list[SectionStats]:
        """Get statistics for section(s).

        Args:
            name: Section name (None = all sections)

        Returns:
            SectionStats or list of SectionStats
        """
        if name:
            if name not in self._sections:
                raise KeyError(f"Section not found: {name}")

            section = self._sections[name]
            entry_count = sum(
                1
                for e in self._context_window.entries
                if self._entry_sections.get(e.id, "default") == name
            )
            token_count = sum(
                e.token_count
                for e in self._context_window.entries
                if self._entry_sections.get(e.id, "default") == name
            )

            return SectionStats(
                name=section.name,
                entry_count=entry_count,
                token_count=token_count,
                max_tokens=section.max_tokens,
                utilization=token_count / section.max_tokens if section.max_tokens > 0 else 0,
                priority=section.priority,
                pinned=section.pinned,
            )
        else:
            return [self.get_section_stats(n) for n in self._sections.keys()]

    # ==================== Original Methods ====================

    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Get an entry by ID.

        Args:
            entry_id: The ID of the entry to get

        Returns:
            The entry if found, None otherwise
        """
        for entry in self._context_window.entries:
            if entry.id == entry_id:
                return entry
        return None

    async def update(self, entry: MemoryEntry) -> MemoryEntry:
        """Update an existing entry.

        Args:
            entry: The entry to update

        Returns:
            The updated entry
        """
        if not self._initialized:
            raise RuntimeError("Working memory not initialized")

        existing = await self.get(entry.id)
        if existing is None:
            raise KeyError(f"Entry not found: {entry.id}")

        # Preserve section assignment
        old_section = self._entry_sections.get(entry.id, "default")
        self._context_window.remove(entry.id)
        self._entry_sections.pop(entry.id, None)
        await self.add(entry, section=old_section)

        return entry

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID.

        Args:
            entry_id: The ID of the entry to delete

        Returns:
            True if deleted, False if not found
        """
        removed = self._context_window.remove(entry_id)
        if removed:
            self._entry_sections.pop(entry_id, None)
        return removed is not None

    async def list_all(
        self,
        limit: int | None = None,
        entry_type: EntryType | None = None,
        section: str | None = None,
        agent_id: str | None = None,
        scope: str | None = None,
    ) -> list[MemoryEntry]:
        """List entries in working memory.

        Args:
            limit: Maximum number of entries to return
            entry_type: Filter by entry type
            section: Filter by section
            agent_id: Filter by agent_id metadata (multi-agent mode)
            scope: Filter by scope metadata (e.g., "private", "shared")

        Returns:
            List of entries
        """
        entries = self._context_window.entries

        if entry_type is not None:
            entries = [e for e in entries if e.entry_type == entry_type]

        if section is not None:
            entries = [e for e in entries if self._entry_sections.get(e.id, "default") == section]

        # Multi-agent filtering
        if agent_id is not None:
            entries = [e for e in entries if e.metadata and e.metadata.get("agent_id") == agent_id]

        if scope is not None:
            entries = [e for e in entries if e.metadata and e.metadata.get("scope") == scope]

        if limit:
            entries = entries[:limit]

        return entries

    async def count(self, entry_type: EntryType | None = None, section: str | None = None) -> int:
        """Count entries in working memory.

        Args:
            entry_type: Filter by entry type
            section: Filter by section

        Returns:
            The count of entries
        """
        entries = await self.list_all(entry_type=entry_type, section=section)
        return len(entries)

    async def clear(self, section: str | None = None) -> None:
        """Clear entries from working memory.

        Args:
            section: If specified, only clear this section
        """
        if section is None:
            self._context_window.clear()
            self._entry_sections.clear()
        else:
            # Remove entries from specific section
            to_remove = [
                e.id
                for e in self._context_window.entries
                if self._entry_sections.get(e.id, "default") == section
            ]
            for entry_id in to_remove:
                self._context_window.remove(entry_id)
                self._entry_sections.pop(entry_id, None)

        logger.debug(f"Working memory cleared (section: {section or 'all'})")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the working memory.

        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "total_entries": len(self._context_window.entries),
            "total_tokens": self._context_window.total_tokens,
            "max_tokens": self._context_window.max_tokens,
            "max_entries": self._context_window.max_entries,
            "utilization": self._context_window.utilization,
            "remaining_tokens": self._context_window.remaining_tokens,
            "strategy": self._strategy.name,
            "sectioned_mode": self._sectioned_mode,
        }

        if self._sectioned_mode:
            stats["sections"] = self.get_section_stats()

        return stats

    def format_context(
        self,
        separator: str = "\n",
        by_section: bool = False,
        section_separator: str = "\n---\n",
    ) -> str:
        """Format entries as a string.

        Args:
            separator: String to join entries with
            by_section: If True, format sections separately
            section_separator: Separator between sections (if by_section=True)

        Returns:
            Formatted context string
        """
        if not by_section or not self._sectioned_mode:
            return self._context_window.get_entry_text(separator=separator)

        # Format by section
        parts = []
        for section_name in self._sections:
            entries = [
                e
                for e in self._context_window.entries
                if self._entry_sections.get(e.id, "default") == section_name
            ]
            if entries:
                parts.append(f"=== {section_name} ===")
                parts.append(
                    separator.join(
                        f"{e.metadata.get('role', e.entry_type.value)}: {e.content}"
                        for e in entries
                    )
                )

        return section_separator.join(parts)

    async def promote_from(
        self, entries: list[MemoryEntry], section: str | None = None
    ) -> list[MemoryEntry]:
        """Promote entries from long-term memory to working memory.

        Args:
            entries: Entries to promote
            section: Section to add entries to

        Returns:
            List of entries that were successfully promoted
        """
        promoted = []

        for entry in entries:
            try:
                await self.add(entry, section=section)
                promoted.append(entry)
            except TokenLimitError:
                break

        return promoted


__all__ = ["WorkingMemory", "SectionConfig", "SectionStats"]
