"""Contexo - LLM Context Management Library.

This library provides a two-tier memory hierarchy for managing LLM context:
- Short-term Memory (Working Memory): First-level cache with compaction
- Long-term Memory (Persistent Storage): Second-level storage with search

Example:
    ```python
    from contexo import Contexo, minimal_config

    ctx = Contexo(config=minimal_config())
    await ctx.initialize()

    await ctx.add_message("Hello, world!", role="user")
    context = await ctx.get_context()

    await ctx.close()
    ```
"""

from __future__ import annotations

import logging
from typing import Any

from contexo.config.defaults import (
    chat_config,
    cloud_config,
    development_config,
    graphdb_config,
    local_config,
    minimal_config,
    production_config,
)
from contexo.config.settings import ContexoConfig
from contexo.core.memory import EntryType, MemoryEntry
from contexo.embeddings import create_embedding_provider
from contexo.persistent_memory.persistent_memory import PersistentMemory
from contexo.persistent_memory.provenance import ProvenanceTracker
from contexo.working_memory.working_memory import WorkingMemory
from contexo.storage import create_storage

logger = logging.getLogger(__name__)


class Contexo:
    """Main Contexo class combining working and persistent memory.

    This is the primary interface for managing LLM context with both
    short-term and long-term memory capabilities.

    Example:
        ```python
        from contexo import Contexo
        from contexo.config import local_config

        ctx = Contexo(config=local_config())
        await ctx.initialize()

        await ctx.add_message("Hello, world!", role="user")
        context = await ctx.get_context()

        await ctx.close()
        ```
    """

    def __init__(
        self,
        config: ContexoConfig | None = None,
        working_memory: WorkingMemory | None = None,
        persistent_memory: PersistentMemory | None = None,
    ) -> None:
        """Initialize the Contexo context manager.

        Args:
            config: Configuration for Contexo
            working_memory: Pre-configured working memory (for advanced use)
            persistent_memory: Pre-configured persistent memory (for advanced use)
        """
        self._config = config or minimal_config()
        self._conversation_id = self._config.conversation_id

        # Initialize storage and embedding provider
        self._storage = create_storage(
            self._config.storage.backend_type,
            db_path=self._config.storage.db_path,
            connection_string=self._config.storage.connection_string,
        )

        self._embedding_provider = create_embedding_provider(
            self._config.embeddings.provider_type,
            model_name=self._config.embeddings.model_name,
            api_key=self._config.embeddings.api_key,
            dimension=self._config.embeddings.dimension,
            device=self._config.embeddings.device,
        )

        # Initialize memory components
        self._working = working_memory or self._create_working_memory()
        self._persistent = persistent_memory or PersistentMemory(
            storage=self._storage,
            embedding_provider=self._embedding_provider,
            auto_embed=True,
        )

        # Provenance tracking
        self._provenance = ProvenanceTracker() if self._config.enable_provenance else None

        self._initialized = False

    def _create_working_memory(self) -> WorkingMemory:
        """Create a working memory instance from config.

        Returns:
            Configured WorkingMemory instance
        """
        from contexo.working_memory.strategies import create_compaction_strategy

        strategy = create_compaction_strategy(
            self._config.working_memory.strategy,
            summary_target_tokens=self._config.working_memory.summary_target_tokens,
            recency_bias=self._config.working_memory.importance_recency_bias,
        )

        return WorkingMemory(
            max_tokens=self._config.working_memory.max_tokens,
            max_entries=self._config.working_memory.max_entries,
            strategy=strategy,
            token_counter=self._config.working_memory.token_counter,
            sections=self._config.working_memory.sections,
        )

    async def initialize(self) -> None:
        """Initialize the Contexo context manager.

        This should be called before using the context manager.
        """
        if self._initialized:
            return

        await self._storage.initialize()
        await self._embedding_provider.initialize()
        await self._persistent.initialize()
        await self._working.initialize()

        self._initialized = True
        logger.info("Contexo initialized")

    async def close(self) -> None:
        """Close the Contexo context manager and release resources."""
        if not self._initialized:
            return

        await self._working.close()
        await self._persistent.close()
        await self._embedding_provider.close()
        await self._storage.close()

        self._initialized = False
        logger.info("Contexo closed")

    @property
    def is_initialized(self) -> bool:
        """Check if the context manager has been initialized."""
        return self._initialized

    @property
    def working_memory(self) -> WorkingMemory:
        """Get the working memory instance."""
        return self._working

    @property
    def persistent_memory(self) -> PersistentMemory:
        """Get the persistent memory instance."""
        return self._persistent

    @property
    def config(self) -> ContexoConfig:
        """Get the configuration."""
        return self._config

    @property
    def conversation_id(self) -> str | None:
        """Get the current conversation ID."""
        return self._conversation_id

    def set_conversation_id(self, conversation_id: str) -> None:
        """Set the conversation ID for future entries.

        Args:
            conversation_id: The conversation ID to use
        """
        self._conversation_id = conversation_id

    async def add_message(
        self,
        content: str,
        role: str = "user",
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """Add a message to memory.

        Args:
            content: The message content
            role: The role (user, assistant, system, etc.)
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)

        Returns:
            The created memory entry
        """
        entry = MemoryEntry(
            entry_type=EntryType.MESSAGE,
            content=content,
            metadata={"role": role, **(metadata or {})},
            importance_score=importance,
            conversation_id=self._conversation_id,
        )

        await self._add_to_memory(entry)
        return entry

    async def add_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """Add a tool call to memory.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)

        Returns:
            The created memory entry
        """
        import json

        content = f"Tool call: {tool_name}({json.dumps(arguments)})"
        entry = MemoryEntry(
            entry_type=EntryType.TOOL_CALL,
            content=content,
            metadata={
                "tool_name": tool_name,
                "arguments": arguments,
                **(metadata or {}),
            },
            importance_score=importance,
            conversation_id=self._conversation_id,
        )

        await self._add_to_memory(entry)
        return entry

    async def add_tool_response(
        self,
        tool_name: str,
        result: Any,
        call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """Add a tool response to memory.

        Args:
            tool_name: Name of the tool that was called
            result: Result from the tool
            call_id: ID of the corresponding tool call
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)

        Returns:
            The created memory entry
        """
        import json

        content = f"Tool response: {tool_name} -> {json.dumps(result)[:500]}"
        entry = MemoryEntry(
            entry_type=EntryType.TOOL_RESPONSE,
            content=content,
            metadata={
                "tool_name": tool_name,
                "result": result,
                **(metadata or {}),
            },
            importance_score=importance,
            parent_id=call_id,
            conversation_id=self._conversation_id,
        )

        await self._add_to_memory(entry)
        return entry

    async def _add_to_memory(self, entry: MemoryEntry) -> None:
        """Add an entry to both working and persistent memory.

        Args:
            entry: The entry to add
        """
        # Add to persistent memory (this will also generate embedding)
        await self._persistent.add(entry)

        # Add to working memory (embedding is already set)
        await self._working.add(entry)

        # Track provenance
        if self._provenance:
            self._provenance.record_event(
                event_type="add",
                entry_ids=[entry.id],
                metadata={"entry_type": entry.entry_type.value},
            )

    async def get_context(
        self,
        include_summaries: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        """Get the current context as a formatted string.

        Args:
            include_summaries: Whether to include summarized entries
            max_tokens: Optional maximum tokens to include

        Returns:
            Formatted context string
        """
        entries = await self._working.list_all()

        if not include_summaries:
            entries = [e for e in entries if e.entry_type != EntryType.SUMMARIZED]

        # Build formatted context
        lines = []
        for entry in entries:
            role = entry.metadata.get("role", "system")
            lines.append(f"{role}: {entry.content}")

        context = "\n".join(lines)

        # Truncate if needed
        if max_tokens:
            # Rough token estimate
            max_chars = max_tokens * 4
            if len(context) > max_chars:
                context = context[:max_chars] + "..."

        return context

    async def search_memory(
        self,
        query: str,
        limit: int = 10,
        semantic: bool = True,
    ) -> list[MemoryEntry]:
        """Search through persistent memory.

        Args:
            query: The search query
            limit: Maximum number of results
            semantic: Whether to use semantic search (falls back to FTS)

        Returns:
            List of matching entries
        """
        results = await self._persistent.search(
            query=query,
            limit=limit,
            conversation_id=self._conversation_id,
        )

        return [r.entry for r in results]

    async def get_message_context(
        self,
        message_id: str | None = None,
        query: str | None = None,
        include_conversation: bool = True,
        conversation_window: int = 5,
        include_tool_calls: bool = True,
        include_related: bool = True,
    ) -> dict[str, Any]:
        """Get full context for a message including related entries.

        This is useful for provenance tracking and understanding what
        context influenced a specific message output. Can find context
        by message ID or search by content.

        Args:
            message_id: Direct ID of the message to get context for
            query: Search query to find the message (if ID not known)
            include_conversation: Include surrounding conversation messages
            conversation_window: Number of messages before/after to include
            include_tool_calls: Include related tool calls and responses
            include_related: Include semantically related messages

        Returns:
            Dictionary with:
                - message: The target message entry
                - parent: Parent entry (e.g., tool call for tool response)
                - children: Child entries (e.g., tool responses for tool call)
                - conversation_context: Surrounding conversation messages
                - related: Semantically related messages
                - evidence_summary: Formatted summary of all evidence

        Raises:
            ValueError: If neither message_id nor query is provided
            KeyError: If message is not found
        """
        if not message_id and not query:
            raise ValueError("Must provide either message_id or query")

        # Find the target message
        target_entry = None

        if message_id:
            target_entry = await self._persistent.get(message_id)
            if target_entry is None:
                raise KeyError(f"Message not found: {message_id}")
        else:
            # Search for the message by query
            results = await self._persistent.search(
                query=query,
                limit=1,
                conversation_id=self._conversation_id,
            )
            if results:
                target_entry = results[0].entry
            else:
                raise KeyError(f"No message found matching query: {query}")

        # Get full context from persistent memory
        context_data = await self._persistent.get_full_context(
            target_entry.id,
            include_children=include_tool_calls,
            include_conversation=include_conversation,
            conversation_window=conversation_window,
        )

        # Add evidence summary
        context_data["evidence_summary"] = self._format_evidence_summary(context_data)

        return context_data

    def _format_evidence_summary(self, context: dict[str, Any]) -> str:
        """Format the context data as a readable evidence summary.

        Args:
            context: Context data from get_full_context

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("MESSAGE EVIDENCE SUMMARY")
        lines.append("=" * 60)

        # Target message
        if context["entry"]:
            entry = context["entry"]
            lines.append("\n[TARGET MESSAGE]")
            role = entry.metadata.get("role", entry.entry_type.value)
            lines.append(f"Role: {role}")
            lines.append(f"Content: {entry.content}")
            lines.append(f"Timestamp: {entry.timestamp}")

        # Parent (e.g., what led to this message)
        if context.get("parent"):
            lines.append("\n[PARENT ENTRY]")
            parent = context["parent"]
            lines.append(f"Type: {parent.entry_type.value}")
            lines.append(f"Content: {parent.content}")

        # Children (e.g., tool calls triggered by this message)
        if context.get("children"):
            lines.append("\n[CHILD ENTRIES]")
            for child in context["children"]:
                lines.append(f"  - {child.entry_type.value}: {child.content[:80]}...")

        # Conversation context
        if context.get("conversation_context"):
            lines.append("\n[CONVERSATION CONTEXT]")
            for entry in context["conversation_context"]:
                if entry.id == context["entry"].id:
                    lines.append(f"  >>> {entry.entry_type.value}: {entry.content[:60]}...")
                else:
                    lines.append(f"      {entry.entry_type.value}: {entry.content[:60]}...")

        # Related entries
        if context.get("related"):
            lines.append("\n[RELATED MESSAGES]")
            for entry in context["related"]:
                lines.append(f"  - {entry.entry_type.value}: {entry.content[:80]}...")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    async def promote_relevant(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.6,
    ) -> list[MemoryEntry]:
        """Promote relevant entries from persistent to working memory.

        Args:
            query: Query to find relevant entries
            limit: Maximum number of entries to promote
            min_score: Minimum relevance score

        Returns:
            List of promoted entries
        """
        results = await self._persistent.search(
            query=query,
            limit=limit,
            min_score=min_score,
            conversation_id=self._conversation_id,
        )

        promoted = []
        for result in results:
            # Check if not already in working memory
            if result.entry.id not in self._working.context_window:
                try:
                    await self._working.add(result.entry)
                    promoted.append(result.entry)
                except Exception:
                    pass  # Skip if can't fit

        return promoted

    async def clear_working_memory(self) -> None:
        """Clear all entries from working memory."""
        await self._working.clear()

    async def clear_all(self) -> None:
        """Clear all entries from both working and persistent memory."""
        await self._working.clear()
        await self._persistent.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the context manager.

        Returns:
            Dictionary with statistics
        """
        return {
            "working_memory": self._working.get_stats(),
            "persistent_memory": self._persistent.get_stats(),
            "conversation_id": self._conversation_id,
            "initialized": self._initialized,
        }


__all__ = [
    "Contexo",
    "ContexoConfig",
    "MemoryEntry",
    "EntryType",
    "minimal_config",
    "local_config",
    "cloud_config",
    "development_config",
    "production_config",
    "graphdb_config",
    "chat_config",
]
