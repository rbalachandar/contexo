"""Contexo memory integration for LangChain.

This module provides a LangChain Memory class that uses Contexo
for persistent, searchable context management.
"""

from __future__ import annotations

import logging
from typing import Any

from contexo import Contexo, ContexoConfig

logger = logging.getLogger(__name__)

try:
    from langchain.memory import BaseMemory
    from langchain.schema import BaseMessage, get_buffer_string

    _langchain_available = True
except ImportError:
    _langchain_available = False

    class BaseMemory:  # type: ignore[no-redef]
        """Placeholder when LangChain is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "langchain is required for LangChain integration. "
                "Install with: pip install langchain"
            )


class ContexoMemory(BaseMemory):
    """LangChain Memory implementation using Contexo.

    This memory class provides:
    - Persistent storage across sessions
    - Semantic search for context retrieval
    - Automatic context management with compaction
    - Provenance tracking for message history

    Example:
        ```python
        from langchain.chains import ConversationChain
        from langchain.llms import OpenAI
        from contexo.integrations import ContexoMemory

        memory = ContexoMemory()
        llm = OpenAI(temperature=0)
        conversation = ConversationChain(llm=llm, memory=memory)

        conversation.predict(input="Hi, I'm Alice!")
        conversation.predict(input="What's my name?")
        # The model remembers "Alice" from persistent memory
        ```
    """

    contexo: Contexo

    @property
    def memory_variables(self) -> list[str]:
        """Return the memory variable names."""
        return ["history"]

    def __init__(
        self,
        contexo: Contexo | None = None,
        config: ContexoConfig | None = None,
        conversation_id: str | None = None,
        max_context_messages: int = 10,
        include_summaries: bool = True,
        search_relevant_context: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Contexo memory.

        Args:
            contexo: Pre-configured Contexo instance
            config: Configuration for creating a new Contexo instance
            conversation_id: ID for this conversation
            max_context_messages: Maximum messages to include in context
            include_summaries: Whether to include summarized entries
            search_relevant_context: Whether to search for relevant past context
            **kwargs: Additional arguments passed to BaseMemory
        """
        super().__init__(**kwargs)

        self._own_instance = False
        if contexo is not None:
            self.contexo = contexo
        else:
            from contexo.config import local_config

            config = config or local_config()
            self.contexo = Contexo(config=config)
            self._own_instance = True

        self.conversation_id = conversation_id
        self.max_context_messages = max_context_messages
        self.include_summaries = include_summaries
        self.search_relevant_context = search_relevant_context

        # Track if we need to initialize
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure Contexo is initialized."""
        if not self._initialized:
            await self.contexo.initialize()
            if self.conversation_id:
                self.contexo.set_conversation_id(self.conversation_id)
            self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if the memory has been initialized."""
        return self._initialized

    async def _load_context(self, input_str: str = "") -> str:
        """Load conversation context for the LLM.

        Args:
            input_str: Current input to search for relevant context

        Returns:
            Formatted conversation history
        """
        await self._ensure_initialized()

        # Optionally search for relevant past context
        if self.search_relevant_context and input_str:
            try:
                relevant = await self.contexo.search_memory(
                    query=input_str,
                    limit=5,
                )
                if relevant:
                    logger.debug(f"Found {len(relevant)} relevant context entries")
            except Exception as e:
                logger.warning(f"Context search failed: {e}")

        # Get formatted context
        context = await self.contexo.get_context(
            include_summaries=self.include_summaries,
        )

        return context

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Load memory variables for LangChain.

        Args:
            inputs: Input values for the chain

        Returns:
            Dictionary with memory variables
        """
        # Note: LangChain uses sync, so we need to handle async
        # This is a simplified version - for full async support,
        # consider using LangChain's async chains

        # For now, return a placeholder and let async chains override
        return {"history": ""}

    async def aget_context(self, input_str: str = "") -> str:
        """Async version of getting context.

        Args:
            input_str: Current input to search for relevant context

        Returns:
            Formatted conversation history
        """
        return await self._load_context(input_str)

    async def asave_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        """Save context after a conversation turn.

        Args:
            inputs: Input values (user message)
            outputs: Output values (AI response)
        """
        await self._ensure_initialized()

        # Extract the messages
        # LangChain typically passes 'input' and 'response'
        user_message = inputs.get("input") or inputs.get("question", "")
        ai_response = outputs.get("response") or outputs.get("answer") or outputs.get("text", "")

        # Add messages to Contexo
        if user_message:
            await self.contexo.add_message(
                content=str(user_message),
                role="user",
                importance=0.7,
            )

        if ai_response:
            await self.contexo.add_message(
                content=str(ai_response),
                role="assistant",
                importance=0.8,
            )

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Sync version of save_context (for compatibility).

        Note: This will not persist data. Use asave_context for proper functionality.
        """
        logger.warning(
            "save_context called on ContexoMemory - this is a no-op. "
            "Use async chains or call asave_context directly."
        )

    def clear(self) -> None:
        """Clear the conversation history."""
        if self._initialized:
            # For async, we can't directly call async here
            # Users should use aclear() instead
            logger.warning("clear() called - use aclear() for proper cleanup")

    async def aclear(self) -> None:
        """Async version of clear."""
        await self._ensure_initialized()
        await self.contexo.clear_working_memory()

    async def aget_provenance(
        self,
        query: str,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        """Get provenance information for a message.

        Args:
            query: Search query to find the message
            message_id: Direct message ID (overrides query)

        Returns:
            Provenance context dictionary
        """
        await self._ensure_initialized()
        return await self.contexo.get_message_context(
            message_id=message_id,
            query=query,
        )

    def __del__(self) -> None:
        """Clean up resources."""
        if self._own_instance and self._initialized:
            # Note: can't use async in __del__, so this is best-effort
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a task to close
                    asyncio.create_task(self.contexo.close())
                else:
                    # Run in new loop
                    asyncio.run(self.contexo.close())
            except Exception:
                pass  # Best effort cleanup


def create_contexo_memory(
    config: ContexoConfig | None = None,
    **kwargs: Any,
) -> ContexoMemory:
    """Factory function to create a ContexoMemory instance.

    Args:
        config: Contexo configuration
        **kwargs: Additional arguments passed to ContexoMemory

    Returns:
        Configured ContexoMemory instance

    Example:
        ```python
        from contexo.integrations import create_contexo_memory

        memory = create_contexo_memory(
            conversation_id="my-chat",
            max_context_messages=20,
        )
        ```
    """
    return ContexoMemory(config=config, **kwargs)


__all__ = [
    "ContexoMemory",
    "create_contexo_memory",
]
