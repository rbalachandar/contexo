"""Summarization compaction strategy - LLM-based compression."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from contexo.core.exceptions import CompactionError
from contexo.core.memory import EntryType, MemoryEntry
from contexo.working_memory.strategies.base import CompactionStrategy


class SummarizationStrategy(CompactionStrategy):
    """Summarization strategy using an LLM to compress older entries.

    This strategy selects the oldest entries and uses an LLM to
    create a condensed summary, preserving key information while
    reducing token count.
    """

    def __init__(
        self,
        summarize_fn: Any | None = None,
        summary_target_tokens: int = 200,
    ) -> None:
        """Initialize the summarization strategy.

        Args:
            summarize_fn: Async function that takes a string and returns a summary.
                If None, a simple truncation-based summary is used.
            summary_target_tokens: Target token count for the summary
        """
        self._summarize_fn = summarize_fn
        self._summary_target_tokens = summary_target_tokens

    @property
    def name(self) -> str:
        """Get the name of this strategy."""
        return "summarization"

    async def select_for_eviction(
        self, entries: list[MemoryEntry], target_count: int
    ) -> list[MemoryEntry]:
        """Select the oldest entries for eviction.

        Args:
            entries: Current entries in the context window
            target_count: Number of entries to select

        Returns:
            List of oldest entries
        """
        return entries[:target_count]

    async def compact(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Compact entries into a summarized form.

        Args:
            entries: Entries to compact

        Returns:
            A list containing a single summary entry
        """
        if not entries:
            return []

        # Combine the content of all entries
        combined_text = self._format_entries_for_summarization(entries)

        # Generate the summary
        if self._summarize_fn is not None:
            try:
                summary_text = await self._summarize_fn(combined_text)
            except Exception as e:
                raise CompactionError(f"Failed to generate summary: {e}") from e
        else:
            # Simple truncation-based fallback
            summary_text = self._simple_summary(combined_text)

        # Create a summary entry
        summary_entry = MemoryEntry(
            entry_type=EntryType.SUMMARIZED,
            content=summary_text,
            metadata={
                "compaction_strategy": self.name,
                "original_entry_count": len(entries),
                "original_entry_ids": [e.id for e in entries],
                "original_timestamps": [e.timestamp for e in entries],
                "summary_generated_at": datetime.now().timestamp(),
            },
            importance_score=max(e.importance_score for e in entries),
            conversation_id=entries[0].conversation_id,
        )

        return [summary_entry]

    def _format_entries_for_summarization(self, entries: list[MemoryEntry]) -> str:
        """Format entries for input to the summarization function.

        Args:
            entries: Entries to format

        Returns:
            Formatted text string
        """
        lines = []
        for entry in entries:
            timestamp = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            type_label = entry.entry_type.value.upper()
            lines.append(f"[{timestamp}] ({type_label}) {entry.content}")
        return "\n".join(lines)

    def _simple_summary(self, text: str) -> str:
        """Generate a simple truncation-based summary.

        Args:
            text: Text to summarize

        Returns:
            Truncated text with indicator
        """
        # Rough approximation of tokens (4 chars per token)
        max_chars = self._summary_target_tokens * 4

        if len(text) <= max_chars:
            return text

        return (
            text[: max_chars - 30]
            + "... [truncated, "
            + str(len(text) - max_chars + 30)
            + " chars omitted]"
        )


class LLMSummarizationStrategy(SummarizationStrategy):
    """Summarization strategy with built-in LLM client support.

    This strategy includes its own LLM client for generating summaries.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: str | None = None,
        summary_target_tokens: int = 200,
        **kwargs: Any,
    ) -> None:
        """Initialize the LLM summarization strategy.

        Args:
            model: Model name to use for summarization
            api_key: API key for the LLM provider
            summary_target_tokens: Target token count for summaries
            **kwargs: Additional arguments for the LLM client
        """
        super().__init__(summary_target_tokens=summary_target_tokens)

        self._model = model
        self._api_key = api_key
        self._client_kwargs = kwargs
        self._client: Any = None

    async def initialize(self) -> None:
        """Initialize the LLM client."""
        if self._client is not None:
            return

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key, **self._client_kwargs)
        except ImportError:
            raise CompactionError("openai package is required for LLMSummarizationStrategy")

    async def close(self) -> None:
        """Close the LLM client."""
        if self._client:
            await self._client.close()
            self._client = None

    async def _summarize_with_llm(self, text: str) -> str:
        """Generate a summary using the LLM.

        Args:
            text: Text to summarize

        Returns:
            Generated summary
        """
        if self._client is None:
            await self.initialize()

        prompt = f"""Please summarize the following conversation context concisely,
preserving the most important information and key decisions.

Context to summarize:
{text}

Summary:"""

        try:
            response = await self._client.chat.completions.create(  # type: ignore
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes conversation context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._summary_target_tokens * 2,
                temperature=0.3,
            )
            return response.choices[0].message.content or "[Summary generation failed]"
        except Exception as e:
            raise CompactionError(f"LLM summarization failed: {e}") from e

    async def compact(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Compact entries using LLM summarization.

        Args:
            entries: Entries to compact

        Returns:
            A list containing a single summary entry
        """
        if not entries:
            return []

        # Set the summarize function and delegate to parent
        self._summarize_fn = self._summarize_with_llm
        return await super().compact(entries)
