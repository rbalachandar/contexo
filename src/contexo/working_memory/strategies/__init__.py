"""Compaction strategy implementations and factory."""

from __future__ import annotations

from typing import Any

from contexo.working_memory.strategies.base import CompactionStrategy, PassthroughStrategy
from contexo.working_memory.strategies.importance import ImportanceStrategy
from contexo.working_memory.strategies.sliding_window import SlidingWindowStrategy
from contexo.working_memory.strategies.summarization import (
    LLMSummarizationStrategy,
    SummarizationStrategy,
)


def create_compaction_strategy(strategy_type: str, **kwargs: Any) -> CompactionStrategy:
    """Create a compaction strategy by type name.

    Args:
        strategy_type: The type of strategy to create
            - "passthrough" or "none": PassthroughStrategy
            - "sliding_window" or "fifo": SlidingWindowStrategy
            - "summarization": SummarizationStrategy
            - "llm_summarization": LLMSummarizationStrategy
            - "importance": ImportanceStrategy
        **kwargs: Additional arguments passed to the strategy constructor

    Returns:
        A compaction strategy instance

    Raises:
        ValueError: If the strategy type is unknown
    """
    strategy_type = strategy_type.lower().replace("-", "_")

    if strategy_type in ("passthrough", "none"):
        return PassthroughStrategy()

    if strategy_type in ("sliding_window", "fifo"):
        return SlidingWindowStrategy()

    if strategy_type == "summarization":
        # SummarizationStrategy accepts summary_target_tokens
        return SummarizationStrategy(
            summary_target_tokens=kwargs.get("summary_target_tokens"),
        )

    if strategy_type in ("llm_summarization", "llm"):
        # LLMSummarizationStrategy accepts summary_target_tokens and recency_bias
        return LLMSummarizationStrategy(
            summary_target_tokens=kwargs.get("summary_target_tokens"),
            recency_bias=kwargs.get("recency_bias"),
        )

    if strategy_type == "importance":
        # ImportanceStrategy accepts recency_bias
        return ImportanceStrategy(recency_bias=kwargs.get("recency_bias"))

    raise ValueError(f"Unknown compaction strategy type: {strategy_type}")


__all__ = [
    "CompactionStrategy",
    "PassthroughStrategy",
    "SlidingWindowStrategy",
    "SummarizationStrategy",
    "LLMSummarizationStrategy",
    "ImportanceStrategy",
    "create_compaction_strategy",
]
