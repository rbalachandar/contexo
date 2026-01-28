"""Pytest configuration and fixtures for Contexo tests."""

import pytest

from contexo.config.defaults import minimal_config
from contexo.core.memory import EntryType, MemoryEntry
from contexo.working_memory.strategies import PassthroughStrategy
from contexo.working_memory.working_memory import WorkingMemory


@pytest.fixture
def sample_entries():
    """Create sample memory entries for testing."""
    return [
        MemoryEntry(
            id="entry-1",
            entry_type=EntryType.MESSAGE,
            content="Hello, how are you?",
            metadata={"role": "user"},
            importance_score=0.8,
        ),
        MemoryEntry(
            id="entry-2",
            entry_type=EntryType.MESSAGE,
            content="I'm doing well, thanks for asking!",
            metadata={"role": "assistant"},
            importance_score=0.7,
        ),
        MemoryEntry(
            id="entry-3",
            entry_type=EntryType.TOOL_CALL,
            content="Tool call: weather(location=NYC)",
            metadata={"tool_name": "weather", "arguments": {"location": "NYC"}},
            importance_score=0.5,
        ),
    ]


@pytest.fixture
async def working_memory():
    """Create a working memory instance for testing."""
    memory = WorkingMemory(
        max_tokens=1000,
        max_entries=10,
        strategy=PassthroughStrategy(),
    )
    await memory.initialize()
    yield memory
    await memory.close()


@pytest.fixture
async def contexo():
    """Create a Contexo instance for testing."""
    config = minimal_config()
    ctx = type("MockContexo", (), {})()
    ctx.config = config

    # Create minimal components
    from contexo.storage import InMemoryStorage
    from contexo.embeddings import MockEmbeddings

    storage = InMemoryStorage()
    await storage.initialize()

    embeddings = MockEmbeddings()
    await embeddings.initialize()

    from contexo.persistent_memory.persistent_memory import PersistentMemory

    working = WorkingMemory(max_tokens=1000)
    persistent = PersistentMemory(
        storage=storage,
        embedding_provider=embeddings,
        auto_embed=False,
    )

    await working.initialize()
    await persistent.initialize()

    ctx.working_memory = working
    ctx.persistent_memory = persistent

    yield ctx

    await working.close()
    await persistent.close()
    await embeddings.close()
    await storage.close()


@pytest.fixture
def event_loop_policy():
    """Use the default event loop policy."""
    import asyncio

    # For Python 3.10+, the default policy is fine
    return asyncio.DefaultEventLoopPolicy()
