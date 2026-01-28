"""Unit tests for core components."""

import pytest

from contexo.core.context import ContextWindow
from contexo.core.exceptions import (
    CompactionError,
    ConfigurationError,
    ContexoError,
    EmbeddingError,
    SearchError,
    StorageError,
    TokenLimitError,
    ValidationError,
)
from contexo.core.memory import EntryType, MemoryEntry


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_create_entry(self):
        """Test creating a basic memory entry."""
        entry = MemoryEntry(
            id="test-id",
            entry_type=EntryType.MESSAGE,
            content="Hello, world!",
        )

        assert entry.id == "test-id"
        assert entry.entry_type == EntryType.MESSAGE
        assert entry.content == "Hello, world!"
        assert entry.token_count == 0
        assert entry.importance_score == 0.5

    def test_validation_importance_score(self):
        """Test that invalid importance scores are rejected."""
        with pytest.raises(ValidationError):
            MemoryEntry(importance_score=1.5)

        with pytest.raises(ValidationError):
            MemoryEntry(importance_score=-0.1)

    def test_validation_token_count(self):
        """Test that negative token counts are rejected."""
        with pytest.raises(ValidationError):
            MemoryEntry(token_count=-1)

    def test_validation_empty_embedding(self):
        """Test that empty embeddings are rejected."""
        with pytest.raises(ValidationError):
            MemoryEntry(embedding=[])

    def test_with_embedding(self):
        """Test creating a new entry with an embedding."""
        entry = MemoryEntry(content="test")
        embedding = [0.1, 0.2, 0.3]

        entry_with_emb = entry.with_embedding(embedding)

        assert entry_with_emb.embedding == embedding
        assert entry_with_emb.content == "test"
        # Original should be unchanged (frozen)
        assert entry.embedding is None

    def test_with_importance(self):
        """Test creating a new entry with a different importance score."""
        entry = MemoryEntry(content="test", importance_score=0.5)

        entry_new = entry.with_importance(0.9)

        assert entry_new.importance_score == 0.9
        assert entry.importance_score == 0.5  # Original unchanged


class TestContextWindow:
    """Tests for ContextWindow class."""

    def test_create_window(self):
        """Test creating a context window."""
        window = ContextWindow(max_tokens=1000, max_entries=10)

        assert window.max_tokens == 1000
        assert window.max_entries == 10
        assert window.total_tokens == 0
        assert window.total_entries == 0

    def test_validation(self):
        """Test that invalid configurations are rejected."""
        with pytest.raises(ValueError):
            ContextWindow(max_tokens=0)

        with pytest.raises(ValueError):
            ContextWindow(max_entries=0)

    def test_add_and_remove(self):
        """Test adding and removing entries."""
        window = ContextWindow(max_tokens=1000)
        entry = MemoryEntry(content="Hello", token_count=50)

        window.add(entry)
        assert window.total_tokens == 50
        assert window.total_entries == 1

        removed = window.remove(entry.id)
        assert removed is not None
        assert removed.id == entry.id
        assert window.total_tokens == 0
        assert window.total_entries == 0

    def test_can_fit(self):
        """Test checking if an entry can fit."""
        window = ContextWindow(max_tokens=100)
        entry = MemoryEntry(content="test", token_count=50)

        assert window.can_fit(entry)
        window.add(entry)
        assert not window.can_fit(entry)

    def test_utilization(self):
        """Test utilization calculation."""
        window = ContextWindow(max_tokens=100)
        entry = MemoryEntry(content="test", token_count=50)

        assert window.utilization == 0.0
        window.add(entry)
        assert window.utilization == 0.5

    def test_token_limit_error(self):
        """Test that exceeding limits raises an error."""
        window = ContextWindow(max_tokens=100)
        entry = MemoryEntry(content="large", token_count=150)

        with pytest.raises(TokenLimitError):
            window.add(entry)

    def test_remove_oldest(self):
        """Test removing oldest entries."""
        window = ContextWindow(max_tokens=1000)
        entries = [
            MemoryEntry(content=f"entry-{i}", token_count=10)
            for i in range(5)
        ]

        for entry in entries:
            window.add(entry)

        removed = window.remove_oldest(2)
        assert len(removed) == 2
        assert removed[0].content == "entry-0"
        assert removed[1].content == "entry-1"
        assert window.total_entries == 3


class TestExceptions:
    """Tests for custom exceptions."""

    def test_exception_hierarchy(self):
        """Test that all exceptions inherit from ContexoError."""
        exceptions = [
            ConfigurationError,
            StorageError,
            EmbeddingError,
            TokenLimitError,
            CompactionError,
            SearchError,
            ValidationError,
        ]

        for exc in exceptions:
            assert issubclass(exc, ContexoError)

    def test_raise_and_catch(self):
        """Test raising and catching custom exceptions."""
        with pytest.raises(ContexoError):
            raise ContexoError("Test error")

        with pytest.raises(StorageError):
            raise StorageError("Storage failed")
