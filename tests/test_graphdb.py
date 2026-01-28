"""Tests for GraphDB storage backend."""

import pytest

# Only run these tests if neo4j is available
pytest.importorskip("neo4j")

from contexo.core.memory import EntryType, MemoryEntry
from contexo.storage.graphdb import GraphDBStorage


@pytest.fixture
async def graphdb_storage():
    """Create a GraphDB storage instance for testing.

    Note: This requires a running Neo4j instance.
    Skip these tests if Neo4j is not available.
    """
    try:
        storage = GraphDBStorage(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",  # Default Docker Neo4j password
            database="neo4j",
        )
        await storage.initialize()
        yield storage
        # Clean up test data
        await storage._execute_query("MATCH (n:MemoryEntry) DETACH DELETE n")
        await storage.close()
    except Exception:
        pytest.skip("Neo4j not available - skipping GraphDB tests")


class TestGraphDBStorage:
    """Tests for GraphDB storage backend."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, graphdb_storage):
        """Test saving and loading entries."""
        entry = MemoryEntry(
            id="test-1",
            entry_type=EntryType.MESSAGE,
            content="Hello, GraphDB!",
            metadata={"role": "user"},
            importance_score=0.8,
            conversation_id="test-conv",
        )

        await graphdb_storage.save(entry)
        loaded = await graphdb_storage.load("test-1")

        assert loaded is not None
        assert loaded.id == "test-1"
        assert loaded.content == "Hello, GraphDB!"
        assert loaded.importance_score == 0.8

    @pytest.mark.asyncio
    async def test_auto_relationships(self, graphdb_storage):
        """Test that automatic relationships are created."""
        # Create two entries in the same conversation
        entry1 = MemoryEntry(
            id="test-2a",
            entry_type=EntryType.MESSAGE,
            content="First message",
            conversation_id="test-conv-2",
            timestamp=1.0,
        )

        entry2 = MemoryEntry(
            id="test-2b",
            entry_type=EntryType.MESSAGE,
            content="Second message",
            conversation_id="test-conv-2",
            timestamp=2.0,
            parent_id="test-2a",
        )

        await graphdb_storage.save(entry1)
        await graphdb_storage.save(entry2)

        # Check for conversation relationship
        related = await graphdb_storage.find_related(
            "test-2a",
            relation_types=["CONVERSATION"],
        )

        assert len(related) >= 1
        assert any(e.id == "test-2b" for e, _, _ in related)

    @pytest.mark.asyncio
    async def test_create_relationship(self, graphdb_storage):
        """Test creating custom relationships."""
        entry1 = MemoryEntry(
            id="test-3a",
            entry_type=EntryType.MESSAGE,
            content="Entry A",
            conversation_id="test-conv-3",
        )

        entry2 = MemoryEntry(
            id="test-3b",
            entry_type=EntryType.MESSAGE,
            content="Entry B",
            conversation_id="test-conv-3",
        )

        await graphdb_storage.save(entry1)
        await graphdb_storage.save(entry2)

        # Create custom relationship
        await graphdb_storage.create_relationship(
            from_id="test-3a",
            to_id="test-3b",
            relation_type="CUSTOM_REL",
            properties={"strength": 0.9},
        )

        # Verify relationship exists
        related = await graphdb_storage.find_related(
            "test-3a",
            relation_types=["CUSTOM_REL"],
        )

        assert len(related) == 1
        assert related[0][0].id == "test-3b"
        assert related[0][1] == "CUSTOM_REL"

    @pytest.mark.asyncio
    async def test_traverse(self, graphdb_storage):
        """Test relationship traversal."""
        # Create a chain: A -> B -> C
        entries = []
        for i, char in enumerate(["A", "B", "C"]):
            entry = MemoryEntry(
                id=f"test-4-{char.lower()}",
                entry_type=EntryType.MESSAGE,
                content=f"Entry {char}",
                conversation_id="test-conv-4",
                timestamp=float(i),
            )
            entries.append(entry)
            await graphdb_storage.save(entry)

        # Create custom chain relationships
        await graphdb_storage.create_relationship(entries[0].id, entries[1].id, "CHAIN")
        await graphdb_storage.create_relationship(entries[1].id, entries[2].id, "CHAIN")

        # Traverse from A
        connected = await graphdb_storage.traverse(
            entries[0].id,
            relation_type="CHAIN",
            direction="outgoing",
            depth=2,
        )

        # Should find B and C
        connected_ids = {e.id for e in connected}
        assert entries[1].id in connected_ids
        assert entries[2].id in connected_ids

    @pytest.mark.asyncio
    async def test_find_entities(self, graphdb_storage):
        """Test finding entries by entity name."""
        entry = MemoryEntry(
            id="test-5",
            entry_type=EntryType.MESSAGE,
            content="Alice went to the store.",
            conversation_id="test-conv-5",
        )

        await graphdb_storage.save(entry)

        results = await graphdb_storage.find_entities("Alice")
        assert len(results) >= 1
        assert any(e.id == "test-5" for e in results)

    @pytest.mark.asyncio
    async def test_conversation_context(self, graphdb_storage):
        """Test getting conversation context."""
        # Create a temporal sequence
        for i in range(5):
            entry = MemoryEntry(
                id=f"test-6-{i}",
                entry_type=EntryType.MESSAGE,
                content=f"Message {i}",
                conversation_id="test-conv-6",
                timestamp=float(i),
            )
            await graphdb_storage.save(entry)

        # Get context around entry 2
        context = await graphdb_storage.get_conversation_context(
            "test-6-2",
            window_size=2,
        )

        # Should include entry 2 and neighbors
        context_ids = {e.id for e in context}
        assert "test-6-2" in context_ids
        assert len(context) >= 1

    @pytest.mark.asyncio
    async def test_search(self, graphdb_storage):
        """Test searching entries."""
        entry1 = MemoryEntry(
            id="test-7a",
            entry_type=EntryType.MESSAGE,
            content="The quick brown fox",
            conversation_id="test-conv-7",
            importance_score=0.9,
        )

        entry2 = MemoryEntry(
            id="test-7b",
            entry_type=EntryType.MESSAGE,
            content="The lazy dog",
            conversation_id="test-conv-7",
            importance_score=0.5,
        )

        await graphdb_storage.save(entry1)
        await graphdb_storage.save(entry2)

        from contexo.storage.base import SearchQuery

        query = SearchQuery(
            query="fox",
            limit=10,
            min_score=0.0,
        )

        results = await graphdb_storage.search(query)
        assert len(results) >= 1
        assert any("fox" in r.entry.content for r in results)

    @pytest.mark.asyncio
    async def test_delete(self, graphdb_storage):
        """Test deleting entries."""
        entry = MemoryEntry(
            id="test-8",
            entry_type=EntryType.MESSAGE,
            content="To be deleted",
            conversation_id="test-conv-8",
        )

        await graphdb_storage.save(entry)
        assert await graphdb_storage.load("test-8") is not None

        deleted = await graphdb_storage.delete("test-8")
        assert deleted is True

        assert await graphdb_storage.load("test-8") is None

    @pytest.mark.asyncio
    async def test_collections(self, graphdb_storage):
        """Test conversation collection methods."""
        # Add entries to different conversations
        for conv_id in ["conv-a", "conv-b"]:
            for i in range(3):
                entry = MemoryEntry(
                    id=f"test-9-{conv_id}-{i}",
                    entry_type=EntryType.MESSAGE,
                    content=f"Message {i}",
                    conversation_id=conv_id,
                )
                await graphdb_storage.save(entry)

        # List collections
        collections = await graphdb_storage.list_collections()
        assert "conv-a" in collections
        assert "conv-b" in collections

        # Count by collection
        count_a = await graphdb_storage.count(collection="conv-a")
        assert count_a == 3

        # Clear collection
        await graphdb_storage.clear_collection("conv-a")
        count_a_after = await graphdb_storage.count(collection="conv-a")
        assert count_a_after == 0
