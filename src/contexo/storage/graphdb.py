"""Graph database storage backend with Neo4j support.

This backend provides relationship-based memory storage, allowing you to:
- Create relationships between memory entries
- Traverse relationships to find related memories
- Query by entity and relationship types
- Support for temporal and semantic relationships
"""

from __future__ import annotations

import json
from typing import Any

from contexo.core.exceptions import StorageError
from contexo.core.memory import EntryType, MemoryEntry
from contexo.storage.base import SearchQuery, SearchResult, StorageBackend

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver

    _neo4j_available = True
except ImportError:
    _neo4j_available = False

    AsyncDriver = Any  # type: ignore


class GraphDBStorage(StorageBackend):
    """Graph database storage backend using Neo4j.

    This backend stores memory entries as nodes and relationships
    as edges, enabling powerful relationship-based queries.

    Entry properties:
        id: Unique identifier (primary key)
        entry_type: Type of entry (MESSAGE, TOOL_CALL, etc.)
        content: The main content
        metadata: Additional metadata (JSON)
        timestamp: Creation timestamp
        token_count: Estimated token count
        importance_score: Importance (0.0-1.0)
        embedding: Vector embedding (list of floats)
        parent_id: ID of parent entry
        conversation_id: ID of conversation

    Relationship types:
        - PARENT: Parent-child relationship (e.g., tool_call -> tool_response)
        - CONVERSATION: Entries in the same conversation
        - RELATED: Semantic similarity
        - FOLLOWS: Temporal ordering
        - REFERENCES: Entry references another entry
    """

    # Default relationship types
    REL_PARENT = "PARENT"
    REL_CONVERSATION = "CONVERSATION"
    REL_RELATED = "RELATED"
    REL_FOLLOWS = "FOLLOWS"
    REL_REFERENCES = "REFERENCES"

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> None:
        """Initialize the Neo4j storage backend.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Database user
            password: Database password
            database: Database name (neo4j default)
        """
        if not _neo4j_available:
            raise ImportError(
                "neo4j is required for GraphDB storage. "
                "Install with: pip install neo4j"
            )

        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver: AsyncDriver | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the storage backend and create constraints."""
        if self._initialized:
            return

        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )

        # Create constraints for unique IDs
        await self._execute_query(
            "CREATE CONSTRAINT entry_id IF NOT EXISTS "
            "FOR (e:MemoryEntry) REQUIRE e.id IS UNIQUE"
        )

        # Create indexes for common queries
        await self._execute_query(
            "CREATE INDEX entry_conversation_id IF NOT EXISTS "
            "FOR (e:MemoryEntry) ON (e.conversation_id)"
        )

        await self._execute_query(
            "CREATE INDEX entry_entry_type IF NOT EXISTS "
            "FOR (e:MemoryEntry) ON (e.entry_type)"
        )

        await self._execute_query(
            "CREATE INDEX entry_timestamp IF NOT EXISTS "
            "FOR (e:MemoryEntry) ON (e.timestamp)"
        )

        self._initialized = True

    async def close(self) -> None:
        """Close the storage backend."""
        if self._driver:
            await self._driver.close()
            self._driver = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the storage backend has been initialized."""
        return self._initialized

    async def save(self, entry: MemoryEntry) -> MemoryEntry:
        """Save a memory entry as a node.

        Args:
            entry: The entry to save

        Returns:
            The saved entry
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        embedding_json = json.dumps(entry.embedding) if entry.embedding else None
        metadata_json = json.dumps(entry.metadata) if entry.metadata else None

        query = """
        MERGE (e:MemoryEntry {id: $id})
        SET e.entry_type = $entry_type,
            e.content = $content,
            e.metadata = $metadata,
            e.timestamp = $timestamp,
            e.token_count = $token_count,
            e.importance_score = $importance_score,
            e.embedding = $embedding,
            e.parent_id = $parent_id,
            e.conversation_id = $conversation_id
        RETURN e
        """

        await self._execute_query(
            query,
            {
                "id": entry.id,
                "entry_type": entry.entry_type.value,
                "content": entry.content,
                "metadata": metadata_json,
                "timestamp": entry.timestamp,
                "token_count": entry.token_count,
                "importance_score": entry.importance_score,
                "embedding": embedding_json,
                "parent_id": entry.parent_id,
                "conversation_id": entry.conversation_id,
            },
        )

        # Create automatic relationships
        await self._create_auto_relationships(entry)

        return entry

    async def _create_auto_relationships(self, entry: MemoryEntry) -> None:
        """Create automatic relationships for an entry.

        Args:
            entry: The entry to create relationships for
        """
        # Parent relationship
        if entry.parent_id:
            await self.create_relationship(
                from_id=entry.parent_id,
                to_id=entry.id,
                relation_type=self.REL_PARENT,
            )

        # Conversation relationship
        if entry.conversation_id:
            # Find other entries in the same conversation
            result = await self._execute_query(
                """
                MATCH (e:MemoryEntry {id: $id})
                MATCH (other:MemoryEntry)
                WHERE other.conversation_id = $conversation_id
                    AND other.id <> $id
                    AND NOT (e)-[:CONVERSATION]->(other)
                MERGE (e)-[:CONVERSATION]->(other)
                """,
                {"id": entry.id, "conversation_id": entry.conversation_id},
            )

        # Follows relationship (temporal ordering)
        await self._execute_query(
            """
            MATCH (e:MemoryEntry {id: $id})
            MATCH (prev:MemoryEntry)
            WHERE prev.conversation_id = $conversation_id
                AND prev.timestamp < $timestamp
                AND NOT EXISTS {
                    MATCH (e)-[:FOLLOWS]->(prev)
                }
            ORDER BY prev.timestamp DESC
            LIMIT 1
            MERGE (e)-[:FOLLOWS]->(prev)
            """,
            {"id": entry.id, "conversation_id": entry.conversation_id, "timestamp": entry.timestamp},
        )

    async def load(self, entry_id: str) -> MemoryEntry | None:
        """Load a memory entry by ID.

        Args:
            entry_id: The ID of the entry to load

        Returns:
            The entry if found, None otherwise
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        query = """
        MATCH (e:MemoryEntry {id: $id})
        RETURN e
        """

        result = await self._execute_query(query, {"id": entry_id})

        if not result:
            return None

        return self._record_to_entry(result[0]["e"])

    async def update(self, entry: MemoryEntry) -> MemoryEntry:
        """Update an existing memory entry.

        Args:
            entry: The entry to update

        Returns:
            The updated entry
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        # Check if entry exists
        existing = await self.load(entry.id)
        if existing is None:
            raise StorageError(f"Entry not found: {entry.id}")

        await self.save(entry)
        return entry

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID (also removes relationships).

        Args:
            entry_id: The ID of the entry to delete

        Returns:
            True if deleted, False if not found
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        query = """
        MATCH (e:MemoryEntry {id: $id})
        DETACH DELETE e
        RETURN count(e) as deleted
        """

        result = await self._execute_query(query, {"id": entry_id})
        return result[0]["deleted"] > 0 if result else False

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for entries using full-text search on content.

        Args:
            query: The search query

        Returns:
            List of search results sorted by relevance
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        # Build the Cypher query
        cypher = """
        MATCH (e:MemoryEntry)
        WHERE e.content CONTAINS $query
        """

        params: dict[str, Any] = {"query": query.query}

        # Add filters
        if query.entry_type is not None:
            cypher += " AND e.entry_type = $entry_type"
            params["entry_type"] = query.entry_type.value

        if query.conversation_id is not None:
            cypher += " AND e.conversation_id = $conversation_id"
            params["conversation_id"] = query.conversation_id

        if query.parent_id is not None:
            cypher += " AND e.parent_id = $parent_id"
            params["parent_id"] = query.parent_id

        # Add metadata filter
        if query.metadata_filter:
            for key, value in query.metadata_filter.items():
                cypher += f" AND e.metadata.${key} = $meta_{key}"
                params[f"meta_{key}"] = value

        # Add importance score filter
        if query.min_score > 0:
            cypher += " AND e.importance_score >= $min_score"
            params["min_score"] = query.min_score

        cypher += " RETURN e ORDER BY e.timestamp DESC LIMIT $limit"
        params["limit"] = query.limit

        result = await self._execute_query(cypher, params)

        results = []
        for record in result:
            entry = self._record_to_entry(record["e"])
            score = entry.importance_score
            results.append(SearchResult(entry=entry, score=score))

        return results

    async def list_collections(self) -> list[str]:
        """List all conversation IDs.

        Returns:
            List of conversation IDs
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        query = """
        MATCH (e:MemoryEntry)
        WHERE e.conversation_id IS NOT NULL
        RETURN DISTINCT e.conversation_id as conversation_id
        """

        result = await self._execute_query(query)
        return [r["conversation_id"] for r in result]

    async def count(
        self,
        collection: str | None = None,
        entry_type: EntryType | None = None,
    ) -> int:
        """Count entries in storage.

        Args:
            collection: Filter by conversation ID
            entry_type: Filter by entry type

        Returns:
            The count of matching entries
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        cypher = "MATCH (e:MemoryEntry)"
        conditions = []
        params: dict[str, Any] = {}

        if collection is not None:
            conditions.append("e.conversation_id = $conversation_id")
            params["conversation_id"] = collection

        if entry_type is not None:
            conditions.append("e.entry_type = $entry_type")
            params["entry_type"] = entry_type.value

        if conditions:
            cypher += " WHERE " + " AND ".join(conditions)

        cypher += " RETURN count(e) as count"

        result = await self._execute_query(cypher, params)
        return result[0]["count"] if result else 0

    async def list_entries(
        self,
        collection: str | None = None,
        entry_type: EntryType | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List entries from storage.

        Args:
            collection: Filter by conversation ID
            entry_type: Filter by entry type
            limit: Maximum number of entries to return
            offset: Number of entries to skip

        Returns:
            List of entries
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        cypher = "MATCH (e:MemoryEntry)"
        conditions = []
        params: dict[str, Any] = {}

        if collection is not None:
            conditions.append("e.conversation_id = $conversation_id")
            params["conversation_id"] = collection

        if entry_type is not None:
            conditions.append("e.entry_type = $entry_type")
            params["entry_type"] = entry_type.value

        if conditions:
            cypher += " WHERE " + " AND ".join(conditions)

        cypher += " RETURN e ORDER BY e.timestamp"

        if offset:
            cypher += " SKIP $offset"
            params["offset"] = offset

        if limit:
            cypher += " LIMIT $limit"
            params["limit"] = limit

        result = await self._execute_query(cypher, params)
        return [self._record_to_entry(r["e"]) for r in result]

    async def clear_collection(self, collection: str) -> None:
        """Clear all entries from a conversation.

        Args:
            collection: The conversation ID to clear
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        await self._execute_query(
            """
            MATCH (e:MemoryEntry {conversation_id: $conversation_id})
            DETACH DELETE e
            """,
            {"conversation_id": collection},
        )

    async def delete_collection(self, collection: str) -> None:
        """Delete a conversation and all its entries.

        Args:
            collection: The conversation ID to delete
        """
        await self.clear_collection(collection)

    # ============ Graph-Specific Methods ============

    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        relation_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create a relationship between two entries.

        Args:
            from_id: Source entry ID
            to_id: Target entry ID
            relation_type: Type of relationship
            properties: Optional properties for the relationship
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        props = properties or {}
        props_json = json.dumps(props) if props else None

        query = f"""
        MATCH (from:MemoryEntry {{id: $from_id}})
        MATCH (to:MemoryEntry {{id: $to_id}})
        MERGE (from)-[r:{relation_type}]->(to)
        SET r.properties = $properties
        """

        await self._execute_query(
            query,
            {"from_id": from_id, "to_id": to_id, "properties": props_json},
        )

    async def traverse(
        self,
        entry_id: str,
        relation_type: str | None = None,
        direction: str = "outgoing",
        depth: int = 1,
    ) -> list[MemoryEntry]:
        """Traverse relationships from an entry.

        Args:
            entry_id: Starting entry ID
            relation_type: Filter by relationship type (None = all)
            direction: "outgoing", "incoming", or "both"
            depth: Maximum traversal depth

        Returns:
            List of connected entries
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        # Build traversal pattern
        if direction == "outgoing":
            pattern = f"-[r:{relation_type}]->" if relation_type else "-[r]->"
        elif direction == "incoming":
            pattern = f"<-[r:{relation_type}]-" if relation_type else "<-[r]-"
        else:  # both
            pattern = f"-[r:{relation_type}]-" if relation_type else "-[r]-"

        query = f"""
        MATCH (start:MemoryEntry {{id: $entry_id}})
        MATCH (start){pattern}(connected:MemoryEntry)
        RETURN DISTINCT connected
        LIMIT $limit
        """

        result = await self._execute_query(
            query,
            {"entry_id": entry_id, "limit": 1000},
        )

        return [self._record_to_entry(r["connected"]) for r in result]

    async def find_related(
        self,
        entry_id: str,
        relation_types: list[str] | None = None,
        min_score: float = 0.0,
        limit: int = 10,
    ) -> list[tuple[MemoryEntry, str, float]]:
        """Find entries related to an entry via relationships.

        Args:
            entry_id: The entry ID to find relations for
            relation_types: Filter by relationship types
            min_score: Minimum importance score
            limit: Maximum results

        Returns:
            List of (entry, relation_type, score) tuples
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        rel_filter = ""
        if relation_types:
            rel_pattern = "|".join(relation_types)
            rel_filter = f":{rel_pattern}"

        query = f"""
        MATCH (e:MemoryEntry {{id: $entry_id}})-[r{rel_filter}]-(related:MemoryEntry)
        WHERE related.importance_score >= $min_score
        RETURN related, type(r) as relation_type, related.importance_score as score
        ORDER BY score DESC
        LIMIT $limit
        """

        result = await self._execute_query(
            query,
            {"entry_id": entry_id, "min_score": min_score, "limit": limit},
        )

        return [
            (self._record_to_entry(r["related"]), r["relation_type"], r["score"])
            for r in result
        ]

    async def find_entities(
        self,
        entity_name: str,
        relation_type: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Find all entries related to an entity.

        This searches for entries that mention an entity or are
        connected via relationships.

        Args:
            entity_name: The entity name to search for
            relation_type: Optional filter by relationship type
            limit: Maximum results

        Returns:
            List of matching entries
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        # Search content first
        query = """
        MATCH (e:MemoryEntry)
        WHERE e.content CONTAINS $entity_name
        """

        params: dict[str, Any] = {"entity_name": entity_name, "limit": limit}

        if relation_type:
            # Also search via relationships
            query = f"""
            MATCH (e:MemoryEntry)
            WHERE e.content CONTAINS $entity_name
            UNION
            MATCH (e:MemoryEntry)-[:{relation_type}]-(related:MemoryEntry)
            WHERE related.content CONTAINS $entity_name
            RETURN related
            """
        else:
            query += " RETURN e"

        query += " LIMIT $limit"

        result = await self._execute_query(query, params)

        entries = []
        for r in result:
            node = r.get("e") or r.get("related")
            if node:
                entries.append(self._record_to_entry(node))

        return entries

    async def find_shortest_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 5,
    ) -> list[MemoryEntry] | None:
        """Find the shortest path between two entries.

        Args:
            from_id: Starting entry ID
            to_id: Target entry ID
            max_depth: Maximum path length

        Returns:
            List of entries in the path, or None if no path found
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        query = f"""
        MATCH path = shortestPath(
            (start:MemoryEntry {{id: $from_id}})-[*1..{max_depth}]-(end:MemoryEntry {{id: $to_id}})
        )
        RETURN [node in nodes(path) | node] as nodes
        """

        result = await self._execute_query(
            query,
            {"from_id": from_id, "to_id": to_id},
        )

        if not result:
            return None

        nodes = result[0]["nodes"]
        return [self._record_to_entry(node) for node in nodes]

    async def get_conversation_context(
        self,
        entry_id: str,
        window_size: int = 5,
    ) -> list[MemoryEntry]:
        """Get the context around an entry in its conversation.

        Uses FOLLOWS relationships to get preceding and following entries.

        Args:
            entry_id: The entry ID to get context for
            window_size: Number of entries before and after

        Returns:
            List of entries in temporal order
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        query = """
        MATCH (target:MemoryEntry {id: $entry_id})
        MATCH (target)-[:FOLLOWS*0..$size]-(context:MemoryEntry)
        WHERE context.conversation_id = target.conversation_id
        RETURN DISTINCT context
        ORDER BY context.timestamp
        """

        result = await self._execute_query(
            query,
            {"entry_id": entry_id, "size": window_size},
        )

        return [self._record_to_entry(r["context"]) for r in result]

    async def get_provenance_context(
        self,
        entry_id: str,
        include_parent: bool = True,
        include_children: bool = True,
        include_conversation: bool = True,
        conversation_window: int = 5,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Get full provenance context for an entry using graph relationships.

        This is optimized for Neo4j's graph traversal capabilities.

        Args:
            entry_id: The entry ID to get context for
            include_parent: Include parent entry (via PARENT relationship)
            include_children: Include child entries (reverse PARENT relationship)
            include_conversation: Include conversation context
            conversation_window: Size of conversation window
            max_depth: Max depth for relationship traversal

        Returns:
            Dictionary with provenance context
        """
        if not self.is_initialized:
            raise StorageError("Storage backend not initialized")

        # Get the target entry
        target = await self.load(entry_id)
        if target is None:
            return {
                "entry": None,
                "parent": None,
                "children": [],
                "conversation_context": [],
                "related": [],
                "relationship_graph": {},
            }

        result: dict[str, Any] = {
            "entry": target,
            "parent": None,
            "children": [],
            "conversation_context": [],
            "related": [],
            "relationship_graph": {"nodes": [entry_id], "edges": []},
        }

        # Get parent via PARENT relationship
        if include_parent:
            query = """
            MATCH (e:MemoryEntry {id: $entry_id})<-[:PARENT]-(parent:MemoryEntry)
            RETURN parent
            LIMIT 1
            """
            parent_result = await self._execute_query(query, {"entry_id": entry_id})
            if parent_result:
                parent = self._record_to_entry(parent_result[0]["parent"])
                result["parent"] = parent
                result["relationship_graph"]["nodes"].append(parent.id)
                result["relationship_graph"]["edges"].append({
                    "from": parent.id,
                    "to": entry_id,
                    "type": "PARENT",
                })

        # Get children (entries that have this as parent)
        if include_children:
            query = """
            MATCH (e:MemoryEntry {id: $entry_id})-[:PARENT]->(child:MemoryEntry)
            RETURN child
            ORDER BY child.timestamp
            """
            child_result = await self._execute_query(query, {"entry_id": entry_id})
            for record in child_result:
                child = self._record_to_entry(record["child"])
                result["children"].append(child)
                result["relationship_graph"]["nodes"].append(child.id)
                result["relationship_graph"]["edges"].append({
                    "from": entry_id,
                    "to": child.id,
                    "type": "PARENT",
                })

        # Get conversation context
        if include_conversation:
            context_entries = await self.get_conversation_context(
                entry_id,
                window_size=conversation_window,
            )
            result["conversation_context"] = context_entries

            # Add conversation relationships to graph
            for entry in context_entries:
                if entry.id != entry_id and entry.id not in result["relationship_graph"]["nodes"]:
                    result["relationship_graph"]["nodes"].append(entry.id)
                    result["relationship_graph"]["edges"].append({
                        "from": entry.id,
                        "to": entry_id,
                        "type": "CONVERSATION",
                    })

        # Get related entries via all relationships
        query = """
        MATCH (e:MemoryEntry {id: $entry_id})-[r]-(related:MemoryEntry)
        WHERE related.id <> $entry_id
        RETURN DISTINCT related, type(r) as rel_type, related.importance_score as score
        ORDER BY score DESC
        LIMIT 10
        """
        related_result = await self._execute_query(query, {"entry_id": entry_id})
        for record in related_result:
            related = self._record_to_entry(record["related"])
            result["related"].append({
                "entry": related,
                "relation_type": record["rel_type"],
                "score": record["score"],
            })
            if related.id not in result["relationship_graph"]["nodes"]:
                result["relationship_graph"]["nodes"].append(related.id)
                result["relationship_graph"]["edges"].append({
                    "from": entry_id,
                    "to": related.id,
                    "type": record["rel_type"],
                })

        return result

    # ============ Helper Methods ============

    async def _execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query.

        Args:
            query: The Cypher query string
            params: Query parameters

        Returns:
            List of result records
        """
        if not self._driver:
            raise StorageError("Driver not initialized")

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params or {})
            records = []
            async for record in result:
                # Convert Node objects to dicts
                record_dict = {}
                for key, value in record.items():
                    if hasattr(value, "items"):  # Node or Relationship
                        record_dict[key] = dict(value)
                    else:
                        record_dict[key] = value
                records.append(record_dict)
            return records

    def _record_to_entry(self, node: dict[str, Any]) -> MemoryEntry:
        """Convert a Neo4j node to a MemoryEntry.

        Args:
            node: The Neo4j node as a dictionary

        Returns:
            A MemoryEntry instance
        """
        embedding = None
        if node.get("embedding"):
            try:
                embedding = json.loads(node["embedding"])
            except (json.JSONDecodeError, TypeError):
                pass

        metadata = {}
        if node.get("metadata"):
            try:
                metadata = json.loads(node["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        return MemoryEntry(
            id=node["id"],
            entry_type=EntryType(node["entry_type"]),
            content=node["content"],
            metadata=metadata,
            timestamp=node["timestamp"],
            token_count=node["token_count"],
            importance_score=node["importance_score"],
            embedding=embedding,
            parent_id=node.get("parent_id"),
            conversation_id=node.get("conversation_id"),
        )

    @property
    def supports_semantic_search(self) -> bool:
        """Check if this backend supports semantic search."""
        return False

    @property
    def supports_fts(self) -> bool:
        """Check if this backend supports full-text search."""
        return True  # Neo4j has full-text search capabilities


__all__ = ["GraphDBStorage"]
