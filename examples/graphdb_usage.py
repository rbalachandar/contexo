"""Example demonstrating GraphDB storage backend with Neo4j.

This example shows how to use Contexo with Neo4j for relationship-based
memory storage, enabling powerful queries on connections between memories.

Requirements:
    pip install neo4j

    You also need a running Neo4j instance:
    - Using Docker: docker run -p 7687:7687 -p 7474:7474 neo4j:latest
    - Or download from: https://neo4j.com/download/
"""

import asyncio


async def main():
    """Demonstrate GraphDB usage with Contexo."""
    from contexo import graphdb_config

    print("=== Contexo GraphDB Example ===\n")

    # Configuration for Neo4j
    config = graphdb_config(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",  # Change to your Neo4j password
        database="neo4j",
    )

    # Note: For this example to work, you need Neo4j running
    # Let's show what the configuration would look like
    print("GraphDB Configuration:")
    print(f"  Storage backend: {config.storage.backend_type}")
    print(f"  Connection: {config.storage.connection_string}")
    print(f"  Working memory tokens: {config.working_memory.max_tokens}")
    print(f"  Strategy: {config.working_memory.strategy}")

    # If you have Neo4j running, uncomment the following:
    """
    ctx = Contexo(config=config)
    await ctx.initialize()

    # Set up a conversation
    ctx.set_conversation_id("demo-conversation")

    # Add some messages with relationships
    print("\n--- Adding Messages ---")
    await ctx.add_message(
        "My name is Alice and I'm a software engineer.",
        role="user",
        importance=0.9,
    )

    await ctx.add_message(
        "Hi Alice! It's nice to meet you. What do you work on?",
        role="assistant",
        importance=0.7,
    )

    await ctx.add_message(
        "I work on distributed systems and machine learning.",
        role="user",
        importance=0.8,
    )

    # Add a tool call with parent relationship
    tool_call = await ctx.add_tool_call(
        "search_papers",
        {"topic": "distributed systems", "limit": 5},
        importance=0.6,
    )

    # Add tool response (references the tool call)
    await ctx.add_tool_response(
        "search_papers",
        {
            "papers": [
                {"title": "Distributed Consensus", "year": 2020},
                {"title": "Scalable ML Systems", "year": 2021},
            ]
        },
        call_id=tool_call.id,
        importance=0.7,
    )

    print(f"Added {len(await ctx.working_memory.list_all())} entries")

    # Access the GraphDB storage for relationship queries
    storage = ctx.persistent_memory.storage
    if isinstance(storage, GraphDBStorage):
        print("\n--- Relationship Queries ---")

        # Find related entries
        entry_list = await storage.list_entries()
        if entry_list:
            first_entry = entry_list[0]

            # Get outgoing relationships
            related = await storage.find_related(
                first_entry.id,
                relation_types=["PARENT", "FOLLOWS", "CONVERSATION"],
            )

            print(f"Found {len(related)} entries related to {first_entry.id[:8]}...")
            for entry, rel_type, score in related[:3]:
                print(f"  [{rel_type}] {entry.entry_type.value}: {entry.content[:50]}...")

            # Get conversation context
            context_entries = await storage.get_conversation_context(
                first_entry.id,
                window_size=2,
            )
            print(f"\nConversation context ({len(context_entries)} entries):")
            for entry in context_entries:
                print(f"  - {entry.entry_type.value}: {entry.content[:50]}...")

            # Find entities
            alice_entries = await storage.find_entities("Alice")
            print(f"\nFound {len(alice_entries)} entries mentioning 'Alice'")

    # Custom relationship example
    if isinstance(storage, GraphDBStorage):
        print("\n--- Creating Custom Relationships ---")

        # Create a semantic "RELATED" relationship
        entries = await storage.list_entries()
        if len(entries) >= 2:
            await storage.create_relationship(
                from_id=entries[0].id,
                to_id=entries[1].id,
                relation_type="RELATED",
                properties={"similarity": 0.85, "reason": "same_topic"},
            )
            print(f"Created RELATED relationship between {entries[0].id[:8]}... and {entries[1].id[:8]}...")

        # Traverse relationships
        if entries:
            connected = await storage.traverse(
                entries[0].id,
                relation_type="CONVERSATION",
                direction="both",
                depth=2,
            )
            print(f"Traversal found {len(connected)} connected entries")

    # Get context for LLM
    print("\n--- LLM Context ---")
    context = await ctx.get_context()
    print(context[:200] + "...")

    # Statistics
    stats = ctx.get_stats()
    print(f"\n--- Statistics ---")
    print(f"Working memory entries: {stats['working_memory']['total_entries']}")
    print(f"Storage type: {stats['persistent_memory']['storage_type']}")

    await ctx.close()
    print("\nExample complete!")
    """

    print("\n=== To run this example with Neo4j ===")
    print("1. Start Neo4j:")
    print("   docker run -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:latest")
    print("\n2. Uncomment the code block in this example")
    print("\n3. Run the script:")
    print("   python examples/graphdb_usage.py")


if __name__ == "__main__":
    asyncio.run(main())
