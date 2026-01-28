"""Example of using a custom storage backend with Contexo."""

import asyncio


async def main():
    """Demonstrate custom storage backend usage."""
    from contexo import Contexo
    from contexo.config import ContexoConfig, StorageConfig, EmbeddingConfig, WorkingMemoryConfig
    from contexo.storage import InMemoryStorage
    from contexo.embeddings import MockEmbeddings

    print("=== Custom Backend Example ===\n")

    # Create custom storage and embedding providers
    storage = InMemoryStorage()
    await storage.initialize()

    embeddings = MockEmbeddings(dimension=384, model_name="custom-mock")
    await embeddings.initialize()

    # Create a configuration that uses these custom providers
    config = ContexoConfig(
        storage=StorageConfig(backend_type="in_memory"),
        embeddings=EmbeddingConfig(provider_type="mock"),
        working_memory=WorkingMemoryConfig(
            max_tokens=2000,
            strategy="sliding_window",
        ),
    )

    # Create Contexo with custom components
    from contexo.working_memory.working_memory import WorkingMemory
    from contexo.persistent_memory.persistent_memory import PersistentMemory
    from contexo.working_memory.strategies import SlidingWindowStrategy

    working = WorkingMemory(
        max_tokens=2000,
        strategy=SlidingWindowStrategy(),
    )

    persistent = PersistentMemory(
        storage=storage,
        embedding_provider=embeddings,
        auto_embed=True,
    )

    await working.initialize()
    await persistent.initialize()

    # Create Contexo with custom memory components
    ctx = Contexo(
        config=config,
        working_memory=working,
        persistent_memory=persistent,
    )

    print("Using custom in-memory storage with mock embeddings...")

    # Add some messages
    await ctx.add_message("Testing custom backend setup", role="user")
    await ctx.add_message("Backend is working correctly!", role="assistant")

    # Verify it works
    context = await ctx.get_context()
    print(f"\nContext:\n{context}")

    stats = ctx.get_stats()
    print(f"\nStorage type: {stats['persistent_memory']['storage_type']}")
    print(f"Embedding provider: {stats['persistent_memory']['embedding_provider']}")

    # Clean up
    await ctx.close()
    await embeddings.close()
    await storage.close()

    print("\nCustom backend example complete!")


if __name__ == "__main__":
    asyncio.run(main())
