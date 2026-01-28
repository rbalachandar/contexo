"""Example of using custom embedding functions with Contexo."""

import asyncio


async def main():
    """Demonstrate custom embedding function usage."""
    from contexo import Contexo
    from contexo.config import local_config
    from contexo.embeddings.base import EmbeddingProvider

    print("=== Custom Embeddings Example ===\n")

    # Option 1: Use a local model with sentence-transformers
    try:
        from contexo.embeddings import SentenceTransformersEmbeddings

        print("Using sentence-transformers for local embeddings...")

        config = local_config(
            db_path="./custom_embeddings.db",
            model_name="all-MiniLM-L6-v2",  # Fast, decent quality
        )

        ctx = Contexo(config=config)
        await ctx.initialize()

        # Add some messages
        await ctx.add_message("I love programming in Python!", role="user", importance=0.9)
        await ctx.add_message("Python is a great language for data science", role="assistant")

        # Semantic search should find related content
        print("\n--- Semantic Search ---")
        results = await ctx.search_memory("coding", limit=5)
        print(f"Found {len(results)} results for 'coding':")
        for r in results:
            print(f"  (score: {r.score:.2f}) {r.entry.content}")

        await ctx.close()

    except ImportError:
        print("sentence-transformers not installed, skipping local embedding example")

    # Option 2: Define a custom embedding provider
    print("\n--- Custom Embedding Provider ---")

    class CustomEmbeddingProvider(EmbeddingProvider):
        """Custom embedding provider using hash-based vectors."""

        def __init__(self, dimension: int = 128):
            self._dimension = dimension
            self._initialized = False

        @property
        def dimension(self) -> int:
            return self._dimension

        @property
        def model_name(self) -> str:
            return "custom-hash"

        async def initialize(self) -> None:
            self._initialized = True

        async def close(self) -> None:
            self._initialized = False

        @property
        def is_initialized(self) -> bool:
            return self._initialized

        async def embed(self, text: str) -> list[float]:
            # Generate a deterministic hash-based embedding
            import hashlib

            hash_bytes = hashlib.sha256(text.encode()).digest()
            values = []
            for i in range(self._dimension):
                byte_idx = i % len(hash_bytes)
                value = (hash_bytes[byte_idx] / 127.5) - 1.0
                values.append(value)
            return values

        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [await self.embed(text) for text in texts]

    # Use the custom provider
    from contexo.persistent_memory.persistent_memory import PersistentMemory
    from contexo.storage import InMemoryStorage

    storage = InMemoryStorage()
    await storage.initialize()

    custom_provider = CustomEmbeddingProvider(dimension=128)
    await custom_provider.initialize()

    persistent = PersistentMemory(
        storage=storage,
        embedding_provider=custom_provider,
        auto_embed=True,
    )
    await persistent.initialize()

    # Test the custom embeddings
    from contexo.core.memory import EntryType, MemoryEntry

    entry1 = MemoryEntry(
        entry_type=EntryType.MESSAGE,
        content="machine learning is fascinating",
        conversation_id="test",
    )
    entry2 = MemoryEntry(
        entry_type=EntryType.MESSAGE,
        content="deep learning models are powerful",
        conversation_id="test",
    )

    await persistent.add(entry1)
    await persistent.add(entry2)

    # Search using the custom embeddings
    results = await persistent.search("AI and neural networks", limit=5)

    print(f"Custom provider search results: {len(results)}")
    for r in results:
        print(f"  (score: {r.score:.3f}) {r.entry.content}")

    # Clean up
    await persistent.close()
    await custom_provider.close()
    await storage.close()

    print("\nCustom embeddings example complete!")


if __name__ == "__main__":
    asyncio.run(main())
