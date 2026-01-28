"""LangChain integration example with Contexo.

This example demonstrates how to use Contexo as a memory backend
for LangChain chains, providing persistent, searchable context.
"""

import asyncio
import os
import sys
from typing import Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def main():
    """Run the LangChain + Contexo example."""
    from contexo import Contexo
    from contexo.config import local_config

    print("=== LangChain + Contexo Integration ===\n")

    # Configure Contexo
    config = local_config(
        db_path="./langchain_context.db",
        model_name="all-MiniLM-L6-v2",
        max_tokens=8192,
    )

    contexo = Contexo(config=config)
    await contexo.initialize()
    contexo.set_conversation_id("langchain-demo")

    # ==================== METHOD 1: Direct Integration ====================
    print("--- Method 1: Direct Integration ---\n")

    # Simulate a conversation
    print("User: Hi, my name is Alice and I'm a software engineer.")
    await contexo.add_message(
        "Hi, my name is Alice and I'm a software engineer.",
        role="user",
        importance=0.9,
    )

    print("Assistant: Hi Alice! Nice to meet you. What kind of engineering do you do?")
    await contexo.add_message(
        "Hi Alice! Nice to meet you. What kind of engineering do you do?",
        role="assistant",
        importance=0.7,
    )

    print("User: I work on distributed systems and ML infrastructure.")
    await contexo.add_message(
        "I work on distributed systems and ML infrastructure.",
        role="user",
        importance=0.9,
    )

    print("Assistant: That's fascinating! Are you working on any specific projects?")
    await contexo.add_message(
        "That's fascinating! Are you working on any specific projects?",
        role="assistant",
        importance=0.7,
    )

    # In a real integration, you would now call the LLM with context from Contexo
    context = await contexo.get_context()
    print("\n--- Context that would be sent to LLM ---")
    print(context)

    # ==================== METHOD 2: Using ContexoMemory Class ====================
    print("\n--- Method 2: Using ContexoMemory Class ---\n")

    from contexo.integrations import ContexoMemory

    memory = ContexoMemory(
        contexo=contexo,
        conversation_id="langchain-demo-2",
        max_context_messages=10,
        search_relevant_context=True,
    )

    # Save a conversation turn
    inputs = {"input": "What's my name?"}
    outputs = {"response": "Your name is Alice!"}

    await memory.asave_context(inputs, outputs)
    print("Saved conversation turn to Contexo")

    # Get context for next turn
    context_str = await memory.aget_context("What do I do for work?")
    print("\n--- Retrieved Context ---")
    print(context_str[:200] + "...")

    # ==================== METHOD 3: With LangChain Chain (if available) ====================
    print("\n--- Method 3: With LangChain Chain ---\n")

    try:
        from langchain.callbacks import AsyncCallbackHandler
        from langchain.chains import ConversationChain
        from langchain.llms import OpenAI

        class ContexoCallbackHandler(AsyncCallbackHandler):
            """Callback to track provenance in LangChain."""

            def __init__(self, contexo_instance: Contexo):
                self.contexo = contexo_instance

            async def on_llm_start(self, prompts: list[str], **kwargs: Any) -> Any:
                """Called when LLM starts processing."""
                print(f"[Contexo] LLM processing with context length: {len(prompts[0])} chars")

            async def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
                """Called when LLM finishes."""
                print("[Contexo] LLM response received")

        # Create a simple chain with Contexo memory
        llm = OpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
        memory2 = ContexoMemory(
            contexo=contexo,
            conversation_id="langchain-chain-demo",
        )

        # Note: This is a simplified example
        # In practice, you'd use ConversationChain which integrates with memory
        print("LangChain is available!")
        print("To use with ConversationChain:")
        print("  chain = ConversationChain(llm=llm, memory=memory)")
        print("  response = await chain.apredict(input='Tell me about Alice')")

    except ImportError:
        print("LangChain not installed. Install with:")
        print("  pip install langchain openai")

    # ==================== Provenance Tracking ====================
    print("\n--- Provenance Tracking ---\n")

    # Trace a specific message
    context_data = await contexo.get_message_context(
        query="Alice",
        include_conversation=True,
        conversation_window=3,
    )

    if context_data["entry"]:
        print(f"Found message: {context_data['entry'].content[:60]}...")
        print(f"Conversation context: {len(context_data['conversation_context'])} messages")

    # ==================== Summary ====================
    print("\n=== Summary ===\n")

    stats = contexo.get_stats()
    print(f"Total entries stored: {stats['working_memory']['total_entries']}")
    print(f"Storage type: {stats['persistent_memory']['storage_type']}")

    await contexo.close()

    print("\n=== Key Benefits of Contexo + LangChain ===")
    print("  ✓ Persistent memory across sessions")
    print("  ✓ Semantic search for context retrieval")
    print("  ✓ Automatic context compaction")
    print("  ✓ Provenance tracking for all messages")
    print("  ✓ Works with any LLM (OpenAI, Anthropic, local models)")


if __name__ == "__main__":
    asyncio.run(main())
