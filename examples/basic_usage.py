"""Basic usage example for Contexo.

This example demonstrates the core functionality of the Contexo library.
"""

import asyncio


async def main():
    """Run a basic usage demonstration."""
    # Import the library
    from contexo import Contexo
    from contexo.config import minimal_config

    # Create a Contexo instance with minimal configuration
    # (in-memory storage, no external dependencies)
    ctx = Contexo(config=minimal_config())
    await ctx.initialize()

    print("=== Contexo Basic Usage Demo ===\n")

    # Add messages to the conversation
    print("Adding messages to memory...")
    await ctx.add_message(
        "Hello! My name is Alice and I'm learning Python.",
        role="user",
        importance=0.8,
    )

    await ctx.add_message(
        "Hi Alice! I'd be happy to help you learn Python. What would you like to know?",
        role="assistant",
        importance=0.7,
    )

    await ctx.add_message(
        "I'm interested in learning about dataclasses. Can you explain them?",
        role="user",
        importance=0.9,
    )

    # Get the current context
    print("\n--- Current Context ---")
    context = await ctx.get_context()
    print(context)

    # Get statistics
    print("\n--- Statistics ---")
    stats = ctx.get_stats()
    print(f"Working memory entries: {stats['working_memory']['total_entries']}")
    print(f"Working memory tokens: {stats['working_memory']['total_tokens']}")
    print(f"Utilization: {stats['working_memory']['utilization']:.1%}")

    # Search through memory
    print("\n--- Search Results ---")
    results = await ctx.search_memory("Python", limit=5)
    print(f"Found {len(results)} results for 'Python':")
    for result in results:
        print(f"  - [{result.entry_type.value}] {result.content[:60]}...")

    # Add a tool call example
    print("\n--- Adding Tool Call ---")
    tool_call = await ctx.add_tool_call(
        tool_name="get_weather",
        arguments={"location": "San Francisco", "units": "celsius"},
    )
    print(f"Added tool call: {tool_call.id}")

    # Add tool response
    tool_response = await ctx.add_tool_response(
        tool_name="get_weather",
        result={"temperature": 22, "condition": "sunny"},
        call_id=tool_call.id,
    )
    print(f"Added tool response: {tool_response.id}")

    # Get updated context
    print("\n--- Updated Context ---")
    context = await ctx.get_context()
    print(context)

    # Clear working memory
    print("\n--- Clearing Working Memory ---")
    await ctx.clear_working_memory()
    stats = ctx.get_stats()
    print(f"Working memory entries after clear: {stats['working_memory']['total_entries']}")

    # Clean up
    await ctx.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
