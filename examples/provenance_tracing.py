"""Example demonstrating provenance tracing and evidence gathering.

This example shows how to trace the full context behind a specific message,
including related tool calls, conversation history, and semantic relationships.
This is useful for understanding "why did the AI say this?" or providing evidence
for decision-making.
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


async def main():
    """Demonstrate provenance tracing with Contexo."""
    from contexo import Contexo, minimal_config

    print("=== Contexo Provenance Tracing Example ===\n")

    # Create a Contexo instance
    ctx = Contexo(config=minimal_config())
    await ctx.initialize()

    # Set up a conversation
    ctx.set_conversation_id("provenance-demo")

    # Build a conversation with tool calls
    print("Building conversation...")

    # User asks about weather
    await ctx.add_message(
        "What's the weather like in San Francisco and Tokyo?",
        role="user",
        importance=0.8,
    )

    # Assistant decides to call weather tools
    await ctx.add_message(
        "I'll check the weather for both cities.",
        role="assistant",
        importance=0.6,
    )

    # Tool calls
    sf_call = await ctx.add_tool_call(
        "get_weather",
        {"location": "San Francisco", "units": "celsius"},
        importance=0.5,
    )

    tokyo_call = await ctx.add_tool_call(
        "get_weather",
        {"location": "Tokyo", "units": "celsius"},
        importance=0.5,
    )

    # Tool responses
    await ctx.add_tool_response(
        "get_weather",
        {"temperature": 22, "condition": "sunny", "humidity": 65},
        call_id=sf_call.id,
        importance=0.7,
    )

    await ctx.add_tool_response(
        "get_weather",
        {"temperature": 28, "condition": "partly_cloudy", "humidity": 70},
        call_id=tokyo_call.id,
        importance=0.7,
    )

    # Final response
    final_response = await ctx.add_message(
        "San Francisco is 22°C and sunny, while Tokyo is 28°C and partly cloudy.",
        role="assistant",
        importance=0.9,
    )

    # Follow-up question
    await ctx.add_message(
        "That's quite a difference! What about New York?",
        role="user",
        importance=0.8,
    )

    print(f"Created {len(await ctx.working_memory.list_all())} entries\n")

    # ============ PROVENANCE TRACING ============

    print("--- Getting Context for Final Response ---\n")

    # Get context for the final assistant response
    context = await ctx.get_message_context(
        message_id=final_response.id,
        include_conversation=True,
        conversation_window=3,
        include_tool_calls=True,
        include_related=True,
    )

    print(context["evidence_summary"])

    # ============ FIND BY QUERY ============

    print("\n--- Finding Message by Query ---\n")

    # You can also find context by searching for the message content
    context_by_query = await ctx.get_message_context(
        query="Tokyo is 28°C",
        include_conversation=True,
        conversation_window=2,
    )

    if context_by_query["entry"]:
        print(f"Found message: {context_by_query['entry'].content}")

        # Show what tool calls were related
        if context_by_query["children"]:
            print("\nRelated tool calls/responses:")
            for child in context_by_query["children"]:
                print(f"  - {child.entry_type.value}: {child.content[:60]}...")

    # ============ TRACE TOOL CALL ============

    print("\n--- Tracing Tool Call Context ---\n")

    # Get context for a tool call to see what led to it
    tool_call_context = await ctx.get_message_context(
        message_id=sf_call.id,
        include_conversation=True,
        conversation_window=2,
    )

    print(f"Tool Call: {tool_call_context['entry'].content}")

    if tool_call_context["parent"]:
        parent = tool_call_context["parent"]
        print(f"Triggered by: {parent.entry_type.value} - {parent.content}")

    if tool_call_context["children"]:
        print(f"Resulted in: {len(tool_call_context['children'])} response(s)")
        for child in tool_call_context["children"]:
            print(f"  - {child.content[:60]}...")

    # ============ EXPORT EVIDENCE ============

    print("\n--- Exporting Evidence as JSON ---\n")

    # Export the evidence data for external use
    from contexo.utils.serialization import serialize_entry

    evidence_export = {
        "target_message": serialize_entry(context["entry"]),
        "parent": serialize_entry(context["parent"]) if context["parent"] else None,
        "children": [serialize_entry(c) for c in context["children"]],
        "conversation_context": [
            serialize_entry(e) for e in context["conversation_context"]
        ],
        "related_count": len(context["related"]),
        "relationship_graph": context.get("relationship_graph", {}),
    }

    import json
    print(json.dumps(evidence_export, indent=2)[:500] + "...")

    # ============ SUMMARY ============

    print("\n--- Summary ---\n")

    stats = ctx.get_stats()
    print(f"Total messages in memory: {stats['working_memory']['total_entries']}")
    print(f"Conversation ID: {ctx.conversation_id}")
    print("\nProvenance tracing allows you to:")
    print("  - Find the full context behind any message")
    print("  - Trace tool calls and their responses")
    print("  - See what conversation context was used")
    print("  - Export evidence for audit/review")

    await ctx.close()
    print("\nExample complete!")


if __name__ == "__main__":
    asyncio.run(main())
