"""Example demonstrating sectioned working memory for chat applications.

This example shows how to use Contexo's sectioned working memory to manage
different types of context with separate token limits and eviction priorities.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from contexo import Contexo, chat_config
from contexo.core.memory import MemoryEntry, EntryType


async def main():
    """Demonstrate sectioned working memory."""
    print("=== Sectioned Working Memory Example ===\n")

    # Create Contexo with sectioned memory (8K context window)
    config = chat_config(max_tokens=8192)

    # Or customize sections
    config = chat_config(
        max_tokens=8192,
        sections={
            "system": {"max_tokens": 500, "priority": 1.0, "pinned": True},
            "user_profile": {"max_tokens": 300, "pinned": True},
            "conversation": {"max_tokens": 5000},  # Largest allocation
            "rag_context": {"max_tokens": 1200},
            "tools": {"max_tokens": 500, "priority": "low"},
        }
    )

    ctx = Contexo(config=config)
    await ctx.initialize()
    ctx.set_conversation_id("sectioned-demo")

    # ==================== SYSTEM SECTION (pinned) ====================

    print("--- Adding System Prompt (Pinned Section) ---")
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.SYSTEM,
            content="You are a helpful AI assistant. Be concise and friendly.",
        ),
        section="system",
    )
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.SYSTEM,
            content="Always maintain professional tone even if user is casual.",
        ),
        section="system",
    )

    # ==================== USER PROFILE (pinned) ====================

    print("\n--- Adding User Profile (Pinned Section) ---")
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.MESSAGE,
            content="User name: Alice",
            metadata={"type": "preference", "key": "name"},
        ),
        section="user_profile",
    )
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.MESSAGE,
            content="User is a software engineer interested in Python and AI.",
            metadata={"type": "preference"},
        ),
        section="user_profile",
    )

    # ==================== CONVERSATION SECTION ====================

    print("\n--- Adding Conversation Messages ---")
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.MESSAGE,
            content="Hi, I'm Alice and I work on distributed systems.",
            metadata={"role": "user"},
        ),
        section="conversation",
    )
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.MESSAGE,
            content="Hello Alice! Distributed systems is a fascinating field. What specific areas?",
            metadata={"role": "assistant"},
        ),
        section="conversation",
    )
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.MESSAGE,
            content="I'm working on consensus algorithms and data pipelines.",
            metadata={"role": "user"},
        ),
        section="conversation",
    )
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.MESSAGE,
            content="That sounds interesting! Are you using any specific frameworks?",
            metadata={"role": "assistant"},
        ),
        section="conversation",
    )

    # ==================== RAG CONTEXT SECTION ====================

    print("\n--- Adding RAG Context ---")
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.MESSAGE,
            content="Retrieved: Raft consensus algorithm documentation...",
            metadata={"type": "rag", "source": "docs"},
        ),
        section="rag_context",
    )
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.MESSAGE,
            content="Retrieved: Kafka stream processing guide...",
            metadata={"type": "rag", "source": "docs"},
        ),
        section="rag_context",
    )

    # ==================== TOOLS SECTION (low priority) ====================

    print("\n--- Adding Tool Results ---")
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.TOOL_RESPONSE,
            content="search_papers returned 5 relevant papers on consensus.",
            metadata={"tool": "search_papers"},
        ),
        section="tools",
    )
    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.TOOL_RESPONSE,
            content="github_api returned 3 repositories for distributed systems.",
            metadata={"tool": "github_search"},
        ),
        section="tools",
    )

    # ==================== SHOW STATS ====================

    stats = ctx.working_memory.get_stats()
    print("\n=== Working Memory Stats ===")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Total tokens: {stats['total_tokens']}/{stats['max_tokens']}")
    print(f"Utilization: {stats['utilization']:.1%}")
    print(f"Sectioned mode: {stats['sectioned_mode']}")

    if stats["sectioned_mode"]:
        print("\n=== Section Breakdown ===")
        for section_stat in stats["sections"]:
            print(f"  {section_stat.name}:")
            print(f"    Entries: {section_stat.entry_count}")
            print(f"    Tokens: {section_stat.token_count}/{section_stat.max_tokens}")
            print(f"    Utilization: {section_stat.utilization:.1%}")
            print(f"    Priority: {section_stat.priority}")
            print(f"    Pinned: {section_stat.pinned}")

    # ==================== FORMATTED OUTPUT ====================

    print("\n=== Formatted Context (by section) ===")
    context = ctx.working_memory.format_context(by_section=True)
    print(context)

    # ==================== ADD MORE MESSAGES TO SHOW COMPACTION ====================

    print("\n--- Adding many messages to demonstrate auto-compaction ---")

    for i in range(20):
        await ctx.working_memory.add(
            MemoryEntry(
                entry_type=EntryType.MESSAGE,
                content=f"Conversation message {i+1}",
                metadata={"role": "user" if i % 2 == 0 else "assistant"},
            ),
            section="conversation",
        )

    stats_after = ctx.working_memory.get_stats()
    print(f"\nAfter adding 20 more messages:")
    print(f"  Total entries: {stats_after['total_entries']}")
    print(f"  Total tokens: {stats_after['total_tokens']}/{stats_after['max_tokens']}")
    print(f"  Utilization: {stats_after['utilization']:.1%}")

    if stats_after["sectioned_mode"]:
        print("\nSection breakdown after compaction:")
        for section_stat in stats_after["sections"]:
            print(
                f"  {section_stat.name}: "
                f"{section_stat.token_count}/{section_stat.max_tokens} "
                f"({section_stat.entry_count} entries)"
            )

    # ==================== PER-SECTION OPERATIONS ====================

    print("\n--- Per-Section Operations ---")

    # Get entries from specific section
    conversation_entries = await ctx.working_memory.list_all(section="conversation")
    print(f"\nConversation section has {len(conversation_entries)} entries")

    # Clear a specific section
    await ctx.working_memory.clear(section="tools")
    print("Cleared tools section")

    # Add a new section dynamically
    ctx.working_memory.add_section(
        name="working_notes",
        max_tokens=500,
        priority=0.3,
    )
    print("Added new 'working_notes' section")

    await ctx.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.MESSAGE,
            content="NOTE: Alice prefers Python over Java for new projects.",
        ),
        section="working_notes",
    )

    final_stats = ctx.working_memory.get_stats()
    print(f"\nWorking notes: {final_stats['sectioned_mode'] and final_stats['sections'][-1].entry_count} entries")

    await ctx.close()
    print("\nExample complete!")


if __name__ == "__main__":
    asyncio.run(main())
