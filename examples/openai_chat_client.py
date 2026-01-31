"""OpenAI-compatible Chat Client with Contexo memory.

This example shows how to integrate Contexo with OpenAI's chat completion API
for a production-ready chatbot with persistent memory.

Supports:
- OpenAI (default)
- GLM-4 (智谱AI) via OPENAI_BASE_URL
- Any OpenAI-compatible API
"""

import asyncio
import json
import os
import sys
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def add_system_prompt(contexo: Any) -> None:
    """Add system prompt to the system section if not already present."""
    from contexo.core.memory import MemoryEntry, EntryType

    # Check if system section already has entries
    stats = contexo.working_memory.get_stats()
    if stats.get("sectioned_mode"):
        for section_stat in stats.get("sections", []):
            if section_stat.name == "system" and section_stat.entry_count > 0:
                return  # System prompt already exists

    # Add system prompt
    await contexo.working_memory.add(
        MemoryEntry(
            entry_type=EntryType.SYSTEM,
            content="You are a helpful assistant with access to conversation history. Be conversational and reference past context when relevant.",
        ),
        section="system",
    )


async def chat_with_memory():
    """Interactive chat using OpenAI-compatible API with Contexo memory."""
    from openai import AsyncOpenAI

    from contexo import Contexo, local_config

    # Detect provider from environment
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("MODEL_NAME")

    provider_name = "OpenAI"
    if base_url:
        if "bigmodel" in base_url or "z.ai" in base_url:
            provider_name = "GLM-4.7 (智谱AI)"
        elif "openrouter" in base_url:
            provider_name = "OpenRouter"
        else:
            provider_name = "OpenAI-compatible API"

    print(f"=== {provider_name} Chat Client with Contexo Memory ===\n")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Initialize Contexo with local storage and sectioned memory
    import dataclasses
    from contexo import local_config
    from contexo.config.settings import WorkingMemoryConfig

    config = local_config(
        db_path="./chat_context.db",
        max_tokens=16384,
    )
    # Add sectioned memory configuration
    config = dataclasses.replace(
        config,
        working_memory=WorkingMemoryConfig(
            max_tokens=16384,
            strategy="sliding_window",
            sections={
                "system": {"max_tokens": 500, "pinned": True},
                "conversation": {"max_tokens": 12000},
                "rag_context": {"max_tokens": 2000},
                "tools": {"max_tokens": 1000, "priority": 0.3},
            },
        ),
    )

    contexo = Contexo(config=config)
    await contexo.initialize()

    # Initialize OpenAI-compatible client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = AsyncOpenAI(**client_kwargs)

    # Set default model based on provider
    default_model = "gpt-4o-mini"
    if base_url and ("bigmodel" in base_url or "z.ai" in base_url):
        default_model = "glm-4.7"
    elif base_url and "openrouter" in base_url:
        default_model = "deepseek/deepseek-r1-0528:free"

    model = default_model

    # Get or create conversation ID
    import uuid

    # Check if conversation ID is provided via environment
    env_conversation_id = os.getenv("CONTEXO_CONVERSATION_ID")

    if env_conversation_id:
        # Use provided conversation ID and load history
        conversation_id = env_conversation_id
        loaded = await contexo.continue_conversation(conversation_id, max_messages=20)
        print(f"Conversation ID: {conversation_id} (from environment)")
        if loaded > 0:
            print(f"✓ Loaded {loaded} messages from history")
        await add_system_prompt(contexo)
    else:
        # List existing conversations and let user choose
        collections = await contexo._persistent._storage.list_collections()

        if collections:
            print("\n=== Existing Conversations ===")

            # Count messages in each conversation
            conv_with_counts = []
            for conv_id in collections:
                entries = await contexo._persistent._storage.list_entries(collection=conv_id)
                conv_with_counts.append((conv_id, len(entries)))

            # Sort by message count (most recent first)
            conv_with_counts.sort(key=lambda x: x[1], reverse=True)

            for i, (conv_id, count) in enumerate(conv_with_counts, 1):
                # Show first 8 chars of ID
                short_id = conv_id[:8] if len(conv_id) >= 8 else conv_id
                print(f"{i}. {short_id}... ({count} messages)")

            print(f"{len(conv_with_counts) + 1}. Create new conversation")
            print()

            choice = input(f"Select conversation (1-{len(conv_with_counts) + 1}): ").strip()

            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(conv_with_counts):
                    # Continue existing conversation (auto-loads history)
                    conversation_id = conv_with_counts[choice_num - 1][0]
                    loaded = await contexo.continue_conversation(conversation_id, max_messages=20)
                    print(f"\nConversation ID: {conversation_id}")
                    print(f"✓ Loaded {loaded} messages from history")
                    await add_system_prompt(contexo)
                elif choice_num == len(conv_with_counts) + 1:
                    # Create new conversation (no history loaded)
                    conversation_id = str(uuid.uuid4())
                    contexo.set_conversation_id(conversation_id)
                    await add_system_prompt(contexo)
                    print(f"\nConversation ID: {conversation_id}")
                    print("✓ Started new conversation")
                else:
                    print("Invalid choice, creating new conversation")
                    conversation_id = str(uuid.uuid4())
                    contexo.set_conversation_id(conversation_id)
                    await add_system_prompt(contexo)
                    print(f"\nConversation ID: {conversation_id}")
            else:
                print("Invalid input, creating new conversation")
                conversation_id = str(uuid.uuid4())
                contexo.set_conversation_id(conversation_id)
                print(f"\nConversation ID: {conversation_id}")
        else:
            # No existing conversations
            conversation_id = str(uuid.uuid4())
            contexo.set_conversation_id(conversation_id)
            await add_system_prompt(contexo)
            print(f"Conversation ID: {conversation_id}")
            print("✓ Started new conversation")

    print("Commands:")
    print("  'quit' or 'exit' - Exit the chat")
    print("  'clear' - Clear working memory")
    print("  'working' or 'wm' - Show context usage + formatted content sent to LLM")
    print("  'sections' - Show working memory sections breakdown")
    print("  'section <name>' - Show entries in a specific section")
    print("  'all' - List all stored messages (persistent)")
    print("  'search <query>' - Search conversation history")
    print("  'trace <query>' - Trace context for a message")
    print()

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if user_input.lower() == "clear":
                await contexo.clear_working_memory()
                print("Memory cleared!")
                continue

            if user_input.lower() == "stats":
                stats = contexo.get_stats()
                print("\n--- Statistics ---")
                print(f"Working memory entries: {stats['working_memory']['total_entries']}")
                print(
                    f"Token usage: {stats['working_memory']['total_tokens']}/{stats['working_memory']['max_tokens']}"
                )
                print(f"Utilization: {stats['working_memory']['utilization']:.1%}")
                continue

            if user_input.lower() in ("working", "wm"):
                # Combine stats and context
                stats = contexo.working_memory.get_stats()
                print(f"\n--- Context Usage ---")
                print(f"Total: {stats['total_tokens']}/{stats['max_tokens']} tokens ({stats['utilization']:.1%})")
                print(f"Entries: {stats['total_entries']}")
                print(f"Strategy: {stats['strategy']}")
                print(f"Remaining: {stats['remaining_tokens']} tokens")
                print()
                print("--- Formatted Context (sent to LLM) ---")
                context = await contexo.get_context()
                print(context)
                print()
                continue

            if user_input.lower() == "all":
                # Get all entries from persistent storage
                all_entries = await contexo._persistent._storage.list_entries(
                    collection=conversation_id, entry_type=None
                )
                print(f"\n--- All Stored Messages ({len(all_entries)} total) ---")
                for i, entry in enumerate(all_entries, 1):
                    role = entry.metadata.get("role", entry.entry_type.value)
                    content = entry.content[:80] + "..." if len(entry.content) > 80 else entry.content
                    print(f"{i}. [{role}] {content}")
                print()
                continue

            if user_input.lower() in ("sections", "context"):
                # Show working memory contents by section
                stats = contexo.working_memory.get_stats()
                total_tokens = stats["total_tokens"]
                max_tokens = stats["max_tokens"]
                usage_pct = (total_tokens / max_tokens) * 100

                print(f"\n--- Working Memory Context Usage ---")
                print(f"Total: {total_tokens}/{max_tokens} tokens ({usage_pct:.1f}%)")
                print(f"Entries: {stats['total_entries']}\n")

                if stats.get("sectioned_mode"):
                    # Show section breakdown
                    print("By section:")
                    for section_stat in stats.get("sections", []):
                        section_tokens = section_stat.token_count
                        section_max = section_stat.max_tokens
                        section_pct = (section_tokens / max_tokens) * 100
                        section_util = (section_tokens / section_max) * 100 if section_max > 0 else 0

                        print(f"  [{section_stat.name}]")
                        print(f"    {section_tokens}/{section_max} tokens ({section_util:.1f}% of section)")
                        print(f"    {section_pct:.1f}% of total context")
                        print(f"    Pinned: {section_stat.pinned}, Priority: {section_stat.priority}")
                        print(f"    Entries: {section_stat.entry_count}")
                        print()
                else:
                    # Not sectioned - show summary
                    print(f"Utilization: {stats['utilization']:.1%}")
                    print(f"Strategy: {stats.get('strategy', 'N/A')}")
                    print()
                continue

            if user_input.lower().startswith("section "):
                # Show entries in a specific section
                section_name = user_input[8:].strip()
                stats = contexo.working_memory.get_stats()

                if not stats.get("sectioned_mode"):
                    print("Sectioned mode is not enabled")
                    continue

                section_entries = await contexo.working_memory.list_all(section=section_name)
                print(f"\n--- Section: {section_name} ({len(section_entries)} entries) ---")

                if section_entries:
                    for i, entry in enumerate(section_entries, 1):
                        role = entry.metadata.get("role", entry.entry_type.value)
                        content = entry.content[:80] + "..." if len(entry.content) > 80 else entry.content
                        tokens = entry.token_count
                        print(f"{i}. [{role}] ({tokens}t) {content}")
                else:
                    print(f"Section '{section_name}' is empty or doesn't exist")
                print()
                continue

            if user_input.lower().startswith("search "):
                query = user_input[7:].strip()
                results = await contexo.search_memory(query, limit=5)
                print(f"\n--- Search Results for '{query}' ---")
                if results:
                    for i, result in enumerate(results, 1):
                        role = result.entry.metadata.get("role", result.entry.entry_type.value)
                        content = result.entry.content[:80] + "..." if len(result.entry.content) > 80 else result.entry.content
                        score_pct = result.score * 100
                        print(f"{i}. [{role}] (score: {score_pct:.1f}%) {content}")
                else:
                    print("No results found.")
                print()
                continue

            if user_input.lower().startswith("trace "):
                query = user_input[6:].strip()
                context_data = await contexo.get_message_context(
                    query=query,
                    include_conversation=True,
                    conversation_window=3,
                )
                print(f"\n--- Context Trace for '{query}' ---")
                if context_data["entry"]:
                    entry = context_data["entry"]
                    print(f"\nTarget Message: [{entry.entry_type.value}] {entry.content}")
                else:
                    print("\nNo message found matching that query.")
                    print()
                    continue

                if context_data["parent"]:
                    print(f"\nParent: {context_data['parent'].content[:60]}...")

                if context_data["children"]:
                    print(f"\nChildren: {len(context_data['children'])}")
                    for child in context_data["children"]:
                        print(f"  - {child.entry_type.value}: {child.content[:60]}...")

                if context_data["conversation_context"]:
                    print(f"\nConversation Context: {len(context_data['conversation_context'])} messages")
                print()
                continue

            # Save user message to Contexo
            await contexo.add_message(
                content=user_input,
                role="user",
                importance=0.8,
            )

            # Get conversation context
            context = await contexo.get_context()

            # Build messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to conversation history and memory tools. "
                    "Use tools when you need to search past context or save important information.",
                },
            ]

            # Add context messages
            for line in context.split("\n"):
                if line.startswith("user: "):
                    messages.append({"role": "user", "content": line[6:]})
                elif line.startswith("assistant: "):
                    messages.append({"role": "assistant", "content": line[11:]})

            # Add current user message
            messages.append({"role": "user", "content": user_input})

            # Define tools for the LLM
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_memory",
                        "description": "Search past conversation history for relevant context. "
                        "Use this when you need to recall what was discussed earlier.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query to find relevant past messages"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "save_note",
                        "description": "Save an important fact or information to memory. "
                        "Use this to remember user preferences, important details, or things to recall later.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The information to save to memory"
                                }
                            },
                            "required": ["content"]
                        }
                    }
                },
            ]

            # Call LLM (with tools, no streaming for tool handling)
            print("Assistant: ", end="", flush=True)

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=0.7,
                max_tokens=500,
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # Handle tool calls
            if tool_calls:
                # Process each tool call
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name == "search_memory":
                        results = await contexo.search_memory(
                            query=function_args["query"],
                            limit=5
                        )
                        tool_result = "\n".join([
                            f"- {r.entry.content[:100]}..."
                            for r in results
                        ]) if results else "No results found"

                    elif function_name == "save_note":
                        await contexo.add_message(
                            content=function_args["content"],
                            role="system",
                            metadata={"type": "note", "section": "user_profile"}
                        )
                        tool_result = f"Saved note: {function_args['content'][:50]}..."

                    # Add tool response to messages
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [{"id": tool_call.id, "type": tool_call.type, "function": {
                            "name": function_name,
                            "arguments": tool_call.function.arguments
                        }}]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # Get final response from LLM after tool execution
                final_response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500,
                )
                assistant_message = final_response.choices[0].message.content or ""
                print(assistant_message)

            else:
                # No tool calls, just use the response directly
                assistant_message = response_message.content or ""
                print(assistant_message)

            print()  # New line

            # Save assistant response to Contexo
            await contexo.add_message(
                content=assistant_message,
                role="assistant",
                importance=0.9,
            )

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

    await contexo.close()
    print("\nSession saved. Come back anytime!")


async def simple_chat_example():
    """Simple one-shot chat example."""
    from openai import AsyncOpenAI

    from contexo import Contexo, minimal_config

    print("\n=== Simple Chat Example ===\n")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping: OPENAI_API_KEY not set")
        return

    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    if base_url and ("bigmodel" in base_url or "z.ai" in base_url):
        model = model or "glm-4.7"

    contexo = Contexo(config=minimal_config())
    await contexo.initialize()

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)

    # Simulate a conversation
    conversations = [
        ("Hi, I'm Bob!", "Hi Bob! How can I help you today?"),
        ("I love hiking.", "That's great! Hiking is a wonderful way to stay active."),
    ]

    for user_msg, asst_msg in conversations:
        await contexo.add_message(user_msg, role="user")
        await contexo.add_message(asst_msg, role="assistant")

    # Now ask about what was discussed
    print("User: What's my name?")
    await contexo.add_message("What's my name?", role="user")

    # Get context from Contexo
    context = await contexo.get_context()
    print(f"\nContext sent to LLM:\n{context}\n")

    # Build messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": context},
        {"role": "user", "content": "Based on the conversation above, what's my name?"},
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100,
    )

    print(f"Assistant: {response.choices[0].message.content}")

    await contexo.close()


async def main():
    """Run the examples."""
    # Check if OPENAI_API_KEY is set
    if os.getenv("OPENAI_API_KEY"):
        # Run the interactive chat
        await chat_with_memory()
    else:
        print("OpenAI API key not found. Running with mock responses...")
        await mock_chat_demo()


async def mock_chat_demo():
    """Demo without actual API calls."""
    from contexo import Contexo, minimal_config

    print("=== Mock Chat Demo (No API Key) ===\n")

    contexo = Contexo(config=minimal_config())
    await contexo.initialize()

    # Simulate conversation
    await contexo.add_message("Hi, I'm Charlie!", role="user")
    await contexo.add_message("Hi Charlie! Nice to meet you.", role="assistant")
    await contexo.add_message("I'm a data scientist.", role="user")
    await contexo.add_message("Interesting! What kind of data do you work with?", role="assistant")

    # Show context
    context = await contexo.get_context()
    print("Context that would be sent to LLM:")
    print(context)

    # Show we can search memory
    results = await contexo.search_memory("Charlie", limit=5)
    print(f"\nFound {len(results)} messages about Charlie")

    # Show provenance
    ctx = await contexo.get_message_context(query="data scientist")
    print("\nProvenance for 'data scientist':")
    print(f"  Message: {ctx['entry'].content if ctx['entry'] else 'Not found'}")
    print(f"  Related: {len(ctx.get('related', []))} messages")

    await contexo.close()

    print("\nTo run with real LLM API:")
    print("\nOpenAI:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  python examples/openai_chat_client.py")
    print("\nGLM-4.7 (智谱AI):")
    print("  export OPENAI_API_KEY='your-glm4-key'")
    print("  export OPENAI_BASE_URL='https://api.z.ai/api/paas/v4/'")
    print("  python examples/openai_chat_client.py")
    print("\nOpenRouter (Free DeepSeek R1):")
    print("  export OPENAI_API_KEY='your-openrouter-key'")
    print("  export OPENAI_BASE_URL='https://openrouter.ai/api/v1'")
    print("  python examples/openai_chat_client.py")
    print("\nOther OpenAI-compatible APIs:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  export OPENAI_BASE_URL='your-api-endpoint'")
    print("  export MODEL_NAME='your-model-name'")
    print("  python examples/openai_chat_client.py")


if __name__ == "__main__":
    asyncio.run(main())
