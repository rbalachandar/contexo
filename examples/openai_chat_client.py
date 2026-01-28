"""OpenAI Chat Client with Contexo memory.

This example shows how to integrate Contexo with OpenAI's chat completion API
for a production-ready chatbot with persistent memory.
"""

import asyncio
import sys
import os
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def chat_with_memory():
    """Interactive chat using OpenAI API with Contexo memory."""
    from openai import AsyncOpenAI
    from contexo import Contexo, local_config

    print("=== OpenAI Chat Client with Contexo Memory ===\n")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Initialize Contexo with local storage
    config = local_config(
        db_path="./chat_context.db",
        max_tokens=16384,  # Larger context for chat
    )

    contexo = Contexo(config=config)
    await contexo.initialize()

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=api_key)

    # Get or create conversation ID
    import uuid
    conversation_id = os.getenv("CONTEXO_CONVERSATION_ID", str(uuid.uuid4()))
    contexo.set_conversation_id(conversation_id)

    print(f"Conversation ID: {conversation_id}")
    print("Type 'quit' to exit, 'clear' to clear memory, 'stats' for statistics\n")

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
                print(f"\n--- Statistics ---")
                print(f"Working memory entries: {stats['working_memory']['total_entries']}")
                print(f"Token usage: {stats['working_memory']['total_tokens']}/{stats['working_memory']['max_tokens']}")
                print(f"Utilization: {stats['working_memory']['utilization']:.1%}")
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
                    "content": "You are a helpful assistant with access to conversation history. "
                    "Be conversational and reference past context when relevant.",
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

            # Call OpenAI
            print("Assistant: ", end="", flush=True)

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=True,
            )

            # Stream and collect the response
            assistant_message = ""
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    assistant_message += content

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

    contexo = Contexo(config=minimal_config())
    await contexo.initialize()

    client = AsyncOpenAI(api_key=api_key)

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
        model="gpt-4o-mini",
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
    print(f"\nProvenance for 'data scientist':")
    print(f"  Message: {ctx['entry'].content if ctx['entry'] else 'Not found'}")
    print(f"  Related: {len(ctx.get('related', []))} messages")

    await contexo.close()

    print("\nTo run with real OpenAI API:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  python examples/openai_chat_client.py")


if __name__ == "__main__":
    asyncio.run(main())
