"""Anthropic Claude integration with Contexo memory.

This example demonstrates using Contexo with Anthropic's Claude API
for AI conversations with persistent memory.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def chat_with_claude():
    """Interactive chat using Claude API with Contexo memory."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        print("Anthropic SDK not installed. Install with:")
        print("  pip install anthropic")
        return

    from contexo import Contexo, local_config

    print("=== Anthropic Claude + Contexo Memory ===\n")

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    # Initialize Contexo
    config = local_config(
        db_path="./claude_context.db",
        max_tokens=200000,  # Claude has large context window
    )

    contexo = Contexo(config=config)
    await contexo.initialize()

    # Initialize Anthropic client
    client = AsyncAnthropic(api_key=api_key)

    # Set conversation ID
    import uuid

    conversation_id = os.getenv("CONTEXO_CONVERSATION_ID", str(uuid.uuid4()))
    contexo.set_conversation_id(conversation_id)

    print(f"Conversation ID: {conversation_id}")
    print("Type 'quit' to exit\n")

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                break

            # Save user message to Contexo
            await contexo.add_message(
                content=user_input,
                role="user",
                importance=0.8,
            )

            # Get conversation context from Contexo
            context = await contexo.get_context()

            # Build messages for Claude
            messages = []

            # Add context as system message
            if context:
                messages.append(
                    {
                        "role": "user",
                        "content": f"<previous_conversation>\n{context}\n</previous_conversation>",
                    }
                )

            # Add current message
            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            # Call Claude
            print("Claude: ", end="", flush=True)

            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system="You are a helpful assistant with access to conversation history. "
                "Reference past context when relevant to provide better responses.",
                messages=messages,
                stream=True,
            )

            # Stream response
            assistant_message = ""
            async for chunk in response:
                if chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        content = chunk.delta.text
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
    print("\nSession saved!")


async def simple_claude_example():
    """Simple example of Claude integration."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        print("Skipping: anthropic not installed")
        return

    from contexo import Contexo, minimal_config

    print("\n=== Simple Claude Example ===\n")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Skipping: ANTHROPIC_API_KEY not set")
        return

    contexo = Contexo(config=minimal_config())
    await contexo.initialize()

    client = AsyncAnthropic(api_key=api_key)

    # Build conversation history
    messages = [
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "I'll remember that blue is your favorite color!"},
        {"role": "user", "content": "What's my favorite color?"},
    ]

    # Save to Contexo
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        await contexo.add_message(msg["content"], role=role)

    # Get context
    context = await contexo.get_context()
    print(f"Context: {context}")

    # Build messages with context
    claude_messages = [
        {
            "role": "user",
            "content": f"<conversation_history>\n{context}\n</conversation_history>\n\n"
            f"Based on the conversation above, answer: What's my favorite color?",
        }
    ]

    # Call Claude
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=claude_messages,
    )

    print(f"\nClaude: {response.content[0].text}")

    await contexo.close()


async def main():
    """Run the Claude integration."""
    if os.getenv("ANTHROPIC_API_KEY"):
        await chat_with_claude()
    else:
        await simple_claude_example()
        print("\nTo run interactive chat:")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  python examples/anthropic_integration.py")


if __name__ == "__main__":
    asyncio.run(main())
