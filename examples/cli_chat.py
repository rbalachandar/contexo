"""Simple CLI Chat Client with Contexo memory.

This is a self-contained chat client that demonstrates Contexo's
memory management without requiring any external API keys.
"""

import asyncio
import sys
import os
import uuid
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class SimpleChatClient:
    """A simple chat client using Contexo for memory management."""

    def __init__(self):
        """Initialize the chat client."""
        from contexo import Contexo, minimal_config

        # Try to use local config with SQLite, fall back to minimal if not available
        try:
            from contexo import local_config
            self.config = local_config(db_path="./chat_memory.db")
        except ImportError:
            self.config = minimal_config()

        self.contexo = Contexo(config=self.config)
        self.conversation_id = str(uuid.uuid4())

        # Predefined responses for demo
        self.responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What would you like to talk about?",
            "how are you": "I'm doing well, thank you for asking! How about you?",
            "my name is": "Nice to meet you! I'll remember that.",
            "i'm": "Thanks for sharing! Tell me more.",
            "i am": "Interesting! Please go on.",
            "weather": "I don't have real-time weather access, but I hope it's nice where you are!",
            "help": "I'm a demo chatbot powered by Contexo for memory management. Ask me anything!",
        }

    async def start(self):
        """Start the chat session."""
        await self.contexo.initialize()
        self.contexo.set_conversation_id(self.conversation_id)

        print("╔════════════════════════════════════════════════════════════╗")
        print("║           Contexo Chat - Memory Management Demo          ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"\nConversation ID: {self.conversation_id}")
        print("\nCommands:")
        print("  'quit' or 'exit' - Exit the chat")
        print("  'clear' - Clear working memory")
        print("  'search <query>' - Search conversation history")
        print("  'trace <query>' - Trace context for a message")
        print("  'stats' - Show memory statistics")
        print("  'export' - Export conversation to JSON")
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ("quit", "exit", "q"):
                    await self.quit()
                    break

                if user_input.lower() == "clear":
                    await self.contexo.clear_working_memory()
                    print("✓ Memory cleared")
                    continue

                if user_input.lower() == "stats":
                    self.show_stats()
                    continue

                if user_input.lower().startswith("search "):
                    query = user_input[7:].strip()
                    await self.search_memory(query)
                    continue

                if user_input.lower().startswith("trace "):
                    query = user_input[6:].strip()
                    await self.trace_context(query)
                    continue

                if user_input.lower() == "export":
                    await self.export_conversation()
                    continue

                # Normal conversation flow
                await self.process_message(user_input)

            except KeyboardInterrupt:
                print("\n")
                await self.quit()
                break
            except Exception as e:
                print(f"Error: {e}")

    async def process_message(self, user_input: str):
        """Process a user message and generate a response."""
        import random

        # Save user message
        await self.contexo.add_message(
            content=user_input,
            role="user",
            importance=0.8,
        )

        # Get conversation context
        context = await self.contexo.get_context()

        # Generate a contextual response
        response = self.generate_response(user_input, context)

        # Save assistant response
        await self.contexo.add_message(
            content=response,
            role="assistant",
            importance=0.9,
        )

        print(f"Bot: {response}")

    def generate_response(self, user_input: str, context: str) -> str:
        """Generate a response based on input and context."""
        user_input_lower = user_input.lower()

        # Check for predefined responses
        for key, response in self.responses.items():
            if key in user_input_lower:
                return response

        # Check context for relevant information
        context_lower = context.lower()

        # Name detection
        if "my name is" in user_input_lower:
            name = user_input_lower.replace("my name is", "").strip()
            name = name.replace("i'm", "").strip()
            name = name.replace("i am", "").strip()
            if name:
                return f"Nice to meet you, {name.capitalize()}! I'll remember your name."

        # Check if user mentioned something before
        if "remember" in user_input_lower or "did i say" in user_input_lower or "do you know" in user_input_lower:
            # Search memory for relevant context
            words = user_input_lower.split()
            for word in words:
                if len(word) > 3 and word not in ("remember", "about", "what", "does", "know"):
                    if word in context_lower:
                        return f"Yes, I remember you mentioned '{word}' earlier!"

        # Contextual responses
        if "python" in user_input_lower or "programming" in user_input_lower:
            return "Programming is fascinating! Are you working on any projects?"
        if "weather" in context_lower and "weather" in user_input_lower:
            return "We've discussed weather - it's always a popular topic!"
        if len(context) > 100:
            return f"I remember our conversation. It's great to chat with you! (Context: {len(context.split(chr(10)))} messages)"

        # Default responses
        defaults = [
            "That's interesting! Tell me more.",
            "I see. What else would you like to discuss?",
            "Could you elaborate on that?",
            "I'm here to help! What else is on your mind?",
            "That's a good point. How does that relate to what we were discussing?",
        ]
        return random.choice(defaults)

    async def search_memory(self, query: str):
        """Search conversation memory."""
        if not query:
            print("Please provide a search query.")
            return

        results = await self.contexo.search_memory(query, limit=5)

        print(f"\n─── Search Results for '{query}' ───")
        if results:
            for i, entry in enumerate(results, 1):
                role = entry.metadata.get("role", entry.entry_type.value)
                content = entry.content[:80] + "..." if len(entry.content) > 80 else entry.content
                print(f"{i}. [{role}] {content}")
        else:
            print("No results found.")
        print()

    async def trace_context(self, query: str):
        """Trace the context for a message."""
        if not query:
            print("Please provide a query to trace.")
            return

        context_data = await self.contexo.get_message_context(
            query=query,
            include_conversation=True,
            conversation_window=3,
        )

        print(f"\n─── Context Trace for '{query}' ───")

        if context_data["entry"]:
            entry = context_data["entry"]
            print(f"\nTarget Message:")
            print(f"  [{entry.entry_type.value}] {entry.content}")
        else:
            print("\nNo message found matching that query.")
            return

        if context_data["parent"]:
            print(f"\nParent:")
            print(f"  {context_data['parent'].content[:60]}...")

        if context_data["children"]:
            print(f"\nChildren: {len(context_data['children'])}")
            for child in context_data["children"]:
                print(f"  - {child.entry_type.value}: {child.content[:60]}...")

        if context_data["conversation_context"]:
            print(f"\nConversation Context: {len(context_data['conversation_context'])} messages")

        print()

    def show_stats(self):
        """Display memory statistics."""
        stats = self.contexo.get_stats()

        print("\n─── Memory Statistics ───")
        print(f"Working Memory:")
        print(f"  Entries: {stats['working_memory']['total_entries']}")
        print(f"  Tokens: {stats['working_memory']['total_tokens']}/{stats['working_memory']['max_tokens']}")
        print(f"  Utilization: {stats['working_memory']['utilization']:.1%}")
        print(f"  Strategy: {stats['working_memory']['strategy']}")

        print(f"\nPersistent Memory:")
        print(f"  Type: {stats['persistent_memory']['storage_type']}")
        print(f"  Embedding: {stats['persistent_memory']['embedding_provider']}")

        print(f"\nConversation:")
        print(f"  ID: {self.conversation_id}")
        print()

    async def export_conversation(self):
        """Export conversation to JSON."""
        from contexo.utils.serialization import export_context

        entries = await self.contexo.working_memory.list_all()

        if not entries:
            print("No conversation to export.")
            return

        # Export as JSON
        json_export = export_context(entries, format="json")

        filename = f"conversation_{self.conversation_id[:8]}.json"
        with open(filename, "w") as f:
            f.write(json_export)

        print(f"✓ Exported {len(entries)} messages to {filename}")

    async def quit(self):
        """Handle quit sequence."""
        stats = self.contexo.get_stats()
        total_messages = stats['working_memory']['total_entries']

        await self.contexo.close()

        print("\n╔════════════════════════════════════════════════════════════╗")
        print("║                   Session Summary                          ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"  Messages exchanged: {total_messages}")
        print(f"  Conversation ID: {self.conversation_id}")
        print(f"  Memory database: ./chat_memory.db")
        print("\n  Your conversation has been saved!")
        print("  Come back anytime - I'll remember what we talked about.")
        print()


async def main():
    """Main entry point."""
    client = SimpleChatClient()
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
