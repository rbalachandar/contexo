"""Generic LLM wrapper with Contexo memory.

This example shows how to create a reusable LLM wrapper class that
integrates Contexo memory management with any LLM provider.
"""

import asyncio
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from contexo import Contexo, ContexoConfig


@dataclass
class ChatMessage:
    """A chat message."""

    role: str
    content: str
    metadata: dict[str, Any] | None = None


@dataclass
class ChatResponse:
    """A chat response from the LLM."""

    content: str
    metadata: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a response from the LLM.

        Args:
            messages: List of chat messages
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatResponse with the generated content
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response.

        Args:
            messages: List of chat messages
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of the response as they're generated
        """
        ...


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._client = None

    async def _get_client(self):
        from openai import AsyncOpenAI

        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> ChatResponse:
        client = await self._get_client()

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = await client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return ChatResponse(
            content=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        client = await self._get_client()

        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = await client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self._client = None

    async def _get_client(self):
        from anthropic import AsyncAnthropic

        if self._client is None:
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ChatResponse:
        client = await self._get_client()

        # Anthropic format
        anthropic_messages = []
        system_msg = None

        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        response = await client.messages.create(
            model=self.model,
            system=system_msg,
            messages=anthropic_messages,
            max_tokens=max_tokens,
            **kwargs,
        )

        return ChatResponse(
            content=response.content[0].text,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        client = await self._get_client()

        anthropic_messages = []
        system_msg = None

        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        response = await client.messages.create(
            model=self.model,
            system=system_msg,
            messages=anthropic_messages,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        async for chunk in response:
            if chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    yield chunk.delta.text


class MockProvider(LLMProvider):
    """Mock provider for testing without API calls."""

    def __init__(self):
        self.responses = [
            "That's an interesting point! Could you tell me more?",
            "I understand. What else would you like to discuss?",
            "Let me think about that... Based on our conversation, I'd say there are several perspectives to consider.",
            "That reminds me of what you mentioned earlier. The connection is quite fascinating!",
            "I'm here to help! Feel free to ask me anything.",
        ]

    async def generate(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        import random

        return ChatResponse(
            content=random.choice(self.responses),
            usage={"total_tokens": sum(len(m.content) for m in messages) // 4},
        )

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        import random

        response = random.choice(self.responses)
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.05)


class ContexoChat:
    """Generic chat client with Contexo memory and pluggable LLM."""

    def __init__(
        self,
        llm: LLMProvider,
        contexo: Contexo | None = None,
        config: ContexoConfig | None = None,
        system_prompt: str | None = None,
    ):
        """Initialize the chat client.

        Args:
            llm: The LLM provider to use
            contexo: Pre-configured Contexo instance
            config: Configuration for creating Contexo
            system_prompt: Optional system prompt for the LLM
        """
        self.llm = llm
        self.system_prompt = system_prompt or (
            "You are a helpful assistant with access to conversation history. "
            "Reference past context when relevant to provide better responses."
        )

        if contexo:
            self.contexo = contexo
        elif config:
            self.contexo = Contexo(config=config)
        else:
            # Try to create with local config, fall back to minimal if needed
            from contexo import Contexo as ContexoClass
            from contexo.config import local_config, minimal_config

            try:
                self.contexo = ContexoClass(config=local_config())
            except (ImportError, Exception):
                self.contexo = ContexoClass(config=minimal_config())

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the chat client."""
        if not self._initialized:
            await self.contexo.initialize()
            self._initialized = True

    async def close(self) -> None:
        """Close the chat client."""
        if self._initialized:
            await self.contexo.close()
            self._initialized = False

    async def chat(
        self,
        user_message: str,
        include_context: bool = True,
        **llm_kwargs: Any,
    ) -> ChatResponse:
        """Send a message and get a response.

        Args:
            user_message: The user's message
            include_context: Whether to include conversation history
            **llm_kwargs: Additional arguments for the LLM

        Returns:
            ChatResponse with the LLM's response
        """
        await self.initialize()

        # Save user message
        await self.contexo.add_message(
            content=user_message,
            role="user",
            importance=0.8,
        )

        # Build messages for LLM
        messages = [ChatMessage(role="system", content=self.system_prompt)]

        # Add conversation context
        if include_context:
            context = await self.contexo.get_context()
            if context:
                # Parse context and add as messages
                for line in context.split("\n"):
                    if line.startswith("user: "):
                        messages.append(ChatMessage(role="user", content=line[6:]))
                    elif line.startswith("assistant: "):
                        messages.append(ChatMessage(role="assistant", content=line[11:]))

        # Add current message
        messages.append(ChatMessage(role="user", content=user_message))

        # Generate response
        response = await self.llm.generate(messages, **llm_kwargs)

        # Save assistant response
        await self.contexo.add_message(
            content=response.content,
            role="assistant",
            importance=0.9,
        )

        return response

    async def chat_stream(
        self,
        user_message: str,
        include_context: bool = True,
        **llm_kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a message and stream the response.

        Args:
            user_message: The user's message
            include_context: Whether to include conversation history
            **llm_kwargs: Additional arguments for the LLM

        Yields:
            Chunks of the response as they're generated
        """
        await self.initialize()

        # Save user message
        await self.contexo.add_message(
            content=user_message,
            role="user",
            importance=0.8,
        )

        # Build messages
        messages = [ChatMessage(role="system", content=self.system_prompt)]

        if include_context:
            context = await self.contexo.get_context()
            if context:
                for line in context.split("\n"):
                    if line.startswith("user: "):
                        messages.append(ChatMessage(role="user", content=line[6:]))
                    elif line.startswith("assistant: "):
                        messages.append(ChatMessage(role="assistant", content=line[11:]))

        messages.append(ChatMessage(role="user", content=user_message))

        # Stream response
        full_response = ""
        async for chunk in self.llm.generate_stream(messages, **llm_kwargs):
            yield chunk
            full_response += chunk

        # Save assistant response
        await self.contexo.add_message(
            content=full_response,
            role="assistant",
            importance=0.9,
        )

    async def get_provenance(
        self,
        query: str,
    ) -> dict[str, Any]:
        """Get provenance information for a query.

        Args:
            query: Search query for the message

        Returns:
            Provenance context dictionary
        """
        await self.initialize()
        return await self.contexo.get_message_context(query=query)

    async def search_history(self, query: str, limit: int = 10) -> list[ChatMessage]:
        """Search conversation history.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching messages
        """
        await self.initialize()
        results = await self.contexo.search_memory(query, limit=limit)
        return [
            ChatMessage(
                role=r.metadata.get("role", "assistant"),
                content=r.content,
                metadata=r.metadata,
            )
            for r in results
        ]


# ==================== USAGE EXAMPLES ====================


async def example_openai():
    """Example using OpenAI provider."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping OpenAI example: OPENAI_API_KEY not set")
        return

    llm = OpenAIProvider(api_key=api_key)
    chat = ContexoChat(llm=llm)

    response = await chat.chat("Hello, my name is David!")
    print(f"Response: {response.content}")

    response = await chat.chat("What's my name?")
    print(f"Response: {response.content}")

    await chat.close()


async def example_anthropic():
    """Example using Anthropic provider."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Skipping Anthropic example: ANTHROPIC_API_KEY not set")
        return

    llm = AnthropicProvider(api_key=api_key)
    chat = ContexoChat(llm=llm)

    response = await chat.chat("Hello, my name is Eve!")
    print(f"Response: {response.content}")

    response = await chat.chat("What's my name?")
    print(f"Response: {response.content}")

    await chat.close()


async def example_mock():
    """Example using mock provider (no API key needed)."""
    llm = MockProvider()
    chat = ContexoChat(llm=llm)

    print("=== Mock LLM Example ===\n")

    response1 = await chat.chat("Hi, I'm Frank!")
    print("You: Hi, I'm Frank!")
    print(f"Bot: {response1.content}\n")

    response2 = await chat.chat("What's my name?")
    print("You: What's my name?")
    print(f"Bot: {response2.content}\n")

    # Search history
    history = await chat.search_history("Frank")
    print(f"Found {len(history)} messages about Frank")

    # Get provenance
    provenance = await chat.get_provenance("Frank")
    if provenance["entry"]:
        print(f"Provenance: {provenance['entry'].content}")

    await chat.close()


async def example_streaming():
    """Example of streaming responses."""
    llm = MockProvider()
    chat = ContexoChat(llm=llm)

    print("=== Streaming Example ===\n")

    print("You: Tell me a joke")
    print("Bot: ", end="", flush=True)

    async for chunk in chat.chat_stream("Tell me a joke"):
        print(chunk, end="", flush=True)

    print("\n")

    await chat.close()


async def main():
    """Run all examples."""
    print("=== Contexo Generic LLM Wrapper Examples ===\n")

    await example_mock()
    await example_streaming()
    await example_openai()
    await example_anthropic()

    print("\n=== Summary ===")
    print("The ContexoChat class provides:")
    print("  ✓ Pluggable LLM providers (OpenAI, Anthropic, or custom)")
    print("  ✓ Automatic memory management")
    print("  ✓ Streaming support")
    print("  ✓ Conversation history search")
    print("  ✓ Provenance tracking")


if __name__ == "__main__":
    asyncio.run(main())
