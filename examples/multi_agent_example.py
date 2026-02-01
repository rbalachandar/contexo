"""Multi-agent context sharing example.

This example demonstrates how Contexo can be used in a multi-agent system
where multiple agents need to share context while maintaining private thoughts.
"""

import asyncio
from contexo import Contexo
from contexo.config.defaults import minimal_config
from contexo.config.settings import ContexoConfig, StorageConfig, EmbeddingConfig


def multi_agent_config() -> ContexoConfig:
    """Create a config for multi-agent mode."""
    from dataclasses import replace

    # Start with minimal config (in-memory, mock embeddings)
    config = minimal_config()

    # Replace with SQLite storage for persistence and multi-agent mode
    return replace(
        config,
        storage=StorageConfig(
            backend_type="sqlite",
            db_path="./multi_agent.db",
        ),
        embeddings=EmbeddingConfig(
            provider_type="mock",  # Use mock embeddings for demo
        ),
        multi_agent=True,
    )


async def main() -> None:
    # Initialize with multi-agent mode enabled
    ctx = Contexo(config=multi_agent_config())
    await ctx.initialize()

    # Set conversation ID
    ctx.set_conversation_id("multi-agent-demo")

    print("=== Multi-Agent Context Sharing Demo ===\n")

    # User asks a complex question
    await ctx.add_message(
        "user",
        "I need to build a web scraper. What's the best approach?",
    )
    print("[User] I need to build a web scraper. What's the best approach?\n")

    # Agent 1: Researcher - Has private thoughts, then contributes
    agent1_id = "researcher"

    await ctx.add_thought(
        "User probably needs Python. Should consider scrapy, beautifulsoup, playwright.",
        agent_id=agent1_id,
    )
    print(f"[{agent1_id}] (private thought) User probably needs Python...")

    await ctx.add_contribution(
        "For web scraping, the main options are: Scrapy (large-scale), "
        "BeautifulSoup (simple parsing), and Playwright/Selenium (dynamic content). "
        "The choice depends on the use case.",
        agent_id=agent1_id,
    )
    print(f"[{agent1_id}] (shared) For web scraping, the main options are...\n")

    # Agent 2: Architect - Adds analysis
    agent2_id = "architect"

    await ctx.add_thought(
        "Researcher gave good options. I should recommend based on scale.",
        agent_id=agent2_id,
    )
    print(f"[{agent2_id}] (private thought) Researcher gave good options...")

    await ctx.add_contribution(
        "Based on the requirements: Use BeautifulSoup for simple static sites, "
        "Playwright for dynamic/JS-heavy sites, and Scrapy for enterprise-scale crawling.",
        agent_id=agent2_id,
    )
    print(f"[{agent2_id}] (shared) Based on the requirements...\n")

    # Agent 3: Coordinator - Makes a decision
    agent3_id = "coordinator"

    await ctx.add_thought(
        "I have good input from both agents. Time to make a recommendation.",
        agent_id=agent3_id,
    )
    print(f"[{agent3_id}] (private thought) I have good input from both agents...")

    await ctx.add_decision(
        "DECISION: Start with BeautifulSoup for simplicity. "
        "If the site uses JavaScript, switch to Playwright. "
        "Only consider Scrapy if crawling thousands of pages.",
        agent_id=agent3_id,
    )
    print(f"[{agent3_id}] (decision) DECISION: Start with BeautifulSoup...\n")

    print("=" * 60)
    print("Context from different perspectives:")
    print("=" * 60)

    # Show what each agent sees
    for agent in ["researcher", "architect", "coordinator"]:
        print(f"\n--- {agent.title()}'s View ---")
        context = await ctx.get_agent_context(agent_id=agent, scope="all")
        print(context)

    print("\n" + "=" * 60)
    print("Shared context only (what all agents can see):")
    print("=" * 60)
    shared_context = await ctx.get_agent_context(agent_id="researcher", scope="shared")
    print(shared_context)

    print("\n" + "=" * 60)
    print("Private context only (researcher's thoughts):")
    print("=" * 60)
    private_context = await ctx.get_context(agent_id="researcher", scope="private")
    print(private_context)

    await ctx.close()


if __name__ == "__main__":
    asyncio.run(main())
