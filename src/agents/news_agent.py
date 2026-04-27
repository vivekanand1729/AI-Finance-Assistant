"""
agents/news_agent.py
Fetches and synthesizes financial news using Tavily search API.
"""
from __future__ import annotations

import json
import os
from datetime import date

from src.agents.base_agent import BaseFinanceAgent
from src.core.state import AgentState


class NewsSynthesizerAgent(BaseFinanceAgent):
    name = "News Synthesizer Agent"
    description = "Fetches and synthesizes latest financial news with educational context"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a financial news analyst helping retail investors understand market-moving "
            "news and its implications. Given recent news articles, you:\n"
            "- Summarize the key facts concisely and objectively\n"
            "- Explain WHY this news matters to investors\n"
            "- Provide context (historical comparisons, broader trends)\n"
            "- Identify potential impacts on different asset classes\n"
            "- Distinguish between short-term noise and long-term signals\n"
            "- Maintain a balanced, non-sensationalist tone\n\n"
            "Always remind users that news reaction can be unpredictable and "
            "not to make hasty investment decisions based on headlines."
        )

    def _run(self, state: AgentState) -> str:
        query = state.get("user_query", "latest financial market news")

        # Try Tavily search
        news_content = self._fetch_news(query)

        prompt = (
            f"## Today's Date: {date.today().isoformat()}\n\n"
            f"## News Content Retrieved\n{news_content}\n\n"
            f"## User Query\n{query}\n\n"
            "Synthesize this news into a clear, educational summary for a retail investor. "
            "Explain the significance and potential market implications."
        )
        return self._chat(prompt)

    def _fetch_news(self, query: str) -> str:
        """Fetch news via Tavily or fallback to a message."""
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_key:
            return self._fallback_news()

        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            results = client.search(
                query=f"financial news {query}",
                search_depth="basic",
                max_results=5,
                include_answer=True,
            )
            articles = results.get("results", [])
            answer = results.get("answer", "")

            parts = []
            if answer:
                parts.append(f"**Summary:** {answer}\n")
            for art in articles[:4]:
                parts.append(
                    f"**{art.get('title', 'Article')}**\n"
                    f"Source: {art.get('url', 'N/A')}\n"
                    f"{art.get('content', '')[:400]}\n"
                )
            return "\n---\n".join(parts) if parts else self._fallback_news()
        except Exception as exc:
            return f"News retrieval unavailable ({exc}). {self._fallback_news()}"

    def _fallback_news(self) -> str:
        return (
            "Live news retrieval is not configured. To enable real-time news, "
            "set the TAVILY_API_KEY environment variable.\n\n"
            "I can still discuss general financial concepts and market principles. "
            "What financial topic would you like to learn about?"
        )
