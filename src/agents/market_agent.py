"""
agents/market_agent.py
Provides real-time market data, quote lookups, and trend analysis.
"""
from __future__ import annotations

import json
import re

from src.agents.base_agent import BaseFinanceAgent
from src.core.state import AgentState
from src.utils.market_data import market_service


class MarketAnalysisAgent(BaseFinanceAgent):
    name = "Market Analysis Agent"
    description = "Provides real-time market data, stock quotes, and trend analysis"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a market analyst providing real-time market insights for retail investors. "
            "Given market data, you:\n"
            "- Explain what the numbers mean in plain English\n"
            "- Provide context (is this high/low historically?)\n"
            "- Highlight important trends or patterns\n"
            "- Always note that past performance doesn't guarantee future results\n"
            "- Never predict specific price movements\n"
            "Be concise, informative, and educational. Use bullet points for clarity."
        )

    def _run(self, state: AgentState) -> str:
        query = state.get("user_query", "")

        # Extract ticker symbols from query
        tickers = self._extract_tickers(query)

        market_data = {}

        if tickers:
            for ticker in tickers[:3]:  # Limit to 3 tickers per query
                quote = market_service.get_quote(ticker)
                company = market_service.get_company_info(ticker)
                history = market_service.get_history(ticker, period="3mo")
                market_data[ticker] = {
                    "quote": quote,
                    "company": company,
                    "price_trend": self._summarize_history(history),
                }
        else:
            # General market overview
            market_data["market_overview"] = market_service.get_market_overview()

        state["market_data"] = market_data

        data_str = json.dumps(market_data, indent=2, default=str)
        prompt = (
            f"## Market Data\n```json\n{data_str}\n```\n\n"
            f"## User Query\n{query}\n\n"
            "Provide a clear, educational market analysis based on this real-time data. "
            "Explain what the numbers mean and provide relevant context."
        )
        return self._chat(prompt)

    def _extract_tickers(self, text: str) -> list[str]:
        """Extract likely ticker symbols from user text."""
        # Match 1-5 uppercase letters (with optional $ prefix)
        candidates = re.findall(r"\$?([A-Z]{1,5})\b", text.upper())
        # Filter out common English words
        stop_words = {
            "A", "I", "IS", "IN", "IT", "AT", "TO", "THE", "AND", "OR",
            "FOR", "OF", "ON", "BE", "BY", "DO", "GO", "IF", "ME", "MY",
            "NO", "SO", "UP", "US", "WE", "AN", "AS", "AT", "BE", "DO",
            "GET", "HAD", "HAS", "HIM", "HIS", "HOW", "ITS", "MAY", "NEW",
            "NOW", "OLD", "OUR", "OUT", "OWN", "SAY", "SHE", "TOO", "USE",
            "WAS", "WAY", "WHO", "WHY", "WHAT", "WHEN", "WITH", "STOCK",
            "PRICE", "RATE", "FUND", "ETF", "BUY", "SELL", "HOLD", "MARKET",
        }
        return [c for c in candidates if c not in stop_words]

    def _summarize_history(self, history: list[dict]) -> dict:
        if not history:
            return {}
        prices = [h["close"] for h in history if "close" in h]
        if not prices:
            return {}
        return {
            "period_start": history[0].get("date"),
            "period_end": history[-1].get("date"),
            "start_price": round(prices[0], 2),
            "end_price": round(prices[-1], 2),
            "period_return_pct": round((prices[-1] - prices[0]) / prices[0] * 100, 2) if prices[0] else 0,
            "period_high": round(max(prices), 2),
            "period_low": round(min(prices), 2),
        }
