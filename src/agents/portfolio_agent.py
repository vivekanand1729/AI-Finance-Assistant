"""
agents/portfolio_agent.py
Analyzes user portfolios: metrics, risk, allocation, recommendations.
"""
from __future__ import annotations

import json

from src.agents.base_agent import BaseFinanceAgent
from src.core.state import AgentState
from src.utils.portfolio_calculator import calculate_portfolio_metrics
from src.utils.market_data import market_service


class PortfolioAnalysisAgent(BaseFinanceAgent):
    name = "Portfolio Analysis Agent"
    description = "Analyzes portfolio allocation, risk, P&L, and provides actionable recommendations"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a portfolio analyst helping retail investors understand their holdings. "
            "Given portfolio metrics and data, you:\n"
            "- Summarize the portfolio's overall health clearly\n"
            "- Highlight strengths and areas of concern\n"
            "- Explain diversification and risk level in simple terms\n"
            "- Provide specific, actionable (but educational) recommendations\n"
            "- Never recommend specific buy/sell actions; focus on education\n"
            "Format your response with clear sections: Overview, Risk Assessment, Key Insights, Recommendations."
        )

    def _run(self, state: AgentState) -> str:
        portfolio_data = state.get("portfolio_data", {})
        holdings_list = portfolio_data.get("holdings", [])

        if not holdings_list:
            return (
                "I don't have any portfolio data to analyze. Please provide your holdings "
                "in the Portfolio tab with ticker symbols, number of shares, and average cost per share."
            )

        # Fetch current prices
        tickers = [h["ticker"].upper() for h in holdings_list]
        prices = market_service.get_portfolio_prices(tickers)

        # Fetch sector info
        sector_map = {}
        for t in tickers:
            info = market_service.get_company_info(t)
            sector_map[t] = info.get("sector", "Unknown")

        # Calculate metrics
        metrics = calculate_portfolio_metrics(holdings_list, prices, sector_map)

        # Build analysis prompt
        metrics_json = json.dumps(
            {
                "total_value": metrics.total_value,
                "total_cost": metrics.total_cost,
                "total_pnl": metrics.total_pnl,
                "total_pnl_pct": metrics.total_pnl_pct,
                "allocation": metrics.allocation,
                "sector_allocation": metrics.sector_allocation,
                "diversification_score": metrics.diversification_score,
                "risk_level": metrics.risk_level,
                "holdings": [
                    {
                        "ticker": h.ticker,
                        "current_value": h.current_value,
                        "unrealized_pnl": h.unrealized_pnl,
                        "unrealized_pnl_pct": h.unrealized_pnl_pct,
                    }
                    for h in metrics.holdings
                ],
            },
            indent=2,
        )

        prompt = (
            f"Here is the calculated portfolio metrics:\n\n```json\n{metrics_json}\n```\n\n"
            f"Also, these specific recommendations were generated:\n"
            + "\n".join(f"- {r}" for r in metrics.recommendations)
            + f"\n\nUser query: {state.get('user_query', 'Analyze my portfolio')}\n\n"
            "Provide a comprehensive, beginner-friendly portfolio analysis based on this data."
        )

        # Store metrics for UI rendering
        state["portfolio_data"]["metrics"] = {
            "total_value": metrics.total_value,
            "total_cost": metrics.total_cost,
            "total_pnl": metrics.total_pnl,
            "total_pnl_pct": metrics.total_pnl_pct,
            "allocation": metrics.allocation,
            "sector_allocation": metrics.sector_allocation,
            "diversification_score": metrics.diversification_score,
            "risk_level": metrics.risk_level,
            "holdings_detail": [
                {
                    "ticker": h.ticker,
                    "shares": h.shares,
                    "avg_cost": h.avg_cost,
                    "current_price": h.current_price,
                    "current_value": h.current_value,
                    "unrealized_pnl": h.unrealized_pnl,
                    "unrealized_pnl_pct": h.unrealized_pnl_pct,
                }
                for h in metrics.holdings
            ],
            "recommendations": metrics.recommendations,
        }

        return self._chat(prompt)
