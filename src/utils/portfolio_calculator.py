"""
utils/portfolio_calculator.py
Pure-function portfolio analytics: allocation, P&L, diversification score,
risk assessment, and Sharpe-like metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PortfolioHolding:
    ticker: str
    shares: float
    avg_cost: float          # per share
    current_price: float = 0.0
    sector: str = "Unknown"

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost

    @property
    def current_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.current_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


@dataclass
class PortfolioMetrics:
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_pct: float
    holdings: list[PortfolioHolding]
    allocation: dict[str, float]           # ticker -> % of total
    sector_allocation: dict[str, float]    # sector -> % of total
    diversification_score: float           # 0-100
    risk_level: str                        # Conservative / Moderate / Aggressive
    recommendations: list[str]


def calculate_portfolio_metrics(
    holdings: list[dict[str, Any]],
    current_prices: dict[str, float],
    sector_map: dict[str, str] | None = None,
) -> PortfolioMetrics:
    """
    Compute comprehensive portfolio metrics.

    Args:
        holdings: list of {"ticker", "shares", "avg_cost"}
        current_prices: {"TICKER": price}
        sector_map: {"TICKER": "sector"} (optional)
    """
    sm = sector_map or {}
    objs: list[PortfolioHolding] = []

    for h in holdings:
        ticker = h["ticker"].upper()
        obj = PortfolioHolding(
            ticker=ticker,
            shares=float(h["shares"]),
            avg_cost=float(h["avg_cost"]),
            current_price=float(current_prices.get(ticker, h["avg_cost"])),
            sector=sm.get(ticker, "Unknown"),
        )
        objs.append(obj)

    total_value = sum(o.current_value for o in objs)
    total_cost = sum(o.cost_basis for o in objs)
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0.0

    allocation: dict[str, float] = {}
    sector_allocation: dict[str, float] = {}
    for o in objs:
        pct = (o.current_value / total_value * 100) if total_value else 0.0
        allocation[o.ticker] = round(pct, 2)
        sector_allocation[o.sector] = sector_allocation.get(o.sector, 0.0) + pct

    sector_allocation = {k: round(v, 2) for k, v in sector_allocation.items()}

    # Herfindahl–Hirschman Index for diversification (lower = more diversified)
    hhi = sum((w / 100) ** 2 for w in allocation.values())
    n = len(objs)
    # Normalize: perfect diversification → hhi = 1/n; max concentration → hhi = 1
    diversification_score = 100.0
    if n > 0 and hhi > 1 / n:
        diversification_score = max(0, round(100 * (1 - (hhi - 1 / n) / (1 - 1 / n)), 1))

    risk_level = _assess_risk(sector_allocation, allocation)
    recommendations = _generate_recommendations(objs, allocation, sector_allocation, diversification_score)

    return PortfolioMetrics(
        total_value=round(total_value, 2),
        total_cost=round(total_cost, 2),
        total_pnl=round(total_pnl, 2),
        total_pnl_pct=round(total_pnl_pct, 2),
        holdings=objs,
        allocation=allocation,
        sector_allocation=sector_allocation,
        diversification_score=diversification_score,
        risk_level=risk_level,
        recommendations=recommendations,
    )


def _assess_risk(sector_alloc: dict[str, float], ticker_alloc: dict[str, float]) -> str:
    high_risk_sectors = {"Technology", "Crypto", "Biotech", "Energy"}
    hr_pct = sum(v for k, v in sector_alloc.items() if k in high_risk_sectors)
    max_single = max(ticker_alloc.values()) if ticker_alloc else 0
    if hr_pct > 60 or max_single > 50:
        return "Aggressive"
    if hr_pct > 35 or max_single > 30:
        return "Moderate-Aggressive"
    return "Conservative" if max_single < 15 and hr_pct < 20 else "Moderate"


def _generate_recommendations(
    holdings: list[PortfolioHolding],
    allocation: dict[str, float],
    sector_alloc: dict[str, float],
    div_score: float,
) -> list[str]:
    recs = []
    if div_score < 50:
        recs.append("⚠️ Portfolio is highly concentrated. Consider diversifying across more assets.")
    max_ticker = max(allocation, key=allocation.__getitem__) if allocation else None
    if max_ticker and allocation[max_ticker] > 40:
        recs.append(f"📊 {max_ticker} makes up {allocation[max_ticker]:.1f}% of your portfolio — consider trimming.")
    losers = [h for h in holdings if h.unrealized_pnl_pct < -15]
    for loser in losers[:2]:
        recs.append(f"📉 {loser.ticker} is down {abs(loser.unrealized_pnl_pct):.1f}%. Review your thesis.")
    if not any(s in sector_alloc for s in ("Consumer Staples", "Healthcare", "Utilities")):
        recs.append("🛡️ Consider adding defensive sectors (Healthcare, Utilities) for stability.")
    if not recs:
        recs.append("✅ Portfolio looks reasonably balanced. Continue monitoring regularly.")
    return recs


def project_goal(
    current_savings: float,
    monthly_contribution: float,
    annual_return_pct: float,
    years: int,
) -> list[dict]:
    """
    Compound interest projection for goal planning.
    Returns year-by-year breakdown.
    """
    rate = annual_return_pct / 100
    monthly_rate = rate / 12
    results = []
    balance = current_savings
    for y in range(1, years + 1):
        for _ in range(12):
            balance = balance * (1 + monthly_rate) + monthly_contribution
        results.append({
            "year": y,
            "balance": round(balance, 2),
            "contributions_total": round(current_savings + monthly_contribution * 12 * y, 2),
            "growth": round(balance - current_savings - monthly_contribution * 12 * y, 2),
        })
    return results
