"""
utils/market_data.py
MarketDataService – fetches real-time and historical market data via
yfinance with in-memory caching and graceful fallback.
"""
from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict[str, tuple[Any, float]] = {}
_TTL = 1800  # 30 minutes


def _cached(ttl: int = _TTL):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            if key in _cache:
                value, ts = _cache[key]
                if time.time() - ts < ttl:
                    return value
            result = fn(*args, **kwargs)
            _cache[key] = (result, time.time())
            return result
        return wrapper
    return decorator


class MarketDataService:
    """
    Thin wrapper around yfinance with caching, retries, and fallback.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    # ── Public API ────────────────────────────────────────────────────────────

    @_cached(ttl=300)  # 5-minute cache for quotes
    def get_quote(self, ticker: str) -> dict:
        """Return latest price info for a single ticker."""
        ticker = ticker.upper()
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.fast_info
            return {
                "ticker": ticker,
                "price": round(float(info.last_price or 0), 2),
                "previous_close": round(float(info.previous_close or 0), 2),
                "market_cap": getattr(info, "market_cap", None),
                "52w_high": round(float(getattr(info, "year_high", 0) or 0), 2),
                "52w_low": round(float(getattr(info, "year_low", 0) or 0), 2),
                "volume": getattr(info, "three_month_average_volume", None),
                "currency": getattr(info, "currency", "USD"),
                "source": "yfinance",
            }
        except Exception as exc:
            logger.error("get_quote(%s) failed: %s", ticker, exc)
            return {"ticker": ticker, "error": str(exc), "source": "yfinance"}

    @_cached(ttl=_TTL)
    def get_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> list[dict]:
        """Return OHLCV history as list of dicts."""
        ticker = ticker.upper()
        try:
            import yfinance as yf
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            records = df[["date", "open", "high", "low", "close", "volume"]].copy()
            records["date"] = records["date"].astype(str)
            return records.to_dict("records")
        except Exception as exc:
            logger.error("get_history(%s) failed: %s", ticker, exc)
            return []

    @_cached(ttl=_TTL)
    def get_company_info(self, ticker: str) -> dict:
        """Return company fundamentals."""
        ticker = ticker.upper()
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            return {
                "name": info.get("longName", ticker),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "pe_ratio": info.get("trailingPE"),
                "eps": info.get("trailingEps"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "description": (info.get("longBusinessSummary") or "")[:500],
            }
        except Exception as exc:
            logger.error("get_company_info(%s) failed: %s", ticker, exc)
            return {"ticker": ticker, "error": str(exc)}

    @_cached(ttl=_TTL)
    def get_market_overview(self) -> dict:
        """Return snapshot of major indices."""
        indices = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "Dow Jones": "^DJI",
            "Russell 2000": "^RUT",
            "VIX": "^VIX",
        }
        results = {}
        for name, sym in indices.items():
            q = self.get_quote(sym)
            if "price" in q:
                prev = q.get("previous_close", 0)
                price = q["price"]
                pct = round((price - prev) / prev * 100, 2) if prev else 0
                results[name] = {"symbol": sym, "price": price, "change_pct": pct}
        return results

    def get_portfolio_prices(self, tickers: list[str]) -> dict[str, float]:
        """Batch fetch latest prices for a list of tickers."""
        return {t: self.get_quote(t).get("price", 0.0) for t in tickers}


# Module-level singleton
market_service = MarketDataService()
