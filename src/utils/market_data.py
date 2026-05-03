"""
utils/market_data.py
MarketDataService – fetches real-time and historical market data via
Finnhub REST API with in-memory caching and graceful fallback.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any

import requests

from src.core.config import finnhub_key, MarketConfig

logger = logging.getLogger(__name__)

_BASE = "https://finnhub.io/api/v1"

_PERIOD_DAYS: dict[str, int] = {
    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
    "6mo": 180, "1y": 365, "2y": 730, "5y": 1825,
}

_RESOLUTION_MAP: dict[str, str] = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "60m": "60", "1h": "60", "1d": "D", "1wk": "W", "1mo": "M",
}

# ETF proxies used for the market overview (reliable on Finnhub free tier)
_MARKET_INDICES: dict[str, str] = {
    "S&P 500":     "SPY",
    "NASDAQ":      "QQQ",
    "Dow Jones":   "DIA",
    "Russell 2000": "IWM",
    "VIX":         "VIXY",
}

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
    """Thin wrapper around Finnhub REST API with caching and fallback."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict:
        """Execute a Finnhub REST call; returns empty dict on any failure."""
        p = dict(params or {})
        key = finnhub_key()
        if not key:
            logger.warning("FINNHUB_API_KEY is not configured.")
            return {}
        p["token"] = key
        try:
            resp = requests.get(
                f"{_BASE}/{endpoint}",
                params=p,
                timeout=MarketConfig.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Finnhub [%s] error: %s", endpoint, exc)
            return {}

    # ── Public API ────────────────────────────────────────────────────────────

    @_cached(ttl=300)
    def get_quote(self, ticker: str) -> dict:
        """Return latest price info for a single ticker."""
        ticker = ticker.upper()
        try:
            q = self._get("quote", {"symbol": ticker})
            m = self._get("stock/metric", {"symbol": ticker, "metric": "all"})
            metric = m.get("metric", {})

            price = float(q.get("c") or 0)
            prev  = float(q.get("pc") or 0)
            change_pct = q.get("dp")
            if change_pct is None and prev:
                change_pct = round((price - prev) / prev * 100, 2)

            return {
                "ticker":         ticker,
                "price":          round(price, 2),
                "previous_close": round(prev, 2),
                "change_pct":     round(float(change_pct or 0), 2),
                "market_cap":     None,  # use get_company_info() for market cap
                "52w_high":       metric.get("52WeekHigh"),
                "52w_low":        metric.get("52WeekLow"),
                "volume":         metric.get("10DayAverageTradingVolume"),
                "currency":       "USD",
                "source":         "finnhub",
            }
        except Exception as exc:
            logger.error("get_quote(%s) failed: %s", ticker, exc)
            return {"ticker": ticker, "error": str(exc), "source": "finnhub"}

    @_cached(ttl=_TTL)
    def get_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> list[dict]:
        """Return OHLCV history as list of dicts."""
        ticker = ticker.upper()
        try:
            days       = _PERIOD_DAYS.get(period, 365)
            resolution = _RESOLUTION_MAP.get(interval, "D")
            to_ts      = int(time.time())
            from_ts    = to_ts - days * 86400

            data = self._get("stock/candle", {
                "symbol":     ticker,
                "resolution": resolution,
                "from":       from_ts,
                "to":         to_ts,
            })

            if data.get("s") != "ok" or not data.get("t"):
                logger.warning("get_history(%s): no data (status=%s)", ticker, data.get("s"))
                return []

            return [
                {
                    "date":   datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                    "open":   data["o"][i],
                    "high":   data["h"][i],
                    "low":    data["l"][i],
                    "close":  data["c"][i],
                    "volume": data["v"][i],
                }
                for i, ts in enumerate(data["t"])
            ]
        except Exception as exc:
            logger.error("get_history(%s) failed: %s", ticker, exc)
            return []

    @_cached(ttl=_TTL)
    def get_company_info(self, ticker: str) -> dict:
        """Return company fundamentals."""
        ticker = ticker.upper()
        try:
            profile = self._get("stock/profile2", {"symbol": ticker})
            m       = self._get("stock/metric",   {"symbol": ticker, "metric": "all"})
            metric  = m.get("metric", {})

            # Finnhub returns dividend yield as a percentage (e.g. 0.82 means 0.82%)
            raw_dy = metric.get("dividendYieldIndicatedAnnual") or 0
            dividend_yield = raw_dy / 100 if raw_dy else None

            market_cap_m = profile.get("marketCapitalization") or 0

            return {
                "name":           profile.get("name", ticker),
                "sector":         profile.get("finnhubIndustry", "N/A"),
                "industry":       profile.get("finnhubIndustry", "N/A"),
                "pe_ratio":       metric.get("peTTM") or metric.get("peBasicExclExtraTTM"),
                "eps":            metric.get("epsTTM") or metric.get("epsBasicExclExtraAnnual"),
                "dividend_yield": dividend_yield,
                "beta":           metric.get("beta"),
                "market_cap":     market_cap_m * 1_000_000 if market_cap_m else None,
                "currency":       profile.get("currency", "USD"),
                "description":    "",
            }
        except Exception as exc:
            logger.error("get_company_info(%s) failed: %s", ticker, exc)
            return {"ticker": ticker, "error": str(exc)}

    @_cached(ttl=_TTL)
    def get_market_overview(self) -> dict:
        """Return snapshot of major market indices via ETF proxies."""
        results = {}
        for name, sym in _MARKET_INDICES.items():
            q = self.get_quote(sym)
            if "price" in q and not q.get("error"):
                results[name] = {
                    "symbol":     sym,
                    "price":      q["price"],
                    "change_pct": q.get("change_pct", 0),
                }
        return results

    def get_portfolio_prices(self, tickers: list[str]) -> dict[str, float]:
        """Batch fetch latest prices for a list of tickers."""
        return {t: self.get_quote(t).get("price", 0.0) for t in tickers}


# Module-level singleton
market_service = MarketDataService()
