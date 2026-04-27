"""
core/guardrails.py
Lightweight guardrails: disclaimer injection, harmful-content check,
and optional Guardrails AI integration.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ *This information is for educational purposes only and does not constitute "
    "financial, investment, or tax advice. Always consult a qualified financial "
    "professional before making investment decisions.*"
)

# Simple blocklist for clearly harmful/misleading financial advice patterns
_BLOCK_PATTERNS = [
    r"\b(guaranteed\s+returns?|100%\s+profit|risk.?free\s+investment)\b",
    r"\b(get.?rich.?quick|double\s+your\s+money\s+in)\b",
    r"\b(insider\s+tip|secret\s+stock|pump\s+and\s+dump)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _BLOCK_PATTERNS]


def check_response(text: str) -> str:
    """
    1. Scan for prohibited patterns → replace with safe message.
    2. Append mandatory disclaimer.
    """
    for pat in _COMPILED:
        if pat.search(text):
            logger.warning("Guardrail triggered: prohibited pattern detected.")
            text = (
                "I'm unable to provide that type of financial guidance as it may be "
                "misleading. Please consult a registered financial advisor for "
                "personalized investment advice."
            )
            break
    return text + DISCLAIMER


def validate_portfolio_input(portfolio: list[dict]) -> tuple[bool, str]:
    """
    Validate user-submitted portfolio dict list.
    Returns (is_valid, error_message).
    """
    required_keys = {"ticker", "shares", "avg_cost"}
    for item in portfolio:
        missing = required_keys - set(item.keys())
        if missing:
            return False, f"Portfolio item missing keys: {missing}"
        if not isinstance(item["shares"], (int, float)) or item["shares"] <= 0:
            return False, f"Shares must be a positive number for {item.get('ticker')}"
        if not isinstance(item["avg_cost"], (int, float)) or item["avg_cost"] < 0:
            return False, f"Average cost must be non-negative for {item.get('ticker')}"
    return True, ""


def sanitize_ticker(ticker: str) -> str:
    """Strip non-alphanumeric chars and upper-case."""
    return re.sub(r"[^A-Z0-9.\-]", "", ticker.upper())[:10]
