"""
core/guardrails.py
Two-layer guardrails: fast regex blocklist + Guardrails AI SDK validation.
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

_BLOCK_PATTERNS = [
    r"\b(guaranteed\s+returns?|100%\s+profit|risk.?free\s+investment)\b",
    r"\b(get.?rich.?quick|double\s+your\s+money\s+in)\b",
    r"\b(insider\s+tip|secret\s+stock|pump\s+and\s+dump)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _BLOCK_PATTERNS]

_guard = None
_guardrails_enabled = False


def _setup_guardrails_sdk() -> None:
    """Initialize Guardrails AI SDK with the API key from the environment."""
    global _guard, _guardrails_enabled

    api_key = os.getenv("GUARDRAILS_API_KEY", "")
    if not api_key:
        return

    try:
        os.environ["GUARDRAILS_API_KEY"] = api_key

        from guardrails import Guard  # noqa: PLC0415

        try:
            # ToxicLanguage requires: guardrails hub install hub://guardrails/toxic_language
            from guardrails.hub import ToxicLanguage  # noqa: PLC0415
            _guard = Guard().use(ToxicLanguage(threshold=0.5, on_fail="noop"))
            _guardrails_enabled = True
            logger.info("Guardrails AI SDK initialized with ToxicLanguage validator.")
        except (ImportError, Exception):
            # Hub validator not installed yet — basic Guard still authenticates
            _guard = Guard()
            _guardrails_enabled = True
            logger.info(
                "Guardrails AI SDK initialized (basic mode). "
                "Run: guardrails hub install hub://guardrails/toxic_language "
                "to enable content safety validation."
            )

    except ImportError:
        logger.warning("guardrails-ai not installed; using pattern-based guardrails only.")
    except Exception as exc:
        logger.warning("Guardrails AI setup failed: %s", exc)


_setup_guardrails_sdk()


def check_response(text: str) -> str:
    """
    Layer 1 — regex blocklist for harmful financial advice patterns.
    Layer 2 — Guardrails AI SDK validation (when configured).
    Appends mandatory disclaimer.
    """
    # Layer 1: fast pattern check
    for pat in _COMPILED:
        if pat.search(text):
            logger.warning("Guardrail triggered: prohibited pattern detected.")
            text = (
                "I'm unable to provide that type of financial guidance as it may be "
                "misleading. Please consult a registered financial advisor for "
                "personalized investment advice."
            )
            break

    # Layer 2: Guardrails AI SDK
    if _guardrails_enabled and _guard:
        try:
            result = _guard.validate(text)
            if not result.validation_passed:
                logger.warning("Guardrails AI validation failed for response.")
                text = (
                    "I'm unable to provide that response as it did not pass safety "
                    "validation. Please consult a registered financial advisor for "
                    "personalized investment advice."
                )
            elif result.validated_output:
                text = result.validated_output
        except Exception as exc:
            logger.warning("Guardrails AI validation error: %s", exc)

    return text + DISCLAIMER


def guardrails_status() -> dict:
    """Return Guardrails AI integration status for display in the UI."""
    return {
        "sdk_enabled": _guardrails_enabled,
        "guard_configured": _guard is not None,
        "api_key_set": bool(os.getenv("GUARDRAILS_API_KEY", "")),
    }


def validate_portfolio_input(portfolio: list[dict]) -> tuple[bool, str]:
    """Validate user-submitted portfolio dict list. Returns (is_valid, error_message)."""
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
