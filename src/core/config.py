"""
core/config.py
Centralised configuration loader with env-override support.

Keys are NEVER validated at import time — they are checked lazily only
when an LLM call is actually made. This allows the Streamlit app to start
without a .env file and accept keys entered in the sidebar at runtime.
"""
from __future__ import annotations

import os
import yaml
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Silent=True: no error if .env doesn't exist (Streamlit Cloud, Docker, etc.)
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv optional; keys can still be set via env or sidebar

_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml() -> dict:
    cfg_path = _ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_CFG: dict = _load_yaml()


def get(section: str, key: str | None = None, default=None):
    """
    Retrieve a config value, falling back to default.

    Usage:
        get("llm", "model")           -> "gpt-4o-mini"
        get("rag", "top_k")           -> 5
        get("rag")                    -> full rag dict
    """
    section_data = _CFG.get(section, {})
    if key is None:
        return section_data
    return section_data.get(key, default)


# ── Convenience accessors ────────────────────────────────────────────────────

class LLMConfig:
    provider: str = get("llm", "provider", "openai")
    model: str = get("llm", "model", "gpt-4o-mini")
    temperature: float = get("llm", "temperature", 0.1)
    max_tokens: int = get("llm", "max_tokens", 2048)
    streaming: bool = get("llm", "streaming", True)


class RAGConfig:
    vector_db: str = get("rag", "vector_db", "faiss")
    embedding_model: str = get("rag", "embedding_model", "text-embedding-3-small")
    chunk_size: int = get("rag", "chunk_size", 600)
    chunk_overlap: int = get("rag", "chunk_overlap", 80)
    top_k: int = get("rag", "top_k", 5)
    score_threshold: float = get("rag", "score_threshold", 0.35)


class MarketConfig:
    primary_source: str = get("market_data", "primary_source", "yfinance")
    cache_ttl: int = get("market_data", "cache_ttl_seconds", 1800)
    max_retries: int = get("market_data", "max_retries", 3)
    timeout: int = get("market_data", "timeout_seconds", 10)


# ── API Key helpers ───────────────────────────────────────────────────────────

def openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")

def google_key() -> str:
    return os.getenv("GOOGLE_API_KEY", "")

def anthropic_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "")

def tavily_key() -> str:
    return os.getenv("TAVILY_API_KEY", "")

def langfuse_keys() -> tuple[str, str, str]:
    pub = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sec = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    return pub, sec, host
