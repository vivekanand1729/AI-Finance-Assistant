"""
core/llm_factory.py
Builds LangChain chat models from config/env, with Langfuse callback support.
"""
from __future__ import annotations

import logging
from typing import Any

from src.core.config import LLMConfig, openai_key, google_key, anthropic_key, langfuse_keys

logger = logging.getLogger(__name__)


def _langfuse_callback():
    """Return Langfuse callback handler if keys are set, else None."""
    try:
        pub, sec, host = langfuse_keys()
        if pub and sec:
            from langfuse.callback import CallbackHandler
            return CallbackHandler(public_key=pub, secret_key=sec, host=host)
    except Exception as exc:
        logger.debug("Langfuse not available: %s", exc)
    return None


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    streaming: bool | None = None,
    **kwargs: Any,
):
    """
    Factory function – returns a BaseChatModel configured from cfg/env.

    Priority: explicit kwargs > config.yaml > sensible defaults.
    Also attaches Langfuse tracing callback when keys are present.
    """
    prov = (provider or LLMConfig.provider).lower()
    mdl = model or LLMConfig.model
    temp = temperature if temperature is not None else LLMConfig.temperature
    strm = streaming if streaming is not None else LLMConfig.streaming

    callbacks = []
    cb = _langfuse_callback()
    if cb:
        callbacks.append(cb)

    common = dict(temperature=temp, streaming=strm, callbacks=callbacks or None, **kwargs)

    if prov == "openai":
        from langchain_openai import ChatOpenAI
        api_key = openai_key()
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        return ChatOpenAI(model=mdl, openai_api_key=api_key, **common)

    elif prov == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = google_key()
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set.")
        return ChatGoogleGenerativeAI(model=mdl, google_api_key=api_key, **common)

    elif prov in ("anthropic", "claude"):
        from langchain_anthropic import ChatAnthropic
        api_key = anthropic_key()
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")
        return ChatAnthropic(model=mdl, anthropic_api_key=api_key, **common)

    else:
        raise ValueError(f"Unknown LLM provider: {prov!r}. Choose openai / google / anthropic.")


def get_embeddings(model: str | None = None):
    """Return an embeddings model (OpenAI by default)."""
    from langchain_openai import OpenAIEmbeddings
    emb_model = model or "text-embedding-3-small"
    return OpenAIEmbeddings(model=emb_model, openai_api_key=openai_key())
