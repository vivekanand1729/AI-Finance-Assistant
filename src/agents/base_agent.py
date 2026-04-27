"""
agents/base_agent.py
Abstract base class all specialized agents inherit from.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.state import AgentState
from src.core.guardrails import check_response

logger = logging.getLogger(__name__)


class BaseFinanceAgent(ABC):
    """
    Template for all specialized agents.

    Subclasses implement:
      - system_prompt  (property)
      - _run()         (core logic)

    The public run() method calls _run(), applies guardrails, and returns
    the updated AgentState.
    """

    #: human-readable name shown in UI
    name: str = "Base Agent"
    #: short description
    description: str = ""

    def __init__(self, llm=None):
        from src.core.llm_factory import get_llm
        self.llm = llm or get_llm()

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system prompt."""

    @abstractmethod
    def _run(self, state: AgentState) -> str:
        """Execute agent logic and return the raw response string."""

    # ── Public method ─────────────────────────────────────────────────────────

    def run(self, state: AgentState) -> AgentState:
        """
        Entrypoint called by the LangGraph workflow.
        Handles logging, error catching, and guardrail application.
        """
        logger.info("[%s] Processing: %s", self.name, state.get("user_query", "")[:80])
        try:
            raw_response = self._run(state)
            safe_response = check_response(raw_response)
            state["agent_response"] = safe_response
            state["active_agent"] = self.name
            state["error"] = None
        except Exception as exc:
            logger.exception("[%s] Error: %s", self.name, exc)
            state["agent_response"] = (
                f"I encountered an issue while processing your request ({exc}). "
                "Please try rephrasing or ask a different question."
            )
            state["error"] = str(exc)
        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _chat(self, user_content: str, extra_system: str = "") -> str:
        """Send a single turn to the LLM and return the text response."""
        system = self.system_prompt + ("\n\n" + extra_system if extra_system else "")
        messages = [SystemMessage(content=system), HumanMessage(content=user_content)]
        response = self.llm.invoke(messages)
        return response.content

    def _build_rag_context(self, chunks: list[dict]) -> str:
        """Format RAG chunks into a context block for the prompt."""
        if not chunks:
            return ""
        parts = ["## Relevant Financial Knowledge\n"]
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"**[{i}] {c.get('title', 'Reference')}** (Source: {c.get('source', 'N/A')})\n"
                f"{c.get('content', '')}\n"
            )
        return "\n".join(parts)
