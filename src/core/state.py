"""
core/state.py
Shared LangGraph state used across all agents.
"""
from __future__ import annotations

from typing import Annotated, Any, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared mutable state passed between nodes in the LangGraph workflow."""

    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], add_messages]
    """Full conversation history (accumulates via add_messages reducer)."""

    user_query: str
    """Current raw user query."""

    # ── Routing ───────────────────────────────────────────────────────────────
    intent: str
    """Detected intent: finance_qa | portfolio | market | goal | news | tax | fallback"""

    active_agent: str
    """Name of the agent currently handling the request."""

    # ── Agent outputs ─────────────────────────────────────────────────────────
    agent_response: str
    """Final formatted response text from the selected agent."""

    rag_context: list[dict]
    """Retrieved RAG chunks: [{content, source, score}]"""

    market_data: dict[str, Any]
    """Raw market data payload from MarketDataService."""

    portfolio_data: dict[str, Any]
    """User portfolio submitted in this session."""

    goal_data: dict[str, Any]
    """User financial goal parameters."""

    # ── Metadata ──────────────────────────────────────────────────────────────
    error: str | None
    """Last error message, if any."""

    iterations: int
    """Guard counter to prevent infinite loops (max 10)."""

    session_id: str
    """Unique session identifier for logging/tracing."""
