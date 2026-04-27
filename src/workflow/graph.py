"""
workflow/graph.py
LangGraph StateGraph orchestrating all six specialized agents.
Includes intent classification, routing, and fallback handling.
"""
from __future__ import annotations

import logging
import uuid
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from src.core.state import AgentState
from src.core.llm_factory import get_llm

logger = logging.getLogger(__name__)

# ── Intent classification ─────────────────────────────────────────────────────

_INTENT_PROMPT = """Classify the user's financial query into exactly ONE of these categories.
Respond with ONLY the category name, nothing else.

Categories:
- finance_qa     : General financial education, concepts, definitions, how-to questions
- portfolio      : Portfolio review, stock holdings analysis, allocation, P&L
- market         : Stock price lookup, market data, current quotes, market overview
- goal           : Financial goal setting, savings projections, retirement planning
- news           : Financial news, current events, market movers
- tax            : Tax questions, IRA/401k/Roth accounts, capital gains, tax strategy
- fallback       : Unclear, off-topic, or requires human advisor

User query: {query}
Category:"""


def classify_intent(state: AgentState) -> AgentState:
    """LangGraph node: classify user intent using LLM."""
    query = state.get("user_query", "")
    if not query.strip():
        state["intent"] = "fallback"
        return state

    try:
        llm = get_llm(temperature=0.0, streaming=False)
        messages = [
            SystemMessage(content="You are a query classifier. Respond with exactly one category name."),
            HumanMessage(content=_INTENT_PROMPT.format(query=query)),
        ]
        response = llm.invoke(messages)
        intent = response.content.strip().lower()

        valid = {"finance_qa", "portfolio", "market", "goal", "news", "tax", "fallback"}
        if intent not in valid:
            intent = "finance_qa"  # safe default

        logger.info("Intent classified: %r → %s", query[:50], intent)
        state["intent"] = intent
    except Exception as exc:
        logger.error("Intent classification failed: %s", exc)
        state["intent"] = "finance_qa"  # safe default

    return state


def route_to_agent(state: AgentState) -> Literal[
    "finance_qa", "portfolio", "market", "goal", "news", "tax", "fallback"
]:
    """LangGraph conditional edge: maps intent to next node name."""
    intent = state.get("intent", "finance_qa")
    return intent


# ── Agent node wrappers ───────────────────────────────────────────────────────

def run_finance_qa(state: AgentState) -> AgentState:
    from src.agents.finance_qa_agent import FinanceQAAgent
    return FinanceQAAgent().run(state)


def run_portfolio(state: AgentState) -> AgentState:
    from src.agents.portfolio_agent import PortfolioAnalysisAgent
    return PortfolioAnalysisAgent().run(state)


def run_market(state: AgentState) -> AgentState:
    from src.agents.market_agent import MarketAnalysisAgent
    return MarketAnalysisAgent().run(state)


def run_goal(state: AgentState) -> AgentState:
    from src.agents.goal_planning_agent import GoalPlanningAgent
    return GoalPlanningAgent().run(state)


def run_news(state: AgentState) -> AgentState:
    from src.agents.news_agent import NewsSynthesizerAgent
    return NewsSynthesizerAgent().run(state)


def run_tax(state: AgentState) -> AgentState:
    from src.agents.tax_agent import TaxEducationAgent
    return TaxEducationAgent().run(state)


def run_fallback(state: AgentState) -> AgentState:
    """Graceful fallback for unclear or off-topic queries."""
    state["agent_response"] = (
        "I'm your AI Finance Assistant, specialized in personal finance and investing education. "
        "I can help you with:\n\n"
        "📚 **Financial Concepts** – Explain stocks, bonds, ETFs, and investment principles\n"
        "📊 **Portfolio Analysis** – Review your holdings and assess allocation/risk\n"
        "📈 **Market Data** – Look up stock prices and market overviews\n"
        "🎯 **Goal Planning** – Project your savings growth toward financial goals\n"
        "📰 **Financial News** – Summarize and contextualize market news\n"
        "💰 **Tax Education** – Explain tax-advantaged accounts and strategies\n\n"
        "What would you like to learn about today?"
    )
    state["active_agent"] = "Assistant"
    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", classify_intent)
    graph.add_node("finance_qa", run_finance_qa)
    graph.add_node("portfolio", run_portfolio)
    graph.add_node("market", run_market)
    graph.add_node("goal", run_goal)
    graph.add_node("news", run_news)
    graph.add_node("tax", run_tax)
    graph.add_node("fallback", run_fallback)

    # Entry point
    graph.set_entry_point("classify")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify",
        route_to_agent,
        {
            "finance_qa": "finance_qa",
            "portfolio": "portfolio",
            "market": "market",
            "goal": "goal",
            "news": "news",
            "tax": "tax",
            "fallback": "fallback",
        },
    )

    # All agent nodes → END
    for node in ["finance_qa", "portfolio", "market", "goal", "news", "tax", "fallback"]:
        graph.add_edge(node, END)

    return graph.compile()


# Module-level compiled graph (lazy singleton)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(
    user_query: str,
    portfolio_data: dict | None = None,
    goal_data: dict | None = None,
    session_id: str | None = None,
    message_history: list | None = None,
) -> AgentState:
    """
    High-level entrypoint: run a user query through the full agent graph.
    """
    from langchain_core.messages import HumanMessage

    session_id = session_id or str(uuid.uuid4())[:8]

    initial_state: AgentState = {
        "messages": (message_history or []) + [HumanMessage(content=user_query)],
        "user_query": user_query,
        "intent": "",
        "active_agent": "",
        "agent_response": "",
        "rag_context": [],
        "market_data": {},
        "portfolio_data": portfolio_data or {},
        "goal_data": goal_data or {},
        "error": None,
        "iterations": 0,
        "session_id": session_id,
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)
    return final_state
