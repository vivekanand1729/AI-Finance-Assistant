"""
agents/goal_planning_agent.py
Helps users set financial goals and projects outcomes with compound interest modeling.
"""
from __future__ import annotations

import json
import re

from src.agents.base_agent import BaseFinanceAgent
from src.core.state import AgentState
from src.utils.portfolio_calculator import project_goal
from src.rag.retriever import get_retriever


class GoalPlanningAgent(BaseFinanceAgent):
    name = "Goal Planning Agent"
    description = "Helps users define financial goals and projects wealth accumulation over time"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a financial planning coach helping users define and achieve financial goals. "
            "You help users create SMART financial goals (Specific, Measurable, Achievable, "
            "Relevant, Time-bound). You:\n"
            "- Explain realistic return expectations based on risk tolerance\n"
            "- Show how small changes in savings rate compound dramatically\n"
            "- Recommend appropriate investment vehicles for different goals\n"
            "- Account for inflation, taxes, and market uncertainty\n"
            "- Use the provided projection data to illustrate timelines\n\n"
            "Be encouraging but realistic. Always mention that projections are estimates, not guarantees.\n"
            "Expected annual return guidelines:\n"
            "- Conservative (bonds): 3-5%\n"
            "- Moderate (60/40 portfolio): 5-7%\n"
            "- Growth (mostly stocks): 7-10%\n"
            "- Aggressive (all stocks): 9-12%"
        )

    def _run(self, state: AgentState) -> str:
        query = state.get("user_query", "")
        goal_data = state.get("goal_data", {})

        # Extract goal parameters from query if not already in state
        if not goal_data:
            goal_data = self._extract_goal_params(query)

        projection = None
        if goal_data.get("years") and goal_data.get("return_pct"):
            projection = project_goal(
                current_savings=goal_data.get("current_savings", 0),
                monthly_contribution=goal_data.get("monthly_contribution", 500),
                annual_return_pct=goal_data.get("return_pct", 7),
                years=goal_data.get("years", 10),
            )
            state["goal_data"]["projection"] = projection

        # RAG retrieval for goal-relevant content
        retriever = get_retriever()
        chunks = retriever.retrieve("financial goal planning savings compound interest", top_k=2)
        rag_text = self._build_rag_context(chunks)

        proj_text = ""
        if projection:
            milestones = [projection[i] for i in [0, 4, 9, 14, 19] if i < len(projection)]
            proj_text = f"\n## Financial Projection\n```json\n{json.dumps(milestones, indent=2)}\n```\n"

        prompt = (
            f"{rag_text}\n"
            f"{proj_text}\n"
            f"## Goal Parameters Identified\n```json\n{json.dumps(goal_data, indent=2)}\n```\n\n"
            f"## User Query\n{query}\n\n"
            "Provide a comprehensive goal planning response that:\n"
            "1. Confirms/clarifies the user's goals\n"
            "2. Interprets the projection data in plain language\n"
            "3. Suggests an actionable plan\n"
            "4. Recommends appropriate investment accounts and vehicles"
        )
        return self._chat(prompt)

    def _extract_goal_params(self, text: str) -> dict:
        """Extract numerical parameters from natural language goal description."""
        params = {}

        # Extract dollar amounts (current savings)
        savings_match = re.search(r"(?:have|saved|savings|currently)\s*\$?([\d,]+)k?", text, re.I)
        if savings_match:
            val = float(savings_match.group(1).replace(",", ""))
            if "k" in savings_match.group(0).lower():
                val *= 1000
            params["current_savings"] = val

        # Monthly contribution
        monthly_match = re.search(r"\$?([\d,]+)\s*(?:per month|monthly|a month|/month)", text, re.I)
        if monthly_match:
            params["monthly_contribution"] = float(monthly_match.group(1).replace(",", ""))

        # Time horizon
        years_match = re.search(r"(\d+)\s*(?:year|yr)", text, re.I)
        if years_match:
            params["years"] = int(years_match.group(1))

        # Risk tolerance → return estimate
        risk_lower = text.lower()
        if any(w in risk_lower for w in ["conservative", "safe", "low risk"]):
            params["return_pct"] = 4.0
            params["risk_profile"] = "Conservative"
        elif any(w in risk_lower for w in ["aggressive", "high risk", "maximum"]):
            params["return_pct"] = 10.0
            params["risk_profile"] = "Aggressive"
        else:
            params["return_pct"] = 7.0
            params["risk_profile"] = "Moderate"

        # Default values if not found
        params.setdefault("current_savings", 5000)
        params.setdefault("monthly_contribution", 500)
        params.setdefault("years", 10)

        return params
