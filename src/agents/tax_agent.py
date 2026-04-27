"""
agents/tax_agent.py
Explains tax concepts, account types, and tax-efficient investing strategies.
"""
from __future__ import annotations

from src.agents.base_agent import BaseFinanceAgent
from src.core.state import AgentState
from src.rag.retriever import get_retriever


class TaxEducationAgent(BaseFinanceAgent):
    name = "Tax Education Agent"
    description = "Explains investment tax concepts, account types, and tax-efficient strategies"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a tax education specialist helping investors understand how taxes affect "
            "investment returns. You:\n"
            "- Explain tax concepts clearly with concrete examples\n"
            "- Cover account types (401k, IRA, Roth, HSA, 529, taxable)\n"
            "- Explain capital gains (short vs long-term), dividends, and ordinary income\n"
            "- Teach tax-efficient strategies (tax-loss harvesting, asset location)\n"
            "- Always clarify this is EDUCATION, not tax advice\n"
            "- Recommend consulting a CPA/tax professional for individual situations\n\n"
            "Tax year: use current general tax rules (2024 brackets, limits).\n"
            "NEVER file taxes for the user or give specific tax advice for their situation."
        )

    def _run(self, state: AgentState) -> str:
        query = state.get("user_query", "")

        # RAG retrieval for tax content
        retriever = get_retriever()
        chunks = retriever.retrieve(query, top_k=4)
        state["rag_context"] = chunks

        # Filter for tax-relevant chunks
        tax_chunks = [c for c in chunks if "Tax" in c.get("category", "") or "tax" in c.get("content", "").lower()]
        rag_chunks = tax_chunks or chunks[:3]
        rag_text = self._build_rag_context(rag_chunks)

        prompt = (
            f"{rag_text}\n\n"
            f"## User Tax Question\n{query}\n\n"
            "Provide a comprehensive, educational answer about the tax topic. "
            "Include:\n"
            "1. Clear explanation of the concept\n"
            "2. Concrete numerical examples where helpful\n"
            "3. Relevant account types or strategies\n"
            "4. Common mistakes to avoid\n"
            "Reference the knowledge base sources provided."
        )
        return self._chat(prompt)
