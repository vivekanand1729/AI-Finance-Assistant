"""
agents/finance_qa_agent.py
Handles general financial education queries with RAG-grounded responses.
"""
from __future__ import annotations

from src.agents.base_agent import BaseFinanceAgent
from src.core.state import AgentState
from src.rag.retriever import get_retriever


class FinanceQAAgent(BaseFinanceAgent):
    name = "Finance Q&A Agent"
    description = "Answers general financial education questions with RAG-grounded responses"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a knowledgeable, friendly financial educator helping beginners understand "
            "personal finance and investing. You explain concepts clearly using everyday language, "
            "concrete examples, and analogies. You always:\n"
            "- Ground responses in the provided knowledge base excerpts\n"
            "- Cite sources when available\n"
            "- Distinguish between facts and general principles\n"
            "- Encourage consulting professionals for personalized advice\n"
            "- Keep responses structured and easy to follow (use bullet points, examples)\n\n"
            "You do NOT give specific investment recommendations or predict market movements."
        )

    def _run(self, state: AgentState) -> str:
        query = state.get("user_query", "")

        # RAG retrieval
        retriever = get_retriever()
        chunks = retriever.retrieve(query, top_k=4)
        state["rag_context"] = chunks

        rag_text = self._build_rag_context(chunks)
        prompt = (
            f"{rag_text}\n\n"
            f"## User Question\n{query}\n\n"
            "Please provide a clear, educational answer based on the knowledge above. "
            "Reference the sources where appropriate."
        )
        return self._chat(prompt)
