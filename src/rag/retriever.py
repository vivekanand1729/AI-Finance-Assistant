"""
rag/retriever.py
RAG pipeline: build FAISS index from knowledge base, retrieve relevant chunks.
Falls back to keyword search if embeddings unavailable.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.rag.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)


class FinanceRAGRetriever:
    """
    Wraps a FAISS vector store built from the curated financial knowledge base.
    Provides retrieve() method returning ranked, scored chunks.
    """

    def __init__(self, top_k: int = 5, persist_dir: str | None = None):
        self.top_k = top_k
        self.persist_dir = persist_dir or str(Path(__file__).parent / "faiss_index")
        self._vectorstore = None
        self._docs = get_knowledge_base()

    # ── Public API ────────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        """Build or load FAISS index from knowledge base."""
        index_path = Path(self.persist_dir)

        if not force_rebuild and index_path.exists():
            try:
                self._load_index()
                logger.info("Loaded existing FAISS index from %s", self.persist_dir)
                return
            except Exception as exc:
                logger.warning("Failed to load index, rebuilding: %s", exc)

        try:
            self._build_faiss_index()
        except Exception as exc:
            logger.error("FAISS build failed (%s). Falling back to keyword search.", exc)
            self._vectorstore = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Retrieve most relevant knowledge chunks for a query.

        Returns list of:
          {"content": str, "title": str, "category": str, "source": str, "score": float}
        """
        k = top_k or self.top_k
        if self._vectorstore is not None:
            return self._faiss_retrieve(query, k)
        return self._keyword_retrieve(query, k)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_faiss_index(self) -> None:
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS
        from src.core.llm_factory import get_embeddings

        docs = [
            Document(
                page_content=f"{d['title']}\n\n{d['content']}",
                metadata={
                    "id": d["id"],
                    "category": d["category"],
                    "title": d["title"],
                    "source": d["source"],
                },
            )
            for d in self._docs
        ]

        embeddings = get_embeddings()
        logger.info("Building FAISS index with %d documents…", len(docs))
        self._vectorstore = FAISS.from_documents(docs, embeddings)
        self._vectorstore.save_local(self.persist_dir)
        logger.info("FAISS index saved to %s", self.persist_dir)

    def _load_index(self) -> None:
        from langchain_community.vectorstores import FAISS
        from src.core.llm_factory import get_embeddings

        embeddings = get_embeddings()
        self._vectorstore = FAISS.load_local(
            self.persist_dir, embeddings, allow_dangerous_deserialization=True
        )

    def _faiss_retrieve(self, query: str, k: int) -> list[dict]:
        results_with_scores = self._vectorstore.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "title": doc.metadata.get("title", ""),
                "category": doc.metadata.get("category", ""),
                "source": doc.metadata.get("source", ""),
                "score": float(1 / (1 + score)),  # convert L2 distance to similarity
            }
            for doc, score in results_with_scores
        ]

    def _keyword_retrieve(self, query: str, k: int) -> list[dict]:
        """Simple keyword overlap fallback when embeddings are unavailable."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self._docs:
            doc_text = f"{doc['title']} {doc['content']}".lower()
            overlap = sum(1 for w in query_words if w in doc_text)
            if overlap:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "content": f"{d['title']}\n\n{d['content']}",
                "title": d["title"],
                "category": d["category"],
                "source": d["source"],
                "score": score / max(len(query_words), 1),
            }
            for score, d in scored[:k]
        ]


# Module-level singleton (lazily initialized)
_retriever: FinanceRAGRetriever | None = None


def get_retriever() -> FinanceRAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FinanceRAGRetriever()
        _retriever.build_index()
    return _retriever
