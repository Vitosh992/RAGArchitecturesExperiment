from __future__ import annotations

from src.indexing.hybrid_retriever import HybridRetriever
from src.llm.protocol import ChatLLM
from src.rag.base import BaseRAGPipeline


class NaiveRAGPipeline(BaseRAGPipeline):
    """Naive RAG: BM25 + BGE-M3 hybrid retrieval."""

    name = "naive"

    def __init__(
        self,
        llm: ChatLLM,
        retriever: HybridRetriever,
        top_k: int = 5,
        bm25_candidates: int = 50,
        dense_candidates: int = 50,
    ) -> None:
        super().__init__(llm)
        self.retriever = retriever
        self.top_k = top_k
        self.bm25_candidates = bm25_candidates
        self.dense_candidates = dense_candidates

    def retrieve(self, question: str) -> list[str]:
        result = self.retriever.hybrid_search(
            question,
            top_k=self.top_k,
            bm25_candidates=self.bm25_candidates,
            dense_candidates=self.dense_candidates,
        )
        return result.texts
