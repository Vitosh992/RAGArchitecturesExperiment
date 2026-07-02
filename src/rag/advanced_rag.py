from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from src.indexing.hybrid_retriever import HybridRetriever
from src.llm.protocol import ChatLLM
from src.rag.base import BaseRAGPipeline

HYDE_SYSTEM = (
    "Write a short scientific paragraph that would contain the answer to the "
    "following question. Use formal scientific language. Do not include the "
    "question itself."
)


class AdvancedRAGPipeline(BaseRAGPipeline):
    """Advanced RAG: HyDE query expansion + Cross-Encoder reranking."""

    name = "advanced"

    def __init__(
        self,
        llm: ChatLLM,
        retriever: HybridRetriever,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        top_k: int = 5,
        candidate_pool: int = 30,
    ) -> None:
        super().__init__(llm)
        self.retriever = retriever
        self.top_k = top_k
        self.candidate_pool = candidate_pool
        self.reranker = CrossEncoder(reranker_model)

    def _hyde_document(self, question: str) -> str:
        return self.llm.chat(HYDE_SYSTEM, question)

    def retrieve(self, question: str) -> list[str]:
        hyde_doc = self._hyde_document(question)
        hyde_emb = self.retriever.embedder.encode(
            [hyde_doc], normalize_embeddings=True
        )[0]
        q_emb = self.retriever.embedder.encode(
            [question], normalize_embeddings=True
        )[0]
        combined_emb = (hyde_emb + q_emb) / 2.0
        combined_emb = combined_emb / (np.linalg.norm(combined_emb) + 1e-9)

        candidate_ids = self.retriever.dense_search_with_embedding(
            combined_emb, self.candidate_pool
        )
        bm25_ids = self.retriever.bm25_search(question, self.candidate_pool)
        merged_ids = list(dict.fromkeys(candidate_ids + bm25_ids))

        pairs = [(question, self.retriever.corpus[i]) for i in merged_ids]
        rerank_scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(merged_ids, rerank_scores), key=lambda x: x[1], reverse=True
        )[: self.top_k]
        return [self.retriever.corpus[i] for i, _ in ranked]
