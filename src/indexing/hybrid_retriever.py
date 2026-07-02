from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.utils.device import resolve_device

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@dataclass
class RetrievalResult:
    doc_ids: list[int]
    texts: list[str]
    scores: list[float]


class HybridRetriever:
    """BM25 + BGE-M3 hybrid retrieval with Reciprocal Rank Fusion."""

    def __init__(
        self,
        corpus: list[str],
        embedding_model: str = "BAAI/bge-m3",
        device: str | None = None,
    ) -> None:
        self.corpus = corpus
        self.tokenized = [tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized)
        if device is None:
            device = resolve_device()
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.doc_embeddings = self.embedder.encode(
            corpus,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )

    def bm25_search(self, query: str, top_n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1][:top_n]
        return ranked.tolist()

    def dense_search(self, query: str, top_n: int) -> list[int]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = self.doc_embeddings @ q_emb
        ranked = np.argsort(sims)[::-1][:top_n]
        return ranked.tolist()

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        bm25_candidates: int = 50,
        dense_candidates: int = 50,
    ) -> RetrievalResult:
        bm25_ids = self.bm25_search(query, bm25_candidates)
        dense_ids = self.dense_search(query, dense_candidates)
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids])[:top_k]

        doc_ids = [doc_id for doc_id, _ in fused]
        scores = [score for _, score in fused]
        texts = [self.corpus[i] for i in doc_ids]
        return RetrievalResult(doc_ids=doc_ids, texts=texts, scores=scores)

    def dense_search_with_embedding(
        self,
        query_embedding: np.ndarray,
        top_n: int,
    ) -> list[int]:
        sims = self.doc_embeddings @ query_embedding
        ranked = np.argsort(sims)[::-1][:top_n]
        return ranked.tolist()
