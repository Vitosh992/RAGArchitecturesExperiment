from __future__ import annotations

import re
from collections import defaultdict

import igraph as ig
import leidenalg
import numpy as np

from src.indexing.hybrid_retriever import HybridRetriever
from src.llm.protocol import ChatLLM
from src.rag.base import BaseRAGPipeline

ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:-[A-Z]?[a-z]+)*|[A-Z]{2,})\b"
    r"|\b\d+(?:\.\d+)?(?:\s*(?:mg|kg|km|m|cm|mm|°C|K|Pa|atm|mol|Hz|kHz|MHz|GHz))?\b"
)


class GraphRAGPipeline(BaseRAGPipeline):
    """GraphRAG: entity graph + Leiden clustering + hybrid retrieval."""

    name = "graph"

    def __init__(
        self,
        llm: ChatLLM,
        retriever: HybridRetriever,
        top_k: int = 5,
        min_cluster_size: int = 3,
        resolution: float = 1.0,
    ) -> None:
        super().__init__(llm)
        self.retriever = retriever
        self.top_k = top_k
        self.min_cluster_size = min_cluster_size
        self.resolution = resolution
        self.doc_entities = self._extract_all_entities()
        self.clusters = self._build_clusters()

    def _extract_entities(self, text: str) -> set[str]:
        found = ENTITY_PATTERN.findall(text)
        stop = {"The", "This", "These", "When", "Which", "What", "How", "Why"}
        return {e.lower() for e in found if e not in stop and len(e) > 2}

    def _extract_all_entities(self) -> list[set[str]]:
        return [self._extract_entities(doc) for doc in self.retriever.corpus]

    def _build_clusters(self) -> dict[int, list[int]]:
        entity_to_docs: dict[str, set[int]] = defaultdict(set)
        for doc_id, entities in enumerate(self.doc_entities):
            for ent in entities:
                entity_to_docs[ent].add(doc_id)

        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for docs in entity_to_docs.values():
            doc_list = sorted(docs)
            for i in range(len(doc_list)):
                for j in range(i + 1, len(doc_list)):
                    edges.append((doc_list[i], doc_list[j]))
                    weights.append(1.0)

        n = len(self.retriever.corpus)
        if not edges:
            return {0: list(range(n))}

        g = ig.Graph(n=n, edges=edges, directed=False)
        g.es["weight"] = weights
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=self.resolution,
        )

        clusters: dict[int, list[int]] = defaultdict(list)
        for doc_id, cluster_id in enumerate(partition.membership):
            clusters[cluster_id].append(doc_id)
        return dict(clusters)

    def _query_entities(self, question: str) -> set[str]:
        return self._extract_entities(question)

    def retrieve(self, question: str) -> list[str]:
        q_entities = self._query_entities(question)
        cluster_scores: dict[int, float] = {}

        for cluster_id, doc_ids in self.clusters.items():
            if len(doc_ids) < self.min_cluster_size:
                continue
            overlap = sum(
                1
                for d in doc_ids
                if self.doc_entities[d] & q_entities
            )
            if overlap > 0:
                cluster_scores[cluster_id] = overlap / len(doc_ids)

        if cluster_scores:
            best_cluster = max(cluster_scores, key=cluster_scores.get)
            pool_ids = self.clusters[best_cluster]
        else:
            pool_ids = list(range(len(self.retriever.corpus)))

        q_emb = self.retriever.embedder.encode(
            [question], normalize_embeddings=True
        )[0]
        pool_embs = self.retriever.doc_embeddings[pool_ids]
        sims = pool_embs @ q_emb
        top_local = np.argsort(sims)[::-1][: self.top_k]
        return [self.retriever.corpus[pool_ids[i]] for i in top_local]
