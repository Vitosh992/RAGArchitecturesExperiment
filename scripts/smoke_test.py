#!/usr/bin/env python3
"""Smoke test: SciQ loader + hybrid retriever (no API calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data.sciq_loader import load_sciq_benchmark
from src.indexing.hybrid_retriever import HybridRetriever


def main() -> None:
    config = Config.load(ROOT / "config.yaml")
    print("Loading SciQ (5 samples)...")
    benchmark = load_sciq_benchmark(split="validation", max_samples=5)
    print(f"  OK: {len(benchmark.samples)} samples, {len(benchmark.corpus)} docs")

    print("Testing hybrid retriever (BM25 + BGE-M3)...")
    retriever = HybridRetriever(
        corpus=benchmark.corpus,
        embedding_model=config.models["embedding"],
    )
    q = benchmark.samples[0].question
    result = retriever.hybrid_search(q, top_k=3)
    print(f"  Query: {q[:80]}...")
    print(f"  Retrieved {len(result.texts)} chunks")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
