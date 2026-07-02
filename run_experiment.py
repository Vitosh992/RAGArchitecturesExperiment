#!/usr/bin/env python3
"""Compare Naive / Advanced / GraphRAG on SciQ with RAGAS metrics (OpenRouter API)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.evaluation.run_benchmark import run_benchmark


def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipelines on SciQ (OpenRouter + RAGAS)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    run_benchmark(args.config)


if __name__ == "__main__":
    main()
