#!/usr/bin/env python3
"""Pre-download local retrieval models (embedding + reranker)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _check_deps() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        print(
            "Install dependencies first:\n"
            "  python -m venv .venv\n"
            "  .venv\\Scripts\\Activate.ps1   # Windows\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def download_models(embedding: str, reranker: str) -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"Embedding: {embedding}")
    SentenceTransformer(embedding)

    print(f"Reranker: {reranker}")
    CrossEncoder(reranker)

    print("Retrieval models ready.")


def main() -> None:
    _check_deps()

    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("Pre-downloading retrieval models...")
    download_models(cfg["models"]["embedding"], cfg["models"]["reranker"])


if __name__ == "__main__":
    main()
