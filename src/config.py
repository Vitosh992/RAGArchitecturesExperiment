from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    experiment: dict[str, Any]
    openrouter: dict[str, Any]
    dataset: dict[str, Any]
    retrieval: dict[str, Any]
    models: dict[str, Any]
    graph_rag: dict[str, Any]
    ragas: dict[str, Any]
    pipelines: list[str]

    @property
    def output_dir(self) -> Path:
        return Path(self.experiment.get("output_dir", "results"))

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if "openrouter" not in raw:
            raise ValueError(
                "config.yaml must contain an 'openrouter' section. "
                "See .env.example for API key setup."
            )
        return cls(
            experiment=raw["experiment"],
            openrouter=raw["openrouter"],
            dataset=raw["dataset"],
            retrieval=raw["retrieval"],
            models=raw["models"],
            graph_rag=raw["graph_rag"],
            ragas=raw["ragas"],
            pipelines=raw["pipelines"],
        )
