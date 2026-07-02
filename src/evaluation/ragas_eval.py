from __future__ import annotations

from typing import Any

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from src.llm.protocol import ChatLLM
from src.rag.base import RAGOutput


METRIC_MAP = {
    "faithfulness": faithfulness,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


def build_run_config(ragas_cfg: dict[str, Any]) -> RunConfig:
    return RunConfig(
        timeout=int(ragas_cfg.get("timeout", 120)),
        max_workers=int(ragas_cfg.get("max_workers", 8)),
    )


def outputs_to_dataset(outputs: list[RAGOutput]) -> Dataset:
    return Dataset.from_dict(
        {
            "question": [o.question for o in outputs],
            "answer": [o.answer for o in outputs],
            "contexts": [o.contexts for o in outputs],
            "ground_truth": [o.ground_truth for o in outputs],
        }
    )


def run_ragas_evaluation(
    outputs: list[RAGOutput],
    llm: ChatLLM,
    embedding_model: str,
    metric_names: list[str],
    ragas_cfg: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    ragas_cfg = ragas_cfg or {}
    run_config = build_run_config(ragas_cfg)
    dataset = outputs_to_dataset(outputs)
    metrics = [METRIC_MAP[m] for m in metric_names if m in METRIC_MAP]

    judge_llm = LangchainLLMWrapper(llm.load(), run_config=run_config)
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=embedding_model)
    )

    print(
        f"  RAGAS run_config: timeout={run_config.timeout}s, "
        f"max_workers={run_config.max_workers}"
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        run_config=run_config,
    )
    return {
        k: float(v) if v is not None else None
        for k, v in result._repr_dict.items()
    }
