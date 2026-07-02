from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.config import Config
from src.data.sciq_loader import load_sciq_benchmark
from src.evaluation.ragas_eval import run_ragas_evaluation
from src.indexing.hybrid_retriever import HybridRetriever
from src.llm.factory import build_llm_pair
from src.llm.protocol import ChatLLM
from src.rag.advanced_rag import AdvancedRAGPipeline
from src.rag.base import BaseRAGPipeline, RAGOutput
from src.rag.graph_rag import GraphRAGPipeline
from src.rag.naive_rag import NaiveRAGPipeline


def build_pipelines(
    config: Config,
    llm: ChatLLM,
    retriever: HybridRetriever,
) -> dict[str, BaseRAGPipeline]:
    top_k = config.retrieval["top_k"]
    pipelines: dict[str, BaseRAGPipeline] = {}

    if "naive" in config.pipelines:
        pipelines["naive"] = NaiveRAGPipeline(
            llm=llm,
            retriever=retriever,
            top_k=top_k,
            bm25_candidates=config.retrieval["bm25_candidates"],
            dense_candidates=config.retrieval["dense_candidates"],
        )
    if "advanced" in config.pipelines:
        pipelines["advanced"] = AdvancedRAGPipeline(
            llm=llm,
            retriever=retriever,
            reranker_model=config.models["reranker"],
            top_k=top_k,
        )
    if "graph" in config.pipelines:
        pipelines["graph"] = GraphRAGPipeline(
            llm=llm,
            retriever=retriever,
            top_k=top_k,
            min_cluster_size=config.graph_rag["min_cluster_size"],
            resolution=config.graph_rag["resolution"],
        )
    return pipelines


def run_pipeline(
    pipeline: BaseRAGPipeline,
    benchmark,
) -> list[RAGOutput]:
    outputs: list[RAGOutput] = []
    for sample in tqdm(benchmark.samples, desc=pipeline.name):
        outputs.append(pipeline.run(sample.question, sample.ground_truth))
    return outputs


def save_results(
    all_metrics: dict[str, dict],
    all_outputs: dict[str, list[RAGOutput]],
    output_dir: Path,
    config: Config,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_rows = []
    for name, metrics in all_metrics.items():
        row = {"pipeline": name}
        row.update(metrics)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"ragas_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)

    meta = {
        "timestamp": timestamp,
        "experiment": config.experiment.get("name"),
        "openrouter_model": config.openrouter.get("model"),
        "openrouter_judge": config.openrouter.get("judge_model"),
        "dataset_split": config.dataset.get("split"),
        "max_samples": config.dataset.get("max_samples"),
        "metrics": all_metrics,
    }
    metrics_json = output_dir / f"ragas_metrics_{timestamp}.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    for name, outputs in all_outputs.items():
        records = [
            {
                "question": o.question,
                "answer": o.answer,
                "ground_truth": o.ground_truth,
                "contexts": o.contexts,
            }
            for o in outputs
        ]
        pd.DataFrame(records).to_json(
            output_dir / f"predictions_{name}_{timestamp}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )

    plot_comparison(summary_df, output_dir / f"comparison_{timestamp}.png")
    print(f"\nResults saved to {output_dir}")


def plot_comparison(summary_df: pd.DataFrame, path: Path) -> None:
    metric_cols = [c for c in summary_df.columns if c != "pipeline"]
    if not metric_cols:
        return

    melted = summary_df.melt(
        id_vars="pipeline",
        value_vars=metric_cols,
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="metric", y="score", hue="pipeline")
    plt.title("RAGAS Metrics Comparison (SciQ)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_benchmark(config_path: str | Path) -> dict[str, dict]:
    config = Config.load(config_path)
    output_dir = config.output_dir

    llm, judge_llm = build_llm_pair(config)
    print(f"Generator: {config.openrouter['model']}")
    print(f"RAGAS judge: {config.openrouter.get('judge_model', config.openrouter['model'])}")

    print("Loading SciQ dataset...")
    benchmark = load_sciq_benchmark(
        split=config.dataset["split"],
        max_samples=config.dataset.get("max_samples"),
        corpus_from_support=config.dataset.get("corpus_from_support", True),
    )
    print(f"  Samples: {len(benchmark.samples)}, Corpus size: {len(benchmark.corpus)}")

    print("Building hybrid retriever (BM25 + BGE-M3)...")
    retriever = HybridRetriever(
        corpus=benchmark.corpus,
        embedding_model=config.models["embedding"],
    )

    pipelines = build_pipelines(config, llm, retriever)
    all_metrics: dict[str, dict] = {}
    all_outputs: dict[str, list[RAGOutput]] = {}

    for name, pipeline in pipelines.items():
        print(f"\n=== Running {name.upper()} RAG ===")
        outputs = run_pipeline(pipeline, benchmark)
        all_outputs[name] = outputs

        print(f"Evaluating {name} with RAGAS...")
        metrics = run_ragas_evaluation(
            outputs=outputs,
            llm=judge_llm,
            embedding_model=config.models["embedding"],
            metric_names=config.ragas["metrics"],
            ragas_cfg=config.ragas,
        )
        all_metrics[name] = metrics
        print(f"  {metrics}")

    save_results(all_metrics, all_outputs, output_dir, config)
    return all_metrics
