from __future__ import annotations

from src.config import Config
from src.llm.client import OpenRouterLLM


def build_llm(config: Config, role: str = "generator") -> OpenRouterLLM:
    or_cfg = config.openrouter
    model_key = "judge_model" if role == "judge" else "model"
    return OpenRouterLLM(
        model=or_cfg[model_key],
        api_key_env=or_cfg.get("api_key_env", "OPENROUTER_API_KEY"),
        max_tokens=or_cfg.get("max_tokens", 256),
        temperature=or_cfg.get("temperature", 0.1),
        site_url=or_cfg.get("site_url", "http://localhost"),
        site_name=or_cfg.get("site_name", "SciQ RAG Benchmark"),
    )


def build_llm_pair(config: Config) -> tuple[OpenRouterLLM, OpenRouterLLM]:
    generator = build_llm(config, role="generator")
    judge_model = config.openrouter.get("judge_model")
    generator_model = config.openrouter.get("model")
    judge = (
        build_llm(config, role="judge")
        if judge_model and judge_model != generator_model
        else generator
    )
    return generator, judge
