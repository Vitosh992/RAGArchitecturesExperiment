from __future__ import annotations

import os
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


@dataclass
class OpenRouterLLM:
    """LLM client for OpenRouter (OpenAI-compatible API)."""

    model: str
    api_key: str | None = None
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 256
    temperature: float = 0.1
    site_url: str = "http://localhost"
    site_name: str = "SciQ RAG Benchmark"
    _client: ChatOpenAI | None = field(default=None, repr=False)

    def _resolve_api_key(self) -> str:
        key = self.api_key or os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"OpenRouter API key not found. Set {self.api_key_env} in .env "
                f"or export it as an environment variable."
            )
        return key

    def load(self) -> ChatOpenAI:
        if self._client is None:
            self._client = ChatOpenAI(
                model=self.model,
                openai_api_key=self._resolve_api_key(),
                openai_api_base=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                default_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
            )
        return self._client

    def chat(self, system: str, user: str) -> str:
        response = self.load().invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        content = response.content
        return content.strip() if isinstance(content, str) else str(content).strip()
