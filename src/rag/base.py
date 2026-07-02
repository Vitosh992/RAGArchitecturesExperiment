from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.llm.protocol import ChatLLM


@dataclass
class RAGOutput:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    pipeline: str


ANSWER_SYSTEM = (
    "You are a scientific assistant. Answer the question using ONLY the provided "
    "context. Be concise and factual. If the context does not contain the answer, "
    "say you cannot determine it from the context."
)


class BaseRAGPipeline(ABC):
    name: str

    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    @abstractmethod
    def retrieve(self, question: str) -> list[str]:
        ...

    def generate(self, question: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        user = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
        return self.llm.chat(ANSWER_SYSTEM, user)

    def run(self, question: str, ground_truth: str) -> RAGOutput:
        contexts = self.retrieve(question)
        answer = self.generate(question, contexts)
        return RAGOutput(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            pipeline=self.name,
        )
