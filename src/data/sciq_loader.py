from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class SciQSample:
    question: str
    ground_truth: str
    support: str
    distractor1: str
    distractor2: str
    distractor3: str
    correct_answer: str


@dataclass
class SciQBenchmark:
    samples: list[SciQSample]
    corpus: list[str]
    corpus_ids: list[str]

    def __len__(self) -> int:
        return len(self.samples)


def load_sciq_benchmark(
    split: str = "validation",
    max_samples: int | None = 50,
    corpus_from_support: bool = True,
) -> SciQBenchmark:
    """Load SciQ and build evaluation set + retrieval corpus."""
    ds = load_dataset("allenai/sciq", split=split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples: list[SciQSample] = []
    for row in ds:
        samples.append(
            SciQSample(
                question=row["question"],
                ground_truth=row["correct_answer"],
                support=row["support"],
                distractor1=row["distractor1"],
                distractor2=row["distractor2"],
                distractor3=row["distractor3"],
                correct_answer=row["correct_answer"],
            )
        )

    if corpus_from_support:
        seen: set[str] = set()
        corpus: list[str] = []
        corpus_ids: list[str] = []
        for i, row in enumerate(ds):
            text = row["support"].strip()
            if text and text not in seen:
                seen.add(text)
                corpus.append(text)
                corpus_ids.append(f"support_{i}")
        for sample in samples:
            text = sample.support.strip()
            if text and text not in seen:
                seen.add(text)
                corpus.append(text)
                corpus_ids.append(f"extra_{len(corpus)}")
    else:
        corpus = [s.support for s in samples]
        corpus_ids = [f"doc_{i}" for i in range(len(corpus))]

    return SciQBenchmark(samples=samples, corpus=corpus, corpus_ids=corpus_ids)
