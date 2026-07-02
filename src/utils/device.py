from __future__ import annotations

import torch


def resolve_device() -> str:
    """Use GPU for embeddings when available, otherwise CPU."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"
