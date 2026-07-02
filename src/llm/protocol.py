from __future__ import annotations

from typing import Any, Protocol


class ChatLLM(Protocol):
    def chat(self, system: str, user: str) -> str: ...

    def load(self) -> Any: ...
