"""NullMemory: a no-op memory implementation for pipeline validation."""

from __future__ import annotations

from himga.data.schema import Message, Session
from himga.memory.base import BaseMemory


class NullMemory(BaseMemory):
    """Memory implementation that stores nothing and returns empty context.

    Useful as a baseline (LLM answers from parametric knowledge only)
    and for wiring up the eval pipeline before real memory systems exist.
    """

    def ingest(self, message: Message, session: Session) -> None:
        pass

    def retrieve(self, query: str) -> str:
        return ""

    def reset(self) -> None:
        pass
