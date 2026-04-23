"""Abstract base class for all memory system implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from himga.data.schema import Message, Session


class BaseMemory(ABC):
    """Unified interface for conversational memory systems.

    All memory variants (NullMemory, MAGMA replay, HiMGA hierarchical)
    implement this interface so the evaluation layer remains decoupled
    from any specific implementation.
    """

    @abstractmethod
    def ingest(self, message: Message, session: Session) -> None:
        """Write one message into the memory system.

        Parameters
        ----------
        message : Message
            The utterance to store.
        session : Session
            Enclosing session; provides timestamp and session-level context.
        """

    @abstractmethod
    def retrieve(self, query: str) -> str:
        """Retrieve memory relevant to *query*.

        Parameters
        ----------
        query : str
            Natural language query used to surface relevant memories.

        Returns
        -------
        str
            Assembled context string ready to inject into a prompt.
            Returns ``""`` when no relevant memory exists.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memory.

        Called by the eval runner before processing each new :class:`~himga.data.schema.Sample`
        to guarantee isolation between evaluation units.
        """
