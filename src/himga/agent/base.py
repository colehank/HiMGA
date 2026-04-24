"""BaseAgent: combines a memory system and an LLM into an evaluatable agent."""

from __future__ import annotations

from himga.data.schema import Sample
from himga.llm.client import BaseLLMClient
from himga.memory.base import BaseMemory


class BaseAgent:
    """Evaluatable agent that ingests conversation history and answers questions.

    Parameters
    ----------
    memory : BaseMemory
        Memory system used to store and retrieve conversation history.
    llm : BaseLLMClient
        LLM client used to generate answers.
    """

    def __init__(self, memory: BaseMemory, llm: BaseLLMClient) -> None:
        self.memory = memory
        self.llm = llm

    def ingest_sample(self, sample: Sample) -> None:
        """Ingest all sessions of *sample* into the memory system.

        Parameters
        ----------
        sample : Sample
            Evaluation unit whose sessions are written to memory in order.
        """
        for session in sample.sessions:
            for message in session.messages:
                self.memory.ingest(message)

    def answer(self, question: str) -> str:
        """Retrieve context and generate an answer for *question*.

        Parameters
        ----------
        question : str
            Natural language question to answer.

        Returns
        -------
        str
            LLM-generated answer text.
        """
        context = self.memory.retrieve(question)
        messages = self._build_messages(question, context)
        return self.llm.chat(messages)

    def _build_messages(self, question: str, context: str) -> list[dict]:
        """Construct the OpenAI-format message list for the LLM call.

        Parameters
        ----------
        question : str
            The question to answer.
        context : str
            Retrieved memory context.  An empty string means no context is available
            and the prompt will not include a "Context:" section.

        Returns
        -------
        list[dict]
            Message list with a system turn followed by a user turn.
        """
        system = (
            "You are a helpful assistant with access to past conversation history. "
            "Answer the question based on the provided context."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}" if context else question
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
