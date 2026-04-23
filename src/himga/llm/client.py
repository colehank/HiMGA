"""LLM client abstractions and implementations."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Unified interface for LLM API providers.

    Upper layers (agent, eval/judge) depend only on this interface,
    keeping them decoupled from any specific provider.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Send a chat request and return the assistant reply text.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format message list: ``[{"role": ..., "content": ...}, ...]``.
            A leading ``{"role": "system", ...}`` entry is handled by each
            provider implementation.
        model : str or None
            Override the client's default model for this call.
        max_tokens : int
            Maximum tokens in the completion.
        temperature : float
            Sampling temperature. Defaults to ``0.0`` for deterministic eval.

        Returns
        -------
        str
            Assistant reply text, stripped of any surrounding whitespace.
        """

    def batch_chat(self, requests: list[dict]) -> list[str]:
        """Send multiple chat requests and return all replies in order.

        The default implementation calls :meth:`chat` sequentially.
        Override in subclasses to enable provider-native concurrency (e.g.
        async I/O for remote APIs) without imposing that complexity on local
        or simple implementations.

        Parameters
        ----------
        requests : list[dict]
            Each dict has a ``"messages"`` key plus optional ``"model"``,
            ``"max_tokens"``, and ``"temperature"`` keys matching :meth:`chat`.

        Returns
        -------
        list[str]
            Reply strings in the same order as *requests*.
        """
        return [
            self.chat(req["messages"], **{k: v for k, v in req.items() if k != "messages"})
            for req in requests
        ]


class AnthropicClient(BaseLLMClient):
    """LLM client backed by the Anthropic Messages API.

    Parameters
    ----------
    model : str
        Default model ID to use when ``chat()`` is called without an override.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        import anthropic  # deferred import — not required if using a different provider

        self._client = anthropic.Anthropic()
        self._default_model = model

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        system, filtered = self._split_system(messages)
        kwargs: dict = dict(
            model=model or self._default_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=filtered,
        )
        if system is not None:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return self._extract_text(resp.content)

    def batch_chat(self, requests: list[dict]) -> list[str]:
        """Send all requests concurrently via the async Anthropic client."""
        import asyncio

        return asyncio.run(self._abatch_async(requests))

    async def _abatch_async(self, requests: list[dict]) -> list[str]:
        import asyncio

        import anthropic

        async_client = anthropic.AsyncAnthropic()

        async def _call(req: dict) -> str:
            system, filtered = self._split_system(req["messages"])
            kwargs: dict = dict(
                model=req.get("model") or self._default_model,
                max_tokens=req.get("max_tokens", 1024),
                temperature=req.get("temperature", 0.0),
                messages=filtered,
            )
            if system is not None:
                kwargs["system"] = system
            resp = await async_client.messages.create(**kwargs)
            return self._extract_text(resp.content)

        return list(await asyncio.gather(*(_call(r) for r in requests)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Separate the optional system message from the rest."""
        system: str | None = None
        filtered: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg["content"]
            else:
                filtered.append(msg)
        return system, filtered

    @staticmethod
    def _extract_text(content: list) -> str:
        """Return the text of the first TextBlock in *content*."""
        for block in content:
            if hasattr(block, "text"):
                return block.text
        raise ValueError(f"No text block in response: {content}")


def get_client(provider: str | None = None) -> BaseLLMClient:
    """Return an LLM client for the requested provider.

    Parameters
    ----------
    provider : str or None
        Provider name (``"anthropic"``).  Falls back to the ``LLM_PROVIDER``
        environment variable, then defaults to ``"anthropic"``.

    Returns
    -------
    BaseLLMClient

    Raises
    ------
    ValueError
        If the resolved provider name is not supported.
    """
    p = provider or os.getenv("LLM_PROVIDER", "anthropic")
    if p == "anthropic":
        return AnthropicClient()
    raise ValueError(f"Unknown provider: {p!r}")
