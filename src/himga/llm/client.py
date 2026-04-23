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
        # Anthropic separates the system prompt from the messages list.
        system = None
        filtered: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg["content"]
            else:
                filtered.append(msg)

        kwargs: dict = dict(
            model=model or self._default_model,
            max_tokens=max_tokens,
            messages=filtered,
        )
        if system is not None:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text


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
