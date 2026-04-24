"""LLM client abstractions and implementations."""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv(override=True)


def _run_async(coro):
    """Run *coro* whether or not there is already a running event loop (e.g. Jupyter)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        # Running inside Jupyter / IPython — use nest_asyncio or a new thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


async def _call_with_backoff(fn, max_retries: int = 5, base_delay: float = 1.0):
    """Call async *fn* with exponential backoff on rate-limit (429) errors."""
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as exc:
            # Catch rate-limit errors from both openai and anthropic SDKs.
            is_rate_limit = (
                getattr(exc, "status_code", None) == 429
                or "rate" in str(type(exc).__name__).lower()
                or "ratelimit" in str(type(exc).__name__).lower().replace("_", "")
            )
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2  # exponential backoff


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

    def __init__(self, model: str = "claude-sonnet-4-6", batch_size: int = 5) -> None:
        import anthropic  # deferred import — not required if using a different provider

        self._client = anthropic.Anthropic()
        self._default_model = model
        self._batch_size = batch_size

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
        return _run_async(self._abatch_async(requests))

    async def _abatch_async(self, requests: list[dict]) -> list[str]:
        import anthropic

        async_client = anthropic.AsyncAnthropic()

        async def _call(req: dict) -> str:
            async with sem:
                system, filtered = self._split_system(req["messages"])
                kwargs: dict = dict(
                    model=req.get("model") or self._default_model,
                    max_tokens=req.get("max_tokens", 1024),
                    temperature=req.get("temperature", 0.0),
                    messages=filtered,
                )
                if system is not None:
                    kwargs["system"] = system

                async def _do():
                    return await async_client.messages.create(**kwargs)

                resp = await _call_with_backoff(_do)
                return self._extract_text(resp.content)

        sem = asyncio.Semaphore(self._batch_size)
        return list(await asyncio.gather(*(_call(r) for r in requests)))

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


class OpenAIClient(BaseLLMClient):
    """LLM client backed by the OpenAI Chat Completions API.

    Uses ``openai.AsyncOpenAI`` for concurrent batch requests and the sync
    ``openai.OpenAI`` client for single calls.  Compatible with any
    OpenAI-spec endpoint (OpenAI, Azure OpenAI, local vLLM, Ollama, etc.)
    via the ``base_url`` parameter.

    Parameters
    ----------
    model : str
        Default model ID (e.g. ``"gpt-4o-mini"``).
    base_url : str or None
        Override the API base URL.  Useful for local servers or Azure.
        Reads ``OPENAI_BASE_URL`` env var when ``None``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        batch_size: int = 5,
    ) -> None:
        import openai

        kwargs: dict = {}
        resolved_base = base_url or os.getenv("OPENAI_BASE_URL")
        if resolved_base:
            kwargs["base_url"] = resolved_base
        self._client = openai.OpenAI(**kwargs)
        self._default_model = model
        self._base_url = resolved_base
        self._batch_size = batch_size

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=model or self._default_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self._extract_content(resp)

    @staticmethod
    def _extract_content(resp) -> str:
        """Extract text from various OpenAI-compatible response shapes.

        Some third-party proxies or openai v2 response objects differ from the
        standard ChatCompletion format.
        """
        if isinstance(resp, str):
            return resp
        if hasattr(resp, "choices"):
            return resp.choices[0].message.content or ""
        if hasattr(resp, "output_text"):
            return resp.output_text or ""
        if hasattr(resp, "output"):
            for item in resp.output:
                if hasattr(item, "content"):
                    for block in item.content:
                        if hasattr(block, "text"):
                            return block.text or ""
        return str(resp)

    def batch_chat(self, requests: list[dict]) -> list[str]:
        """Send all requests concurrently via the async OpenAI client."""
        return _run_async(self._abatch_async(requests))

    async def _abatch_async(self, requests: list[dict]) -> list[str]:
        import openai

        kwargs: dict = {}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        async_client = openai.AsyncOpenAI(**kwargs)

        async def _call(req: dict) -> str:
            async with sem:

                async def _do():
                    return await async_client.chat.completions.create(
                        model=req.get("model") or self._default_model,
                        messages=req["messages"],  # type: ignore[arg-type]
                        max_tokens=req.get("max_tokens", 1024),
                        temperature=req.get("temperature", 0.0),
                    )

                resp = await _call_with_backoff(_do)
                return self._extract_content(resp)

        sem = asyncio.Semaphore(self._batch_size)
        return list(await asyncio.gather(*(_call(r) for r in requests)))


def get_client(provider: str | None = None, **kwargs) -> BaseLLMClient:
    """Return an LLM client for the requested provider.

    Parameters
    ----------
    provider : str or None
        Provider name: ``"anthropic"`` or ``"openai"``.  Falls back to the
        ``LLM_PROVIDER`` environment variable, then defaults to
        ``"anthropic"``.
    **kwargs
        Forwarded to the provider client constructor (e.g. ``model=``,
        ``base_url=`` for OpenAI).

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
        return AnthropicClient(**kwargs)
    if p == "openai":
        return OpenAIClient(**kwargs)
    raise ValueError(f"Unknown provider: {p!r}")
