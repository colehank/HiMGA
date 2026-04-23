"""Tests for himga.llm: BaseLLMClient contract, MockLLMClient, and get_client()."""

import pytest

from himga.llm import BaseLLMClient, get_client

# ---------------------------------------------------------------------------
# MockLLMClient helper
# ---------------------------------------------------------------------------


class MockLLMClient(BaseLLMClient):
    """Returns a fixed response string for all chat calls."""

    def __init__(self, response: str = "mock response"):
        self._response = response
        self.call_count = 0
        self.last_messages: list[dict] | None = None
        self.last_kwargs: dict = {}

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        self.call_count += 1
        self.last_messages = messages
        self.last_kwargs = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
        return self._response


# ---------------------------------------------------------------------------
# TestBaseLLMClientInterface
# ---------------------------------------------------------------------------


class TestBaseLLMClientInterface:
    """BaseLLMClient cannot be instantiated; subclasses must implement chat()."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseLLMClient()  # type: ignore[abstract]

    def test_subclass_missing_chat_raises(self):
        class Incomplete(BaseLLMClient):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_full_subclass_instantiates_successfully(self):
        assert isinstance(MockLLMClient(), BaseLLMClient)


# ---------------------------------------------------------------------------
# TestMockLLMClient
# ---------------------------------------------------------------------------


class TestMockLLMClient:
    """MockLLMClient behaves correctly for use in other tests."""

    def test_returns_configured_response(self):
        client = MockLLMClient(response="hello")
        result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "hello"

    def test_tracks_call_count(self):
        client = MockLLMClient()
        client.chat([{"role": "user", "content": "a"}])
        client.chat([{"role": "user", "content": "b"}])
        assert client.call_count == 2

    def test_records_last_messages(self):
        client = MockLLMClient()
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        client.chat(msgs)
        assert client.last_messages == msgs

    def test_records_kwargs_defaults(self):
        client = MockLLMClient()
        client.chat([{"role": "user", "content": "x"}])
        assert client.last_kwargs["model"] is None
        assert client.last_kwargs["max_tokens"] == 1024
        assert client.last_kwargs["temperature"] == 0.0

    def test_records_kwargs_overrides(self):
        client = MockLLMClient()
        client.chat(
            [{"role": "user", "content": "x"}],
            model="claude-haiku-4-5",
            max_tokens=256,
            temperature=0.5,
        )
        assert client.last_kwargs["model"] == "claude-haiku-4-5"
        assert client.last_kwargs["max_tokens"] == 256
        assert client.last_kwargs["temperature"] == 0.5

    def test_returns_string(self):
        client = MockLLMClient(response="answer")
        result = client.chat([{"role": "user", "content": "q"}])
        assert isinstance(result, str)

    def test_empty_messages_does_not_raise(self):
        client = MockLLMClient()
        result = client.chat([])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestGetClient
# ---------------------------------------------------------------------------


class TestGetClient:
    """get_client() factory returns the correct client type."""

    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_client("nonexistent_provider")

    def test_explicit_anthropic_returns_anthropic_client(self, monkeypatch):
        from unittest.mock import patch

        from himga.llm.client import AnthropicClient

        with patch("anthropic.Anthropic"):
            client = get_client("anthropic")
        assert isinstance(client, AnthropicClient)

    def test_env_var_anthropic_returns_anthropic_client(self, monkeypatch):
        from unittest.mock import patch

        from himga.llm.client import AnthropicClient

        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        with patch("anthropic.Anthropic"):
            client = get_client()
        assert isinstance(client, AnthropicClient)

    def test_env_var_unknown_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        with pytest.raises(ValueError, match="Unknown provider"):
            get_client()

    def test_default_provider_is_anthropic(self, monkeypatch):
        from unittest.mock import patch

        from himga.llm.client import AnthropicClient

        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        with patch("anthropic.Anthropic"):
            client = get_client()
        assert isinstance(client, AnthropicClient)

    def test_explicit_provider_overrides_env_var(self, monkeypatch):
        from unittest.mock import patch

        monkeypatch.setenv("LLM_PROVIDER", "nonexistent_provider")
        with pytest.raises(ValueError):
            get_client()  # env var wins when no explicit arg
        # but explicit arg overrides env
        from himga.llm.client import AnthropicClient

        with patch("anthropic.Anthropic"):
            client = get_client("anthropic")
        assert isinstance(client, AnthropicClient)


# ---------------------------------------------------------------------------
# TestAnthropicClientUnit  (no real API calls)
# ---------------------------------------------------------------------------


class TestAnthropicClientUnit:
    """Unit tests for AnthropicClient that do not make real API calls."""

    def test_is_base_llm_client_instance(self):
        from unittest.mock import patch

        from himga.llm.client import AnthropicClient

        with patch("anthropic.Anthropic"):
            client = AnthropicClient()
        assert isinstance(client, BaseLLMClient)

    def test_default_model_is_sonnet(self):
        from unittest.mock import patch

        from himga.llm.client import AnthropicClient

        with patch("anthropic.Anthropic"):
            client = AnthropicClient()
        assert client._default_model == "claude-sonnet-4-6"

    def test_custom_model_is_stored(self):
        from unittest.mock import patch

        from himga.llm.client import AnthropicClient

        with patch("anthropic.Anthropic"):
            client = AnthropicClient(model="claude-haiku-4-5")
        assert client._default_model == "claude-haiku-4-5"

    def test_has_chat_method(self):
        from himga.llm.client import AnthropicClient

        assert callable(getattr(AnthropicClient, "chat", None))


# ---------------------------------------------------------------------------
# TestAnthropicClientIntegration  (real API, skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAnthropicClientIntegration:
    """Real Anthropic API calls — skipped unless HIMGA_INTEGRATION_TESTS=1."""

    def test_chat_returns_nonempty_string(self):
        from himga.llm.client import AnthropicClient

        client = AnthropicClient(model="claude-haiku-4-5-20251001")
        result = client.chat(
            [{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=10,
        )
        assert isinstance(result, str)
        assert len(result) > 0
