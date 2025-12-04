"""
Unit tests for LiteLLMProvider.

Tests the LiteLLM provider implementation with mocked LiteLLM calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Skip all tests if litellm is not installed
pytest.importorskip("litellm")

from kosmos.core.providers.base import LLMResponse, Message
from kosmos.core.providers.litellm_provider import LiteLLMProvider


class TestLiteLLMProviderInit:
    """Test LiteLLMProvider initialization."""

    def test_init_with_model_only(self):
        """Test initialization with just a model name."""
        provider = LiteLLMProvider({"model": "gpt-3.5-turbo"})

        assert provider.model == "gpt-3.5-turbo"
        assert provider.provider_type == "openai"
        assert provider.max_tokens_default == 4096
        assert provider.temperature_default == 0.7

    def test_init_with_ollama_model(self):
        """Test initialization with Ollama model."""
        provider = LiteLLMProvider(
            {"model": "ollama/llama3.1:8b", "api_base": "http://localhost:11434"}
        )

        assert provider.model == "ollama/llama3.1:8b"
        assert provider.provider_type == "ollama"
        assert provider.api_base == "http://localhost:11434"

    def test_init_with_deepseek_model(self):
        """Test initialization with DeepSeek model."""
        provider = LiteLLMProvider({"model": "deepseek/deepseek-chat", "api_key": "sk-test-key"})

        assert provider.model == "deepseek/deepseek-chat"
        assert provider.provider_type == "deepseek"

    def test_init_with_anthropic_model(self):
        """Test initialization with Anthropic model."""
        provider = LiteLLMProvider(
            {"model": "claude-3-5-sonnet-20241022", "api_key": "sk-ant-test"}
        )

        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.provider_type == "anthropic"

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        provider = LiteLLMProvider(
            {"model": "gpt-4-turbo", "max_tokens": 8192, "temperature": 0.5, "timeout": 60}
        )

        assert provider.max_tokens_default == 8192
        assert provider.temperature_default == 0.5
        assert provider.timeout == 60


class TestLiteLLMProviderGenerate:
    """Test LiteLLMProvider generation methods."""

    @pytest.fixture
    def mock_provider(self):
        """Create a provider with mocked LiteLLM."""
        with patch(
            "kosmos.core.providers.litellm_provider.LiteLLMProvider.__init__", return_value=None
        ):
            provider = LiteLLMProvider.__new__(LiteLLMProvider)
            provider.config = {"model": "gpt-3.5-turbo"}
            provider.provider_name = "litellm"
            provider.model = "gpt-3.5-turbo"
            provider.api_key = None
            provider.api_base = None
            provider.max_tokens_default = 4096
            provider.temperature_default = 0.7
            provider.timeout = 120
            provider.provider_type = "openai"

            # Usage tracking
            provider.total_input_tokens = 0
            provider.total_output_tokens = 0
            provider.total_cost_usd = 0.0
            provider.request_count = 0

            # Mock litellm module
            provider.litellm = MagicMock()

            yield provider

    def test_generate_success(self, mock_provider):
        """Test successful generation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Test response"), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_response.model = "gpt-3.5-turbo"

        mock_provider.litellm.completion.return_value = mock_response

        response = mock_provider.generate("Test prompt", system="Test system")

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.finish_reason == "stop"

    def test_generate_with_options(self, mock_provider):
        """Test generation with custom options."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Response"), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        mock_response.model = "gpt-3.5-turbo"

        mock_provider.litellm.completion.return_value = mock_response

        mock_provider.generate("Prompt", max_tokens=100, temperature=0.2, stop_sequences=["END"])

        # Verify completion was called with correct args
        call_kwargs = mock_provider.litellm.completion.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["stop"] == ["END"]


class TestLiteLLMProviderStructured:
    """Test structured output generation."""

    @pytest.fixture
    def mock_provider(self):
        """Create a provider with mocked LiteLLM."""
        with patch(
            "kosmos.core.providers.litellm_provider.LiteLLMProvider.__init__", return_value=None
        ):
            provider = LiteLLMProvider.__new__(LiteLLMProvider)
            provider.config = {"model": "gpt-3.5-turbo"}
            provider.provider_name = "litellm"
            provider.model = "gpt-3.5-turbo"
            provider.api_key = None
            provider.api_base = None
            provider.max_tokens_default = 4096
            provider.temperature_default = 0.7
            provider.timeout = 120
            provider.provider_type = "openai"

            provider.total_input_tokens = 0
            provider.total_output_tokens = 0
            provider.total_cost_usd = 0.0
            provider.request_count = 0

            provider.litellm = MagicMock()

            yield provider

    def test_generate_structured_success(self, mock_provider):
        """Test structured JSON generation."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content='{"name": "test", "value": 42}'), finish_reason="stop"
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        mock_response.model = "gpt-3.5-turbo"

        mock_provider.litellm.completion.return_value = mock_response

        result = mock_provider.generate_structured(
            "Generate JSON", schema={"name": "string", "value": "int"}
        )

        assert result == {"name": "test", "value": 42}

    def test_generate_structured_with_markdown(self, mock_provider):
        """Test structured output with markdown code block."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content='```json\n{"key": "value"}\n```'), finish_reason="stop"
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        mock_response.model = "gpt-3.5-turbo"

        mock_provider.litellm.completion.return_value = mock_response

        result = mock_provider.generate_structured("Generate JSON", schema={"key": "string"})

        assert result == {"key": "value"}


class TestLiteLLMProviderMessages:
    """Test message-based generation."""

    @pytest.fixture
    def mock_provider(self):
        """Create a provider with mocked LiteLLM."""
        with patch(
            "kosmos.core.providers.litellm_provider.LiteLLMProvider.__init__", return_value=None
        ):
            provider = LiteLLMProvider.__new__(LiteLLMProvider)
            provider.config = {"model": "gpt-3.5-turbo"}
            provider.provider_name = "litellm"
            provider.model = "gpt-3.5-turbo"
            provider.api_key = None
            provider.api_base = None
            provider.max_tokens_default = 4096
            provider.temperature_default = 0.7
            provider.timeout = 120
            provider.provider_type = "openai"

            provider.total_input_tokens = 0
            provider.total_output_tokens = 0
            provider.total_cost_usd = 0.0
            provider.request_count = 0

            provider.litellm = MagicMock()

            yield provider

    def test_generate_with_messages(self, mock_provider):
        """Test generation with message history."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Reply to conversation"), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        mock_response.model = "gpt-3.5-turbo"

        mock_provider.litellm.completion.return_value = mock_response

        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]

        response = mock_provider.generate_with_messages(messages)

        assert response.content == "Reply to conversation"

        # Verify messages were converted correctly
        call_kwargs = mock_provider.litellm.completion.call_args.kwargs
        assert len(call_kwargs["messages"]) == 4


class TestLiteLLMProviderModelInfo:
    """Test model info retrieval."""

    def test_get_model_info_openai(self):
        """Test model info for OpenAI model."""
        provider = LiteLLMProvider({"model": "gpt-4-turbo"})
        info = provider.get_model_info()

        assert info["name"] == "gpt-4-turbo"
        assert info["provider"] == "litellm/openai"
        assert info["supports_streaming"] is True
        assert info["supports_async"] is True

    def test_get_model_info_ollama(self):
        """Test model info for Ollama model (free)."""
        provider = LiteLLMProvider(
            {"model": "ollama/llama3.1:8b", "api_base": "http://localhost:11434"}
        )
        info = provider.get_model_info()

        assert info["name"] == "ollama/llama3.1:8b"
        assert info["provider"] == "litellm/ollama"
        assert info["cost_per_input_token"] == 0.0
        assert info["cost_per_output_token"] == 0.0


class TestLiteLLMProviderCostEstimation:
    """Test cost estimation."""

    def test_cost_estimation_openai(self):
        """Test cost estimation for OpenAI models."""
        provider = LiteLLMProvider({"model": "gpt-4-turbo"})

        # gpt-4-turbo: $10/M input, $30/M output
        cost = provider._estimate_cost(1000, 500)
        expected = (1000 / 1_000_000) * 10.0 + (500 / 1_000_000) * 30.0
        assert abs(cost - expected) < 0.0001

    def test_cost_estimation_free_model(self):
        """Test cost estimation for free models (Ollama)."""
        provider = LiteLLMProvider(
            {"model": "ollama/llama3.1:8b", "api_base": "http://localhost:11434"}
        )

        cost = provider._estimate_cost(10000, 5000)
        assert cost == 0.0  # Ollama is free

    def test_cost_estimation_unknown_model(self):
        """Test cost estimation for unknown models."""
        provider = LiteLLMProvider({"model": "unknown/model"})

        cost = provider._estimate_cost(1000, 500)
        assert cost == 0.0  # Unknown models default to free


class TestLiteLLMProviderUsageTracking:
    """Test usage statistics tracking."""

    @pytest.fixture
    def mock_provider(self):
        """Create a provider with mocked LiteLLM."""
        with patch(
            "kosmos.core.providers.litellm_provider.LiteLLMProvider.__init__", return_value=None
        ):
            provider = LiteLLMProvider.__new__(LiteLLMProvider)
            provider.config = {"model": "gpt-3.5-turbo"}
            provider.provider_name = "litellm"
            provider.model = "gpt-3.5-turbo"
            provider.api_key = None
            provider.api_base = None
            provider.max_tokens_default = 4096
            provider.temperature_default = 0.7
            provider.timeout = 120
            provider.provider_type = "openai"

            provider.total_input_tokens = 0
            provider.total_output_tokens = 0
            provider.total_cost_usd = 0.0
            provider.request_count = 0

            provider.litellm = MagicMock()

            yield provider

    def test_usage_tracking(self, mock_provider):
        """Test that usage is tracked across calls."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Response"), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        mock_response.model = "gpt-3.5-turbo"

        mock_provider.litellm.completion.return_value = mock_response

        # Make two calls
        mock_provider.generate("Test 1")
        mock_provider.generate("Test 2")

        stats = mock_provider.get_usage_stats()
        assert stats["total_requests"] == 2
        assert stats["total_input_tokens"] == 200
        assert stats["total_output_tokens"] == 100


@pytest.mark.asyncio
class TestLiteLLMProviderAsync:
    """Test async generation methods."""

    @pytest.fixture
    def mock_provider(self):
        """Create a provider with mocked async LiteLLM."""
        with patch(
            "kosmos.core.providers.litellm_provider.LiteLLMProvider.__init__", return_value=None
        ):
            provider = LiteLLMProvider.__new__(LiteLLMProvider)
            provider.config = {"model": "gpt-3.5-turbo"}
            provider.provider_name = "litellm"
            provider.model = "gpt-3.5-turbo"
            provider.api_key = None
            provider.api_base = None
            provider.max_tokens_default = 4096
            provider.temperature_default = 0.7
            provider.timeout = 120
            provider.provider_type = "openai"

            provider.total_input_tokens = 0
            provider.total_output_tokens = 0
            provider.total_cost_usd = 0.0
            provider.request_count = 0

            provider.litellm = MagicMock()

            yield provider

    async def test_generate_async_success(self, mock_provider):
        """Test async generation."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Async response"), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_response.model = "gpt-3.5-turbo"

        # Use AsyncMock for acompletion
        mock_provider.litellm.acompletion = AsyncMock(return_value=mock_response)

        response = await mock_provider.generate_async("Test prompt")

        assert response.content == "Async response"
        mock_provider.litellm.acompletion.assert_called_once()


class TestProviderRegistration:
    """Test provider registration in factory."""

    def test_litellm_registered(self):
        """Test that LiteLLM provider is registered."""
        from kosmos.core.providers import list_providers

        providers = list_providers()
        assert "litellm" in providers

    def test_aliases_registered(self):
        """Test that provider aliases are registered."""
        from kosmos.core.providers import list_providers

        providers = list_providers()
        assert "ollama" in providers
        assert "deepseek" in providers
        assert "lmstudio" in providers

    def test_get_provider_litellm(self):
        """Test getting LiteLLM provider by name."""
        from kosmos.core.providers import get_provider

        provider = get_provider("litellm", {"model": "gpt-3.5-turbo"})
        assert isinstance(provider, LiteLLMProvider)
