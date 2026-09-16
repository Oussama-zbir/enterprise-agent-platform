"""Build the LLM client from settings.

The choice of backend is deployment configuration, not application logic, so it
lives here instead of inside the client or the adapter. Everything the platform
calls a model through comes out of ``build_llm_client``.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic, AsyncAnthropicBedrockMantle

from enterprise_agent_platform.config import Settings
from enterprise_agent_platform.llm.anthropic_provider import AnthropicProvider, AsyncAnthropicClient
from enterprise_agent_platform.llm.client import LLMClient
from enterprise_agent_platform.llm.provider import FakeLLMProvider, LLMProvider


def build_llm_client(settings: Settings) -> LLMClient:
    return LLMClient(_build_provider(settings), timeout_seconds=settings.llm_timeout_seconds)


def build_anthropic_client(settings: Settings) -> AsyncAnthropicClient:
    """Construct the SDK client for the configured backend.

    Two deliberate settings: ``max_retries=0`` keeps retry policy with the
    caller (see ``anthropic_provider``), and the SDK timeout mirrors the
    client's own deadline so a call that ``LLMClient`` abandons also has its
    socket torn down instead of lingering in the pool.
    """
    if settings.llm_provider == "bedrock":
        return AsyncAnthropicBedrockMantle(
            aws_region=settings.aws_region,
            max_retries=0,
            timeout=settings.llm_timeout_seconds,
        )
    api_key = settings.anthropic_api_key
    return AsyncAnthropic(
        api_key=api_key.get_secret_value() if api_key is not None else None,
        max_retries=0,
        timeout=settings.llm_timeout_seconds,
    )


def _build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "fake":
        # Offline placeholder: an unscripted call fails loudly rather than
        # silently returning something that looks like a model response.
        return FakeLLMProvider([])
    return AnthropicProvider(
        build_anthropic_client(settings),
        model=settings.llm_model,
        name=settings.llm_provider,
    )
