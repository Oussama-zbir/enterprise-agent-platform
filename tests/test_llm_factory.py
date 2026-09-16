"""Tests for LLM wiring: settings to backend, and the app-level lifecycle."""

from __future__ import annotations

from typing import Any

import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrockMantle
from pydantic import ValidationError

from enterprise_agent_platform.config import Settings
from enterprise_agent_platform.llm.factory import build_anthropic_client, build_llm_client
from enterprise_agent_platform.main import create_app


def settings(**overrides: Any) -> Settings:
    """Settings built from explicit values only, ignoring any local ``.env``."""
    # `_env_file` is a pydantic-settings runtime argument, absent from the
    # model's generated __init__ signature.
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_default_backend_is_the_offline_fake() -> None:
    assert build_llm_client(settings()).provider_name == "fake"


def test_anthropic_backend_uses_the_configured_key_and_no_sdk_retries() -> None:
    client = build_anthropic_client(
        settings(llm_provider="anthropic", anthropic_api_key="sk-test", llm_timeout_seconds=12.0)
    )

    assert isinstance(client, AsyncAnthropic)
    assert client.api_key == "sk-test"
    assert client.max_retries == 0
    assert client.timeout == 12.0


def test_anthropic_backend_defers_to_the_sdk_when_no_key_is_configured() -> None:
    client = build_anthropic_client(settings(llm_provider="anthropic"))

    assert isinstance(client, AsyncAnthropic)


def test_bedrock_backend_is_selected_by_settings() -> None:
    client = build_anthropic_client(settings(llm_provider="bedrock", aws_region="eu-west-1"))

    assert isinstance(client, AsyncAnthropicBedrockMantle)
    assert client.max_retries == 0


def test_provider_name_reaches_the_client_for_logging() -> None:
    client = build_llm_client(settings(llm_provider="bedrock", llm_model="anthropic.claude-opus-5"))

    assert client.provider_name == "bedrock"


def test_the_fake_backend_is_rejected_in_production() -> None:
    with pytest.raises(ValidationError, match="not usable in production"):
        settings(environment="production", llm_provider="fake")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValidationError):
        settings(llm_provider="openai")


def test_app_exposes_an_llm_client() -> None:
    app = create_app()

    assert app.state.llm_client.provider_name == "fake"


def test_an_injected_llm_client_is_used() -> None:
    injected = build_llm_client(settings())
    app = create_app(llm_client=injected)

    assert app.state.llm_client is injected
