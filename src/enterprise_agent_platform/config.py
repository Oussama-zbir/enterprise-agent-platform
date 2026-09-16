"""Application configuration.

Environment-based settings loaded via Pydantic Settings. All values can be
overridden with ``EAP_``-prefixed environment variables or an ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production", "test"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProviderName = Literal["fake", "anthropic", "bedrock"]


class Settings(BaseSettings):
    """Runtime configuration for the platform."""

    model_config = SettingsConfigDict(
        env_prefix="EAP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "enterprise-agent-platform"
    environment: Environment = "development"
    log_level: LogLevel = "INFO"
    debug: bool = Field(
        default=False,
        description="Enable verbose framework behaviour; never enable in production.",
    )

    llm_provider: LLMProviderName = Field(
        default="fake",
        description="Model backend. 'fake' is offline only: every call raises.",
    )
    llm_model: str = Field(
        default="claude-opus-5",
        description="Model ID. Bedrock IDs carry an 'anthropic.' prefix.",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Deadline for a single model call, enforced by LLMClient.",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Falls back to the SDK's own credential resolution when unset.",
    )
    aws_region: str = Field(default="us-east-1", description="Region for the Bedrock backend.")

    @model_validator(mode="after")
    def _require_a_real_provider_in_production(self) -> Settings:
        """Fail at startup rather than on the first customer request."""
        if self.environment == "production" and self.llm_provider == "fake":
            raise ValueError("EAP_LLM_PROVIDER=fake is not usable in production")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Caching keeps configuration resolution cheap and consistent across the app.
    Call ``get_settings.cache_clear()`` in tests when overriding environment.
    """
    return Settings()
