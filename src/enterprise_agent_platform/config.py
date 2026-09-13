"""Application configuration.

Environment-based settings loaded via Pydantic Settings. All values can be
overridden with ``EAP_``-prefixed environment variables or an ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production", "test"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


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


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Caching keeps configuration resolution cheap and consistent across the app.
    Call ``get_settings.cache_clear()`` in tests when overriding environment.
    """
    return Settings()
