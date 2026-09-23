"""Application configuration.

Environment-based settings loaded via Pydantic Settings. All values can be
overridden with ``EAP_``-prefixed environment variables or an ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from enterprise_agent_platform.mcp.policy import MCPServerConfig
from enterprise_agent_platform.tools.models import RiskLevel

Environment = Literal["development", "staging", "production", "test"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProviderName = Literal["fake", "anthropic", "bedrock"]
TaskStoreName = Literal["memory", "postgres"]


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

    agent_max_steps: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Model calls allowed in one agent run; bounds cost and tool loops.",
    )
    agent_max_tokens: int = Field(
        default=1024, ge=1, description="Output token budget for a single agent step."
    )
    agent_auto_approve_up_to: RiskLevel = Field(
        default=RiskLevel.READ,
        description="Highest tool risk an agent may run unattended; above it, a human decides.",
    )

    mcp_servers: tuple[MCPServerConfig, ...] = Field(
        default=(),
        description=(
            "MCP servers whose tools the agent may call, as a JSON list. Each entry names the "
            "server, the command that runs it, and the risk this deployment assigns to its "
            "tools; risk is never read from the server itself."
        ),
    )

    task_store: TaskStoreName = Field(
        default="memory",
        description="Task persistence backend. 'memory' is per-process and lost on restart.",
    )
    database_url: SecretStr | None = Field(
        default=None,
        description="postgresql://user:password@host:5432/database, for task_store='postgres'.",
    )
    database_pool_min_size: int = Field(
        default=1, ge=0, description="Connections kept open per process."
    )
    database_pool_max_size: int = Field(
        default=10,
        ge=1,
        description="Connection ceiling per process; replicas multiply it against the server's.",
    )
    database_command_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Deadline for a single statement, so a blocked write fails instead of hanging.",
    )

    @model_validator(mode="after")
    def _require_a_real_provider_in_production(self) -> Settings:
        """Fail at startup rather than on the first customer request."""
        if self.environment == "production" and self.llm_provider == "fake":
            raise ValueError("EAP_LLM_PROVIDER=fake is not usable in production")
        return self

    @model_validator(mode="after")
    def _require_a_durable_store_in_production(self) -> Settings:
        """A production deployment must not hold task state in one process.

        Since tasks carry the checkpoint a human approval resumes from, an
        in-memory store in production means a restart can strand a decision
        that has already been made.
        """
        if self.environment == "production" and self.task_store == "memory":
            raise ValueError("EAP_TASK_STORE=memory is not usable in production")
        return self

    @model_validator(mode="after")
    def _mcp_server_names_are_unique(self) -> Settings:
        """Names namespace tools, so a duplicate would make the prefix ambiguous."""
        names = [server.name for server in self.mcp_servers]
        if len(set(names)) != len(names):
            raise ValueError("EAP_MCP_SERVERS entries must have unique names")
        return self

    @model_validator(mode="after")
    def _database_settings_are_consistent(self) -> Settings:
        if self.task_store == "postgres" and self.database_url is None:
            raise ValueError("EAP_TASK_STORE=postgres requires EAP_DATABASE_URL")
        if self.database_pool_max_size < self.database_pool_min_size:
            raise ValueError(
                "EAP_DATABASE_POOL_MAX_SIZE must be greater than or equal to "
                "EAP_DATABASE_POOL_MIN_SIZE"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Caching keeps configuration resolution cheap and consistent across the app.
    Call ``get_settings.cache_clear()`` in tests when overriding environment.
    """
    return Settings()
