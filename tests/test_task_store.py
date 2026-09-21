"""Tests for task-store wiring: settings to adapter, and the app lifecycle.

These stay offline. What they check is the decision the configuration makes and
the resources the application opens — not the SQL, which the contract tests
cover against a real server.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from enterprise_agent_platform.config import Settings
from enterprise_agent_platform.main import create_app
from enterprise_agent_platform.tasks.factory import build_task_repository
from enterprise_agent_platform.tasks.repository import InMemoryTaskRepository, ManagedRepository

DSN = "postgresql://eap:secret@localhost:5432/eap"


def settings(**overrides: Any) -> Settings:
    """Settings built from explicit values only, ignoring any local ``.env``."""
    # `_env_file` is a pydantic-settings runtime argument, absent from the
    # model's generated __init__ signature.
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


class RecordingRepository(InMemoryTaskRepository):
    """A repository that owns a resource, standing in for the pooled adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.connects = 0
        self.closes = 0

    async def connect(self) -> None:
        self.connects += 1

    async def aclose(self) -> None:
        self.closes += 1


def test_default_store_is_in_memory() -> None:
    assert isinstance(build_task_repository(settings()), InMemoryTaskRepository)


def test_postgres_store_is_built_without_touching_the_database() -> None:
    """Construction must not connect: the pool needs a running loop and startup
    is where a connection failure should surface."""
    pytest.importorskip("asyncpg")
    from enterprise_agent_platform.tasks.postgres import PostgresTaskRepository

    repository = build_task_repository(settings(task_store="postgres", database_url=DSN))

    assert isinstance(repository, PostgresTaskRepository)


def test_postgres_store_requires_a_database_url() -> None:
    with pytest.raises(ValidationError, match="requires EAP_DATABASE_URL"):
        settings(task_store="postgres")


def test_in_memory_store_is_rejected_in_production() -> None:
    """Task state that dies with the process would strand a human approval."""
    with pytest.raises(ValidationError, match="not usable in production"):
        settings(environment="production", llm_provider="anthropic", task_store="memory")


def test_pool_bounds_must_be_consistent() -> None:
    with pytest.raises(ValidationError, match="POOL_MAX_SIZE"):
        settings(
            task_store="postgres",
            database_url=DSN,
            database_pool_min_size=5,
            database_pool_max_size=2,
        )


def test_database_url_is_not_printed_by_accident() -> None:
    """The URL carries a password, so it must be a secret, not a plain string."""
    configured = settings(task_store="postgres", database_url=DSN)

    assert "secret" not in repr(configured)
    assert configured.database_url is not None
    assert configured.database_url.get_secret_value() == DSN


def test_only_a_repository_owning_resources_is_managed() -> None:
    assert isinstance(RecordingRepository(), ManagedRepository)
    assert not isinstance(InMemoryTaskRepository(), ManagedRepository)


def test_lifespan_opens_and_closes_a_pooled_repository() -> None:
    repository = RecordingRepository()

    with TestClient(create_app(task_repository=repository)):
        assert (repository.connects, repository.closes) == (1, 0)

    assert (repository.connects, repository.closes) == (1, 1)
