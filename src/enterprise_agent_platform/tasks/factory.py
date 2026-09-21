"""Build the task repository from settings.

Which store a deployment uses is configuration, not application logic — the same
reasoning that keeps the model backend out of ``LLMClient``. Callers depend on
the ``TaskRepository`` port and never learn which adapter they got.
"""

from __future__ import annotations

from enterprise_agent_platform.config import Settings
from enterprise_agent_platform.tasks.repository import InMemoryTaskRepository, TaskRepository


def build_task_repository(settings: Settings) -> TaskRepository:
    """Return the repository named by ``EAP_TASK_STORE``.

    ``Settings`` has already rejected a ``postgres`` store with no URL, and a
    ``memory`` store in production.
    """
    if settings.task_store == "memory":
        return InMemoryTaskRepository()

    # Imported here so that asyncpg stays an optional dependency: a deployment
    # running the in-memory store (or the test suite) does not need the driver
    # installed to import the application.
    from enterprise_agent_platform.tasks.postgres import PostgresTaskRepository

    assert settings.database_url is not None  # guaranteed by Settings validation
    return PostgresTaskRepository(
        settings.database_url.get_secret_value(),
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
        command_timeout=settings.database_command_timeout_seconds,
    )
