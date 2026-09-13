"""FastAPI application entrypoint.

Exposes the application factory and a module-level ``app`` for ASGI servers.
Wires configuration, logging, health, and the task API. Agent orchestration,
MCP integration, and tool calling arrive in later milestones.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from enterprise_agent_platform import __version__
from enterprise_agent_platform.config import get_settings
from enterprise_agent_platform.logging import configure_logging
from enterprise_agent_platform.tasks.repository import InMemoryTaskRepository, TaskRepository
from enterprise_agent_platform.tasks.router import router as tasks_router

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str
    service: str
    environment: str
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging on startup and log lifecycle events."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("service.startup", extra={"environment": settings.environment})
    try:
        yield
    finally:
        logger.info("service.shutdown")


def create_app(task_repository: TaskRepository | None = None) -> FastAPI:
    """Build and configure a FastAPI application instance.

    ``task_repository`` lets callers (tests, alternative deployments) inject a
    storage adapter; it defaults to the in-memory implementation.
    """
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.task_repository = (
        task_repository if task_repository is not None else InMemoryTaskRepository()
    )
    app.include_router(tasks_router)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            environment=settings.environment,
            version=__version__,
        )

    return app


app = create_app()
