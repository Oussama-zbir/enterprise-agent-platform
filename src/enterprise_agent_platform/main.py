"""FastAPI application entrypoint.

Exposes the application factory and a module-level ``app`` for ASGI servers.
Wires configuration, logging, request correlation IDs, health, and the task
API. Agent orchestration, MCP integration, and tool calling arrive in later
milestones.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from enterprise_agent_platform import __version__
from enterprise_agent_platform.config import get_settings
from enterprise_agent_platform.llm.client import LLMClient
from enterprise_agent_platform.llm.factory import build_llm_client
from enterprise_agent_platform.logging import configure_logging
from enterprise_agent_platform.request_context import RequestContextMiddleware
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
        # Releases the model backend's HTTP connection pool.
        await app.state.llm_client.aclose()
        logger.info("service.shutdown")


def create_app(
    task_repository: TaskRepository | None = None,
    llm_client: LLMClient | None = None,
) -> FastAPI:
    """Build and configure a FastAPI application instance.

    ``task_repository`` and ``llm_client`` let callers (tests, alternative
    deployments) inject adapters; they default to the in-memory repository and
    the model backend named by ``EAP_LLM_PROVIDER``.
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
    app.state.llm_client = llm_client if llm_client is not None else build_llm_client(settings)
    app.add_middleware(RequestContextMiddleware)
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
