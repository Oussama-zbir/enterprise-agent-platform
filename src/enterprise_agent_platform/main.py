"""FastAPI application entrypoint.

Exposes the application factory and a module-level ``app`` for ASGI servers.
Wires configuration, logging, request correlation IDs, health, the task API
(including the human approval routes), and the agent runner with the tool
registry it may call. MCP integration arrives in a later milestone.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from enterprise_agent_platform import __version__
from enterprise_agent_platform.agent.runner import AgentRunner
from enterprise_agent_platform.config import get_settings
from enterprise_agent_platform.llm.client import LLMClient
from enterprise_agent_platform.llm.factory import build_llm_client
from enterprise_agent_platform.logging import configure_logging
from enterprise_agent_platform.request_context import RequestContextMiddleware
from enterprise_agent_platform.tasks.factory import build_task_repository
from enterprise_agent_platform.tasks.repository import ManagedRepository, TaskRepository
from enterprise_agent_platform.tasks.router import router as tasks_router
from enterprise_agent_platform.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str
    service: str
    environment: str
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging, open pooled resources, and log lifecycle events."""
    settings = get_settings()
    configure_logging(settings.log_level)
    repository = app.state.task_repository
    if isinstance(repository, ManagedRepository):
        # A pooled adapter needs a running event loop, so its pool cannot be
        # opened in the constructor. Doing it here also means an unreachable
        # database fails startup instead of the first request that touches it.
        await repository.connect()
    logger.info(
        "service.startup",
        extra={"environment": settings.environment, "task_store": settings.task_store},
    )
    try:
        yield
    finally:
        # Releases the model backend's HTTP connection pool.
        await app.state.llm_client.aclose()
        if isinstance(repository, ManagedRepository):
            await repository.aclose()
        logger.info("service.shutdown")


def create_app(
    task_repository: TaskRepository | None = None,
    llm_client: LLMClient | None = None,
    tool_registry: ToolRegistry | None = None,
) -> FastAPI:
    """Build and configure a FastAPI application instance.

    The three adapters are injectable (tests, alternative deployments) and
    default to the store named by ``EAP_TASK_STORE``, the model backend named by
    ``EAP_LLM_PROVIDER``, and an empty tool registry. An empty registry is a
    deliberate default: a deployment declares the tools its agents may use, so
    the platform ships with no capabilities of its own.
    """
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.task_repository = (
        task_repository if task_repository is not None else build_task_repository(settings)
    )
    app.state.llm_client = llm_client if llm_client is not None else build_llm_client(settings)
    app.state.tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
    app.state.agent_runner = AgentRunner(
        app.state.llm_client,
        app.state.tool_registry,
        app.state.task_repository,
        max_steps=settings.agent_max_steps,
        max_tokens=settings.agent_max_tokens,
        auto_approve_up_to=settings.agent_auto_approve_up_to,
    )
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
