"""FastAPI application entrypoint.

Exposes the application factory and a module-level ``app`` for ASGI servers.
Wires configuration, logging, request correlation IDs, health, the task API
(including the human approval routes), the agent runner with the tool
registry it may call, and the MCP servers whose tools that registry offers.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from enterprise_agent_platform import __version__
from enterprise_agent_platform.agent.runner import AgentRunner
from enterprise_agent_platform.config import Settings, get_settings
from enterprise_agent_platform.demo.tools import build_demo_tools
from enterprise_agent_platform.llm.client import LLMClient
from enterprise_agent_platform.llm.factory import build_llm_client
from enterprise_agent_platform.logging import configure_logging
from enterprise_agent_platform.mcp.factory import MCPConnections, connect_mcp_servers
from enterprise_agent_platform.request_context import RequestContextMiddleware
from enterprise_agent_platform.tasks.factory import build_task_repository
from enterprise_agent_platform.tasks.repository import ManagedRepository, TaskRepository
from enterprise_agent_platform.tasks.router import router as tasks_router
from enterprise_agent_platform.tools.models import Tool
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
    # Remote tools are discovered once, here, for the same reason: a server that
    # is unreachable is a capability this deployment does not have, and finding
    # that out at boot is cheaper than finding it out mid-task.
    app.state.mcp_connections = await connect_mcp_servers(
        settings.mcp_servers, app.state.tool_registry
    )
    logger.info(
        "service.startup",
        extra={
            "environment": settings.environment,
            "task_store": settings.task_store,
            "tool_count": len(app.state.tool_registry.names),
        },
    )
    try:
        yield
    finally:
        # Releases the model backend's HTTP connection pool.
        await app.state.llm_client.aclose()
        await app.state.mcp_connections.aclose()
        if isinstance(repository, ManagedRepository):
            await repository.aclose()
        logger.info("service.shutdown")


def _configured_tools(settings: Settings) -> tuple[Tool[Any], ...]:
    """The tools a deployment gets without registering any itself.

    None, unless it asked for the demo set: a platform that grants capabilities
    by default grants them to every agent running on it.
    """
    return build_demo_tools() if settings.demo_tools else ()


def create_app(
    task_repository: TaskRepository | None = None,
    llm_client: LLMClient | None = None,
    tool_registry: ToolRegistry | None = None,
) -> FastAPI:
    """Build and configure a FastAPI application instance.

    The three adapters are injectable (tests, alternative deployments) and
    default to the store named by ``EAP_TASK_STORE``, the model backend named by
    ``EAP_LLM_PROVIDER``, and a registry holding only what configuration asked
    for. An empty registry is the deliberate default: a deployment declares the
    tools its agents may use, so the platform ships with no capabilities of its
    own. ``EAP_DEMO_TOOLS`` adds the synthetic demo set so the platform can be
    run end to end out of the box, and the tools published by the servers in
    ``EAP_MCP_SERVERS`` are added to the registry during startup, once an event
    loop exists to connect them over.
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
    app.state.tool_registry = (
        tool_registry if tool_registry is not None else ToolRegistry(_configured_tools(settings))
    )
    app.state.mcp_connections = MCPConnections()
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
