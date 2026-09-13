"""FastAPI application entrypoint.

Exposes the application factory and a module-level ``app`` for ASGI servers.
Only foundational concerns live here today (config, logging, health). Agent
routes, MCP integration, and tool calling arrive in later milestones.
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


def create_app() -> FastAPI:
    """Build and configure a FastAPI application instance."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )

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
