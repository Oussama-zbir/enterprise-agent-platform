"""Connect the MCP servers a deployment configured, and register their tools.

Connecting happens once, at application startup, and an unreachable server
fails startup rather than the first agent run that needed it — the same call
the database pool makes. A server that is down is a missing capability, and a
task that fails halfway through because a tool vanished is more expensive to
diagnose than a service that refuses to boot.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from enterprise_agent_platform.mcp.client import MCPClient
from enterprise_agent_platform.mcp.policy import MCPServerConfig
from enterprise_agent_platform.mcp.tools import register_mcp_tools
from enterprise_agent_platform.mcp.transport import StdioTransport
from enterprise_agent_platform.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class MCPConnections:
    """The connected servers an application owns, closed together at shutdown."""

    def __init__(self, clients: Sequence[MCPClient] = ()) -> None:
        self._clients = tuple(clients)

    @property
    def clients(self) -> tuple[MCPClient, ...]:
        return self._clients

    async def aclose(self) -> None:
        """Close every server, even if one of them fails to close."""
        for client in self._clients:
            try:
                await client.aclose()
            except Exception:  # pragma: no cover - defensive
                logger.exception("mcp.server.close_failed", extra={"mcp_server": client.name})


async def connect_mcp_servers(
    configs: Sequence[MCPServerConfig], registry: ToolRegistry
) -> MCPConnections:
    """Start each configured server, discover its tools, and register them.

    If any server fails, the ones already connected are closed before the error
    propagates: a failed startup must not leave orphaned child processes behind.
    """
    connections = MCPConnections()
    clients: list[MCPClient] = []
    try:
        for config in configs:
            client = await _connect(config, registry)
            clients.append(client)
            connections = MCPConnections(clients)
    except Exception:
        await connections.aclose()
        raise
    return connections


async def _connect(config: MCPServerConfig, registry: ToolRegistry) -> MCPClient:
    transport = StdioTransport(config.command, config.args, env=config.env, cwd=config.cwd)
    await transport.start()
    client = MCPClient(
        transport,
        name=config.name,
        request_timeout_seconds=config.call_timeout_seconds,
    )
    try:
        # One deadline over the whole handshake, so a server that accepts the
        # connection and then says nothing cannot hold up startup indefinitely.
        async with asyncio.timeout(config.startup_timeout_seconds):
            await client.initialize()
            declarations = await client.list_tools()
        register_mcp_tools(registry, client, config, declarations)
    except Exception:
        await client.aclose()
        raise
    return client
