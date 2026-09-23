"""Tests for the stdio transport and startup wiring, against a real subprocess.

Everything else in the MCP suite runs against an in-process stub, which proves
the client's logic but not its framing. These tests spawn ``stub_mcp_server.py``
with this interpreter — offline, no network, no installed server — so the
newline-delimited JSON-RPC, the reply routing, and the process lifetime are
actually exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from enterprise_agent_platform.mcp.client import MCPClient
from enterprise_agent_platform.mcp.factory import connect_mcp_servers
from enterprise_agent_platform.mcp.policy import MCPServerConfig
from enterprise_agent_platform.mcp.protocol import MCPRemoteError, MCPTransportError
from enterprise_agent_platform.mcp.transport import StdioTransport
from enterprise_agent_platform.tools.models import RiskLevel
from enterprise_agent_platform.tools.registry import ToolRegistry

STUB_SERVER = str(Path(__file__).parent / "stub_mcp_server.py")


def stub_config(**overrides: object) -> MCPServerConfig:
    return MCPServerConfig.model_validate(
        {"name": "stub", "command": sys.executable, "args": [STUB_SERVER], **overrides}
    )


async def connected() -> MCPClient:
    transport = StdioTransport(sys.executable, [STUB_SERVER])
    await transport.start()
    client = MCPClient(transport, name="stub", request_timeout_seconds=10.0)
    await client.initialize()
    return client


async def test_a_real_child_process_completes_the_handshake_and_a_tool_call() -> None:
    client = await connected()
    try:
        (tool,) = await client.list_tools()
        outcome = await client.call_tool("echo", {"message": "hello"})
    finally:
        await client.aclose()

    assert tool.name == "echo"
    assert (outcome.content, outcome.is_error) == ("echo: hello", False)


async def test_concurrent_calls_are_matched_to_their_own_replies() -> None:
    # One pipe carries every reply, so a run issuing parallel tool calls depends
    # on ids being routed rather than responses being read in order.
    import asyncio

    client = await connected()
    try:
        outcomes = await asyncio.gather(
            *(client.call_tool("echo", {"message": str(index)}) for index in range(8))
        )
    finally:
        await client.aclose()

    assert [outcome.content for outcome in outcomes] == [f"echo: {index}" for index in range(8)]


async def test_a_jsonrpc_error_from_the_child_reaches_the_caller() -> None:
    client = await connected()
    try:
        with pytest.raises(MCPRemoteError) as caught:
            await client.call_tool("no_such_tool", {})
    finally:
        await client.aclose()

    assert caught.value.code == -32602


async def test_a_command_that_does_not_exist_fails_to_start() -> None:
    transport = StdioTransport("/nonexistent/mcp-server")

    with pytest.raises(MCPTransportError, match="could not start"):
        await transport.start()


async def test_a_request_after_close_is_refused() -> None:
    client = await connected()
    await client.aclose()

    with pytest.raises(MCPTransportError, match="not connected"):
        await client.call_tool("echo", {"message": "hi"})


async def test_startup_registers_the_servers_tools_under_local_risk() -> None:
    registry = ToolRegistry()

    # The stub declares `readOnlyHint: true` on its only tool; this deployment
    # maps it to WRITE, and that is the risk the platform uses.
    connections = await connect_mcp_servers(
        [stub_config(tool_risk={"echo": RiskLevel.WRITE})], registry
    )
    try:
        assert registry.names == ("stub__echo",)
        tool = registry.get("stub__echo")
        assert tool is not None and tool.risk is RiskLevel.WRITE
    finally:
        await connections.aclose()


async def test_a_server_that_cannot_be_started_fails_startup() -> None:
    registry = ToolRegistry()

    with pytest.raises(MCPTransportError):
        await connect_mcp_servers([stub_config(command="/nonexistent/mcp-server")], registry)

    # No half-configured deployment: a failed connection registers nothing.
    assert registry.names == ()


async def test_one_unreachable_server_aborts_startup_for_the_whole_deployment() -> None:
    registry = ToolRegistry()

    with pytest.raises(MCPTransportError):
        await connect_mcp_servers(
            [stub_config(), stub_config(name="broken", command="/nonexistent/mcp-server")],
            registry,
        )

    # The earlier server was registered before the failure, and the clients
    # opened so far are closed on the way out so startup leaves no orphaned
    # child processes. The half-populated registry dies with the process:
    # a missing server is a missing capability, not something to boot around.
    assert registry.names == ("stub__echo",)
