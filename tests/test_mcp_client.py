"""Tests for the MCP client: handshake, discovery, and tool invocation.

The client's job is to treat a server as a remote party that this repository
does not control, so most of what is asserted here is refusal — an unsupported
protocol revision, an endless tool list, a server that answers with nothing
usable — rather than the happy path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from enterprise_agent_platform.mcp.client import MAX_LIST_PAGES, MAX_TOOLS_PER_SERVER, MCPClient
from enterprise_agent_platform.mcp.protocol import (
    MCPProtocolError,
    MCPRemoteError,
    MCPTransportError,
)


class StubTransport:
    """An in-process MCP server, scripted per method."""

    def __init__(self, responses: Mapping[str, Any] | None = None) -> None:
        self.responses: dict[str, Any] = dict(responses or {})
        self.requests: list[tuple[str, dict[str, Any] | None]] = []
        self.notifications: list[str] = []
        self.closed = False

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.requests.append((method, dict(params) if params is not None else None))
        response = self.responses.get(method)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(params)
        if response is None:
            raise MCPProtocolError(f"stub has no response for {method}")
        assert isinstance(response, dict)
        return response

    async def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        self.notifications.append(method)

    async def aclose(self) -> None:
        self.closed = True


def handshake(*, version: str = "2025-06-18", tools: bool = True) -> dict[str, Any]:
    return {
        "protocolVersion": version,
        "capabilities": {"tools": {}} if tools else {"resources": {}},
        "serverInfo": {"name": "stub", "version": "1.2.3"},
    }


def declaration(name: str = "echo", **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "description": "Echo a message.",
        "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}},
        **extra,
    }


def client(transport: StubTransport, *, timeout: float = 5.0) -> MCPClient:
    return MCPClient(transport, name="stub", request_timeout_seconds=timeout)


async def test_initialize_negotiates_and_sends_the_initialized_notification() -> None:
    transport = StubTransport({"initialize": handshake()})

    info = await client(transport).initialize()

    assert (info.name, info.version, info.protocol_version) == ("stub", "1.2.3", "2025-06-18")
    assert transport.notifications == ["notifications/initialized"]


async def test_initialize_rejects_a_protocol_revision_the_client_cannot_read() -> None:
    transport = StubTransport({"initialize": handshake(version="2099-01-01")})

    with pytest.raises(MCPProtocolError, match="unsupported protocol version"):
        await client(transport).initialize()

    # The handshake never completed, so the server was never told to proceed.
    assert transport.notifications == []


async def test_initialize_rejects_a_server_that_offers_no_tools() -> None:
    transport = StubTransport({"initialize": handshake(tools=False)})

    with pytest.raises(MCPProtocolError, match="does not offer tools"):
        await client(transport).initialize()


async def test_list_tools_follows_pagination() -> None:
    pages = [
        {"tools": [declaration("first")], "nextCursor": "page-2"},
        {"tools": [declaration("second")]},
    ]
    transport = StubTransport({"tools/list": lambda params: pages.pop(0)})

    tools = await client(transport).list_tools()

    assert [tool.name for tool in tools] == ["first", "second"]
    assert transport.requests[1][1] == {"cursor": "page-2"}


async def test_list_tools_refuses_a_cursor_that_never_ends() -> None:
    # A server can keep handing out cursors forever; the client, not the server,
    # decides when discovery stops.
    transport = StubTransport(
        {"tools/list": lambda params: {"tools": [declaration()], "nextCursor": "more"}}
    )

    with pytest.raises(MCPProtocolError, match="paginated tools/list without end"):
        await client(transport).list_tools()

    assert len(transport.requests) == MAX_LIST_PAGES


async def test_list_tools_refuses_a_server_that_publishes_too_many_tools() -> None:
    # Every tool becomes a schema in the prompt of every run, so the size of the
    # tool list is this platform's cost, not the server's.
    many = [declaration(f"tool_{index}") for index in range(MAX_TOOLS_PER_SERVER + 1)]
    transport = StubTransport({"tools/list": {"tools": many}})

    with pytest.raises(MCPProtocolError, match="more than"):
        await client(transport).list_tools()


async def test_list_tools_rejects_a_malformed_declaration() -> None:
    transport = StubTransport({"tools/list": {"tools": [{"description": "no name"}]}})

    with pytest.raises(MCPProtocolError, match="malformed tool"):
        await client(transport).list_tools()


async def test_a_declaration_drops_everything_the_platform_does_not_consume() -> None:
    transport = StubTransport(
        {
            "tools/list": {
                "tools": [
                    declaration(annotations={"readOnlyHint": True}, title="Safe", risk="read")
                ]
            }
        }
    )

    (tool,) = await client(transport).list_tools()

    # The parsed declaration has no field a server could use to describe itself
    # as safe, which is what makes local risk policy unbypassable.
    assert set(tool.model_dump()) == {"name", "description", "input_schema"}


async def test_call_tool_flattens_text_blocks() -> None:
    transport = StubTransport(
        {
            "tools/call": {
                "content": [
                    {"type": "text", "text": "line one"},
                    {"type": "image", "data": "..."},
                    {"type": "text", "text": "line two"},
                ]
            }
        }
    )

    outcome = await client(transport).call_tool("echo", {"message": "hi"})

    assert outcome.content == "line one\n[image content omitted]\nline two"
    assert outcome.is_error is False
    assert transport.requests == [("tools/call", {"name": "echo", "arguments": {"message": "hi"}})]


async def test_call_tool_reports_a_server_side_tool_failure_as_an_outcome() -> None:
    transport = StubTransport(
        {"tools/call": {"content": [{"type": "text", "text": "no such invoice"}], "isError": True}}
    )

    outcome = await client(transport).call_tool("lookup", {})

    assert (outcome.is_error, outcome.content) == (True, "no such invoice")


async def test_a_jsonrpc_error_surfaces_as_a_remote_error() -> None:
    transport = StubTransport({"tools/call": MCPRemoteError(-32602, "invalid params")})

    with pytest.raises(MCPRemoteError) as caught:
        await client(transport).call_tool("echo", {})

    assert caught.value.code == -32602


async def test_a_server_that_never_answers_hits_the_client_deadline() -> None:
    class SilentTransport(StubTransport):
        async def request(
            self, method: str, params: Mapping[str, Any] | None = None
        ) -> dict[str, Any]:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    with pytest.raises(MCPTransportError, match="did not answer"):
        await client(SilentTransport(), timeout=0.01).call_tool("echo", {})


async def test_closing_the_client_closes_the_transport() -> None:
    transport = StubTransport()

    await client(transport).aclose()

    assert transport.closed is True
