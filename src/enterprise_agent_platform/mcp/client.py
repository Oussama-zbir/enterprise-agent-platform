"""The MCP client: handshake, tool discovery, and tool invocation.

Everything a server sends is treated as untrusted input from a remote party
that the platform did not write. That shows up as three concrete bounds — how
many tools a server may publish, how many pages of them this client will fetch,
and how much text one tool call may return — and as a parse step that keeps
unknown fields out of the platform entirely (see ``MCPToolDeclaration``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from enterprise_agent_platform import __version__
from enterprise_agent_platform.mcp.protocol import (
    CLIENT_NAME,
    PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    MCPCallOutcome,
    MCPProtocolError,
    MCPServerInfo,
    MCPToolDeclaration,
    MCPTransportError,
)
from enterprise_agent_platform.mcp.transport import MCPTransport

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
MAX_TOOLS_PER_SERVER = 128
MAX_LIST_PAGES = 20
MAX_RESULT_CHARS = 16_000
_TRUNCATION_NOTE = "\n[truncated]"


class MCPClient:
    """One connected MCP server, exposed as tool discovery and tool calls.

    The client is deliberately not a context manager over the transport: a
    server is connected once at application startup and lives for the process,
    because rediscovering tools per request would put a subprocess spawn and a
    handshake on the latency path of every agent run.
    """

    def __init__(
        self,
        transport: MCPTransport,
        *,
        name: str,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._transport = transport
        self._name = name
        self._timeout = request_timeout_seconds
        self._server_info: MCPServerInfo | None = None

    @property
    def name(self) -> str:
        """The local name for this server, used to namespace its tools."""
        return self._name

    @property
    def server_info(self) -> MCPServerInfo | None:
        return self._server_info

    async def initialize(self) -> MCPServerInfo:
        """Perform the MCP handshake.

        A server that negotiates a protocol revision this client does not
        understand is refused here rather than accommodated: the alternative is
        guessing at the shape of its ``tools/call`` results, and a wrong guess
        is a tool result that silently means something else.
        """
        result = await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": __version__},
            },
        )
        negotiated = result.get("protocolVersion")
        if not isinstance(negotiated, str) or negotiated not in SUPPORTED_PROTOCOL_VERSIONS:
            raise MCPProtocolError(
                f"MCP server '{self._name}' negotiated unsupported protocol version {negotiated!r}"
            )
        if "tools" not in _as_dict(result.get("capabilities")):
            raise MCPProtocolError(f"MCP server '{self._name}' does not offer tools")

        # The spec requires this notification before any other request; a
        # compliant server rejects tools/list until it arrives.
        await self._transport.notify("notifications/initialized")

        info = _as_dict(result.get("serverInfo"))
        self._server_info = MCPServerInfo(
            name=str(info.get("name", self._name)),
            version=str(info.get("version", "unknown")),
            protocol_version=negotiated,
        )
        logger.info(
            "mcp.server.connected",
            extra={
                "mcp_server": self._name,
                "server_name": self._server_info.name,
                "server_version": self._server_info.version,
                "protocol_version": negotiated,
            },
        )
        return self._server_info

    async def list_tools(self) -> tuple[MCPToolDeclaration, ...]:
        """Discover the server's tools, following pagination.

        Both bounds here exist because the loop is driven by the server: it
        chooses whether to return another cursor, and every tool it returns
        becomes a schema in the prompt of every agent run. An unbounded answer
        would be a remote party deciding this platform's context window size and
        per-call cost.
        """
        declarations: list[MCPToolDeclaration] = []
        cursor: str | None = None
        for _ in range(MAX_LIST_PAGES):
            params = {"cursor": cursor} if cursor is not None else None
            result = await self._request("tools/list", params)
            raw_tools = result.get("tools")
            if not isinstance(raw_tools, list):
                raise MCPProtocolError(f"MCP server '{self._name}' returned no tool list")
            for raw in raw_tools:
                declarations.append(self._parse_declaration(raw))
            if len(declarations) > MAX_TOOLS_PER_SERVER:
                raise MCPProtocolError(
                    f"MCP server '{self._name}' published more than {MAX_TOOLS_PER_SERVER} tools"
                )
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            cursor = next_cursor
        else:
            raise MCPProtocolError(f"MCP server '{self._name}' paginated tools/list without end")

        logger.info(
            "mcp.tools.discovered",
            extra={"mcp_server": self._name, "tool_count": len(declarations)},
        )
        return tuple(declarations)

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> MCPCallOutcome:
        """Invoke one remote tool and flatten its result to text."""
        result = await self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        return MCPCallOutcome(
            content=_render_content(result.get("content")),
            is_error=result.get("isError") is True,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def _request(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """One request under a hard deadline.

        The deadline lives here rather than in the transport so that every
        transport inherits it, and so a server that accepts a request and never
        answers costs one timeout instead of an agent run that never ends.
        """
        try:
            async with asyncio.timeout(self._timeout):
                return await self._transport.request(method, params)
        except TimeoutError as exc:
            raise MCPTransportError(
                f"MCP server '{self._name}' did not answer '{method}' within {self._timeout}s"
            ) from exc

    def _parse_declaration(self, raw: object) -> MCPToolDeclaration:
        if not isinstance(raw, dict):
            raise MCPProtocolError(f"MCP server '{self._name}' published a malformed tool")
        try:
            return MCPToolDeclaration.model_validate(raw)
        except ValidationError as exc:
            raise MCPProtocolError(
                f"MCP server '{self._name}' published a malformed tool: "
                f"{exc.errors(include_input=False, include_url=False)}"
            ) from exc


def _render_content(content: object) -> str:
    """Flatten MCP content blocks into the text a tool result carries.

    Non-text blocks (images, audio, embedded resources) are named rather than
    inlined: the tool contract in this platform is ``str``, and a base64 image
    dropped into a prompt is cost with no meaning attached.
    """
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        else:
            parts.append(f"[{block_type or 'unknown'} content omitted]")
    rendered = "\n".join(parts)
    if len(rendered) <= MAX_RESULT_CHARS:
        return rendered
    return rendered[: MAX_RESULT_CHARS - len(_TRUNCATION_NOTE)] + _TRUNCATION_NOTE


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
