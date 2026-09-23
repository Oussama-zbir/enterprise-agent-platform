"""MCP wire types and the failure taxonomy around them.

The protocol is JSON-RPC 2.0 over a byte stream. Only the client half of the
tools feature is modelled here — ``initialize``, ``tools/list``, ``tools/call``
— because that is all an agent platform needs to consume a server, and every
field the platform does not consume is one more thing a hostile server could
put in front of the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

JSONRPC_VERSION = "2.0"

PROTOCOL_VERSION = "2025-06-18"
"""The MCP revision this client speaks.

Sent in ``initialize``; a server that answers with a version it prefers is
accepted only if it is one this client also understands, because the shape of
``tools/call`` results differs between revisions.
"""

SUPPORTED_PROTOCOL_VERSIONS = frozenset({PROTOCOL_VERSION, "2025-03-26", "2024-11-05"})

CLIENT_NAME = "enterprise-agent-platform"


class MCPError(Exception):
    """Base class for every failure that originates below the tool layer."""


class MCPTransportError(MCPError):
    """The server could not be reached, spoke past the framing, or went away."""


class MCPProtocolError(MCPError):
    """The server answered, but not with something this client can use."""


class MCPRemoteError(MCPError):
    """The server returned a JSON-RPC error object for a request."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"MCP error {code}: {message}")
        self.code = code
        self.remote_message = message


class MCPToolDeclaration(BaseModel):
    """One tool as a server advertises it.

    ``extra="ignore"`` is a security control, not convenience. Servers may
    attach ``annotations`` (``readOnlyHint``, ``destructiveHint``), titles, and
    other self-asserted metadata; dropping them at the parse boundary means no
    server-asserted claim about its own safety can reach the platform's risk
    policy even by accident, because the platform has nowhere to put it. Risk is
    decided locally, from configuration — see ``policy``.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict, alias="inputSchema")


@dataclass(frozen=True, slots=True)
class MCPServerInfo:
    """What a server said about itself during the handshake, for logs only."""

    name: str
    version: str
    protocol_version: str


@dataclass(frozen=True, slots=True)
class MCPCallOutcome:
    """The result of one ``tools/call``.

    ``is_error`` is the server reporting that *the tool* failed, which is an
    outcome the model should see and may recover from. A protocol or transport
    failure raises instead: the model cannot fix a dead subprocess, and the
    detail belongs in logs rather than in a prompt.
    """

    content: str
    is_error: bool
