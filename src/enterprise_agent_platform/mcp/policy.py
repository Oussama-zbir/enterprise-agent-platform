"""Configuration for one MCP server, and the risk policy applied to its tools.

The rule this module exists to enforce: **an MCP server supplies capability, it
does not get a say in whether a human approves using it.**

Tool discovery happens at runtime against code outside this repository. Without
a locally owned mapping, adding a tool named ``transfer_funds`` to a remote
server would silently grant a running agent an ungated capability, and a
compromised server could advertise ``readOnlyHint: true`` on it. So risk is
resolved from three local sources only, in order:

1. an explicit per-tool override, keyed by the name the server publishes;
2. the server's configured default risk;
3. ``RiskLevel.CRITICAL``, the default of that default, so a tool that appears
   on a server after this deployment was configured is gated rather than waved
   through.

The failure mode of this design is an unnecessary approval prompt. It is never
an ungated action, and no value from the server is an input to the decision.
"""

from __future__ import annotations

from re import fullmatch

from pydantic import BaseModel, ConfigDict, Field

from enterprise_agent_platform.llm.models import TOOL_NAME_PATTERN
from enterprise_agent_platform.tools.models import RiskLevel

SERVER_NAME_PATTERN = r"^[a-z0-9][a-z0-9_]{0,23}$"
"""Server names are prefixed onto tool names, so they share that alphabet and
are short enough to leave room for the tool's own name."""


class MCPServerConfig(BaseModel):
    """A deployment's declaration of one MCP server it trusts to supply tools."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        pattern=SERVER_NAME_PATTERN,
        description="Local identifier; prefixed onto every tool this server publishes.",
    )
    command: str = Field(min_length=1, description="Executable to run as the server process.")
    args: tuple[str, ...] = ()
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment for the child process. Nothing is inherited from this one.",
    )
    cwd: str | None = None
    default_risk: RiskLevel = Field(
        default=RiskLevel.CRITICAL,
        description="Risk assigned to any tool this server publishes without an override.",
    )
    tool_risk: dict[str, RiskLevel] = Field(
        default_factory=dict,
        description="Per-tool risk, keyed by the name the server publishes. Wins over the default.",
    )
    startup_timeout_seconds: float = Field(
        default=15.0, gt=0, description="Deadline for the handshake and first tool discovery."
    )
    call_timeout_seconds: float = Field(
        default=30.0, gt=0, description="Deadline for a single tools/call."
    )

    def risk_for(self, remote_tool_name: str) -> RiskLevel:
        """The risk level this platform assigns to one of the server's tools.

        Takes only the tool's name — not its declaration — so there is no way
        for server-supplied metadata to reach the decision.
        """
        return self.tool_risk.get(remote_tool_name, self.default_risk)

    def local_name(self, remote_tool_name: str) -> str:
        """The name this tool is registered and offered to the model under.

        Namespacing is what keeps two servers that both publish ``search`` from
        colliding in one registry, and it keeps a remote server from shadowing a
        first-party tool the platform ships.
        """
        return f"{self.name}__{remote_tool_name}"

    def is_registrable(self, remote_tool_name: str) -> bool:
        """Whether the namespaced name is one every provider will accept."""
        return fullmatch(TOOL_NAME_PATTERN, self.local_name(remote_tool_name)) is not None
