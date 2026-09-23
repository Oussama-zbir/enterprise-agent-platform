"""Tool definitions and the risk metadata that drives approval policy.

A tool is a typed function the model may ask the platform to run: a Pydantic
model describes its arguments (and generates the JSON Schema the model sees), an
async handler performs the work, and a ``RiskLevel`` records what running it can
do to the outside world. The risk level is a property of the tool; whether a
given risk needs a human is policy, and lives in ``requires_approval`` so the
orchestrator and the approval workflow cannot disagree about it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from re import fullmatch
from typing import Any

from pydantic import BaseModel

from enterprise_agent_platform.llm.models import TOOL_NAME_PATTERN, ToolSpec


class RiskLevel(StrEnum):
    """What a tool can do, ordered by how much a mistake costs.

    ``READ`` observes state, ``WRITE`` changes state the platform can undo, and
    ``CRITICAL`` is irreversible or visible outside the organisation — paying an
    invoice, emailing a customer, deleting a record.
    """

    READ = "read"
    WRITE = "write"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _RISK_RANK[self]

    def exceeds(self, other: RiskLevel) -> bool:
        return self.rank > other.rank


_RISK_RANK: dict[RiskLevel, int] = {
    RiskLevel.READ: 0,
    RiskLevel.WRITE: 1,
    RiskLevel.CRITICAL: 2,
}


class ToolExecutionError(Exception):
    """A tool-level failure the model is allowed to see and act on.

    Handlers raise this for outcomes that are part of the tool's own contract —
    an invoice that does not exist, a remote tool that reported a failure — and
    the message is handed back to the model so it can correct itself. Every
    other exception is treated as a defect: it is logged and replaced with a
    generic message, because its text can carry internals the model must never
    see.
    """


@dataclass(frozen=True, slots=True)
class Tool[ArgsT: BaseModel]:
    """One callable capability.

    ``arguments`` is both the contract sent to the model and the validator
    applied to what comes back, so the handler receives a typed model and never
    a raw dictionary from a language model.

    ``input_schema`` overrides the schema shown to the model while leaving
    validation with ``arguments``. Only tools whose contract is defined
    elsewhere need it — an MCP server publishes its own JSON Schema, with
    per-property descriptions and constraints a locally derived model cannot
    reproduce, and showing the model less would cost tool-call accuracy. The
    override must be a *relaxation* of what ``arguments`` accepts, never a
    tightening, or the platform would reject arguments the model was told were
    valid.
    """

    name: str
    description: str
    arguments: type[ArgsT]
    risk: RiskLevel
    handler: Callable[[ArgsT], Awaitable[str]]
    input_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if fullmatch(TOOL_NAME_PATTERN, self.name) is None:
            raise ValueError(f"invalid tool name '{self.name}'")
        if not self.description.strip():
            # The description is the model's only guidance on when to call this
            # tool; an empty one is a bug that shows up as bad tool choices.
            raise ValueError(f"tool '{self.name}' needs a description")

    def spec(self) -> ToolSpec:
        """The provider-neutral declaration sent to the model."""
        schema = (
            self.input_schema
            if self.input_schema is not None
            else self.arguments.model_json_schema()
        )
        return ToolSpec(name=self.name, description=self.description, input_schema=schema)


def requires_approval(tool: Tool[Any], *, auto_approve_up_to: RiskLevel) -> bool:
    """Whether running ``tool`` needs a human decision first.

    Expressed as a threshold rather than a per-tool flag so a deployment can
    tighten the whole platform (``auto_approve_up_to=READ``) without editing
    every tool definition.
    """
    return tool.risk.exceeds(auto_approve_up_to)
