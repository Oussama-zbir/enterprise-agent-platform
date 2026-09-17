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


@dataclass(frozen=True, slots=True)
class Tool[ArgsT: BaseModel]:
    """One callable capability.

    ``arguments`` is both the contract sent to the model and the validator
    applied to what comes back, so the handler receives a typed model and never
    a raw dictionary from a language model.
    """

    name: str
    description: str
    arguments: type[ArgsT]
    risk: RiskLevel
    handler: Callable[[ArgsT], Awaitable[str]]

    def __post_init__(self) -> None:
        if fullmatch(TOOL_NAME_PATTERN, self.name) is None:
            raise ValueError(f"invalid tool name '{self.name}'")
        if not self.description.strip():
            # The description is the model's only guidance on when to call this
            # tool; an empty one is a bug that shows up as bad tool choices.
            raise ValueError(f"tool '{self.name}' needs a description")

    def spec(self) -> ToolSpec:
        """The provider-neutral declaration sent to the model."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.arguments.model_json_schema(),
        )


def requires_approval(tool: Tool[Any], *, auto_approve_up_to: RiskLevel) -> bool:
    """Whether running ``tool`` needs a human decision first.

    Expressed as a threshold rather than a per-tool flag so a deployment can
    tighten the whole platform (``auto_approve_up_to=READ``) without editing
    every tool definition.
    """
    return tool.risk.exceeds(auto_approve_up_to)
