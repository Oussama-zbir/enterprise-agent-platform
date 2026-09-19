"""Agent task domain model.

An ``AgentTask`` is the unit of work the platform executes on behalf of a user.
Its lifecycle is an explicit state machine so that later milestones
(orchestration, human approval, evaluation) share a single source of truth
about which state changes are legal. Tasks are immutable: every transition
returns a new instance with an incremented ``version`` and an audit record.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from enterprise_agent_platform.llm.models import Message, Role, TokenUsage, ToolCall


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


ALLOWED_TRANSITIONS: Mapping[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED}),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.AWAITING_APPROVAL,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    # A human either approves (resume running) or rejects (cancel).
    TaskStatus.AWAITING_APPROVAL: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


class InvalidTransitionError(Exception):
    """Raised when a status change is not permitted by the lifecycle."""

    def __init__(self, current: TaskStatus, target: TaskStatus) -> None:
        super().__init__(f"cannot transition task from '{current}' to '{target}'")
        self.current = current
        self.target = target


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _check_checkpoint(status: TaskStatus, checkpoint: RunCheckpoint | None) -> None:
    """Enforce that only a task waiting on a human carries run state.

    A task that is not paused has nothing to resume, and a conversation left
    behind on a finished task is both a stale-resumption hazard and customer
    data kept for no reason.
    """
    if checkpoint is not None and status is not TaskStatus.AWAITING_APPROVAL:
        raise ValueError(f"a '{status}' task cannot carry a run checkpoint")


class StatusTransition(BaseModel):
    """Audit record of a single lifecycle change."""

    model_config = ConfigDict(frozen=True)

    from_status: TaskStatus
    to_status: TaskStatus
    at: datetime
    reason: str | None = None


class RunCheckpoint(BaseModel):
    """Everything a paused run needs to continue where it stopped.

    A run that pauses for approval has already spent steps, tokens, and tool
    calls. Throwing that away and restarting on approval would re-ask the model
    questions it has already answered, re-run the tools it has already run, and
    quietly reset the step budget — so a task could be paused and approved
    indefinitely without ever exhausting it. The checkpoint carries the
    conversation *and* the counters, so approval continues one run rather than
    starting a second.

    It lives on the task instead of in a separate store because approving,
    rejecting, and cancelling all decide the same thing — what this task does
    next — and the version check exists to make exactly one of them win. Two
    stores would allow a task in ``awaiting_approval`` with no conversation
    behind it, which is a state nothing can act on.
    """

    model_config = ConfigDict(frozen=True)

    messages: tuple[Message, ...] = Field(min_length=1)
    steps: int = Field(ge=1, description="Model calls already spent, against the run's budget.")
    tool_calls: int = Field(ge=0)
    usage: TokenUsage

    @property
    def pending_calls(self) -> tuple[ToolCall, ...]:
        """The tool calls the pause is holding back — the whole paused turn."""
        return self.messages[-1].tool_calls

    @model_validator(mode="after")
    def _ends_on_the_paused_turn(self) -> RunCheckpoint:
        # Resuming means answering the last assistant turn with tool results; a
        # checkpoint that does not end on one cannot be resumed into a valid
        # conversation, and the provider would reject it with an opaque 400.
        last = self.messages[-1]
        if last.role is not Role.ASSISTANT or not last.tool_calls:
            raise ValueError("a checkpoint must end with the assistant turn that requested tools")
        return self


class AgentTask(BaseModel):
    """A unit of agent work and its lifecycle state."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: UUID = Field(default_factory=uuid4)
    goal: str = Field(min_length=1, max_length=4000)
    requested_by: str = Field(min_length=1, max_length=256)
    status: TaskStatus = TaskStatus.PENDING
    version: int = Field(default=1, ge=1)
    created_at: datetime
    updated_at: datetime
    history: tuple[StatusTransition, ...] = ()
    checkpoint: RunCheckpoint | None = None

    @classmethod
    def create(cls, *, goal: str, requested_by: str, now: datetime | None = None) -> AgentTask:
        """Build a new pending task with consistent creation timestamps."""
        timestamp = now or _utcnow()
        return cls(goal=goal, requested_by=requested_by, created_at=timestamp, updated_at=timestamp)

    @model_validator(mode="after")
    def _checkpoint_belongs_to_a_paused_task(self) -> AgentTask:
        _check_checkpoint(self.status, self.checkpoint)
        return self

    @property
    def is_terminal(self) -> bool:
        return not ALLOWED_TRANSITIONS[self.status]

    def can_transition_to(self, target: TaskStatus) -> bool:
        return target in ALLOWED_TRANSITIONS[self.status]

    def transition_to(
        self,
        target: TaskStatus,
        *,
        reason: str | None = None,
        checkpoint: RunCheckpoint | None = None,
        now: datetime | None = None,
    ) -> AgentTask:
        """Return a copy of this task in ``target`` status.

        ``checkpoint`` is the paused run state the task should hold *after* the
        transition. It defaults to ``None``, which clears any existing one:
        every transition means the task moved on, and only a pause has something
        to resume. Making clearing the default rather than the exception is what
        keeps a resumed, cancelled, or finished task from carrying a stale
        conversation that a later approval could replay.

        Raises:
            InvalidTransitionError: if the lifecycle does not allow the change.
            ValueError: if a checkpoint is attached to a status that cannot hold
                one (anything but ``awaiting_approval``).
        """
        if not self.can_transition_to(target):
            raise InvalidTransitionError(self.status, target)
        # `model_copy` skips validation by design, so the invariant the model
        # validator guards on construction is checked here too.
        _check_checkpoint(target, checkpoint)

        timestamp = now or _utcnow()
        record = StatusTransition(
            from_status=self.status, to_status=target, at=timestamp, reason=reason
        )
        return self.model_copy(
            update={
                "status": target,
                "version": self.version + 1,
                "updated_at": timestamp,
                "history": (*self.history, record),
                "checkpoint": checkpoint,
            }
        )
