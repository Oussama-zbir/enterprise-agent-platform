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

from pydantic import BaseModel, ConfigDict, Field


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


class StatusTransition(BaseModel):
    """Audit record of a single lifecycle change."""

    model_config = ConfigDict(frozen=True)

    from_status: TaskStatus
    to_status: TaskStatus
    at: datetime
    reason: str | None = None


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

    @classmethod
    def create(cls, *, goal: str, requested_by: str, now: datetime | None = None) -> AgentTask:
        """Build a new pending task with consistent creation timestamps."""
        timestamp = now or _utcnow()
        return cls(goal=goal, requested_by=requested_by, created_at=timestamp, updated_at=timestamp)

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
        now: datetime | None = None,
    ) -> AgentTask:
        """Return a copy of this task in ``target`` status.

        Raises:
            InvalidTransitionError: if the lifecycle does not allow the change.
        """
        if not self.can_transition_to(target):
            raise InvalidTransitionError(self.status, target)

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
            }
        )
