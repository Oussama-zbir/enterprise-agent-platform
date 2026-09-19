"""HTTP routes for agent tasks.

API schemas are kept separate from the domain model so the persisted shape and
the public contract can evolve independently.
"""

from __future__ import annotations

import logging
from datetime import datetime
from http import HTTPStatus
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from enterprise_agent_platform.agent.runner import (
    AgentRunner,
    AgentRunResult,
    NotResumableError,
    PendingToolCall,
)
from enterprise_agent_platform.request_context import get_request_id
from enterprise_agent_platform.tasks.models import (
    AgentTask,
    InvalidTransitionError,
    StatusTransition,
    TaskStatus,
)
from enterprise_agent_platform.tasks.repository import (
    ConcurrentUpdateError,
    TaskNotFoundError,
    TaskRepository,
)
from enterprise_agent_platform.tasks.service import transition_task
from enterprise_agent_platform.tools.models import RiskLevel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    goal: str = Field(min_length=1, max_length=4000)
    requested_by: str = Field(min_length=1, max_length=256)


class CancelTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(default=None, max_length=1000)


class ApproveTaskRequest(BaseModel):
    """Who is authorising the held tool calls, and optionally why.

    ``approved_by`` is required: an approval with no one attached to it is not
    an audit trail. Like ``requested_by`` it is unverified until the platform has
    authentication, which is why it is recorded rather than trusted — and never
    put in front of the model.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    approved_by: str = Field(min_length=1, max_length=256)
    note: str | None = Field(default=None, max_length=1000)


class RejectTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    rejected_by: str = Field(min_length=1, max_length=256)
    reason: str | None = Field(default=None, max_length=1000)


class PendingToolCallResponse(BaseModel):
    """One held tool call, with enough detail to decide on it.

    The arguments are included because approving a tool by name alone is
    theatre: paying *an* invoice and paying *this* invoice are different
    decisions.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    risk: RiskLevel | None = Field(default=None, description="null if the tool is no longer known")
    needs_approval: bool

    @classmethod
    def from_domain(cls, call: PendingToolCall) -> PendingToolCallResponse:
        return cls(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            risk=call.risk,
            needs_approval=call.needs_approval,
        )


class ApprovalResponse(BaseModel):
    """What a human is being asked to decide."""

    task_id: UUID
    goal: str
    status: TaskStatus
    paused_at: datetime
    detail: str | None = Field(default=None, description="Why the run paused.")
    pending_tool_calls: list[PendingToolCallResponse]

    @classmethod
    def from_domain(cls, task: AgentTask, calls: tuple[PendingToolCall, ...]) -> ApprovalResponse:
        return cls(
            task_id=task.id,
            goal=task.goal,
            status=task.status,
            paused_at=task.updated_at,
            detail=task.history[-1].reason if task.history else None,
            pending_tool_calls=[PendingToolCallResponse.from_domain(call) for call in calls],
        )


class TaskResponse(BaseModel):
    id: UUID
    goal: str
    requested_by: str
    status: TaskStatus
    is_terminal: bool
    version: int
    created_at: datetime
    updated_at: datetime
    history: list[StatusTransition]

    @classmethod
    def from_domain(cls, task: AgentTask) -> TaskResponse:
        # The task's run checkpoint is deliberately not a field here: it carries
        # model output and tool results, and the only part anyone needs to act on
        # is served, in a shaped form, by the approval route.
        return cls(**task.model_dump(), is_terminal=task.is_terminal)


class RunTaskResponse(BaseModel):
    """The outcome of one agent run.

    The task is the durable record; the rest is run telemetry the caller would
    otherwise have to dig out of the logs. ``pending_tool_calls`` names the tools
    a paused run wants to use; ``GET /tasks/{id}/approval`` shows their
    arguments and risk to whoever has to decide. Counters are cumulative over
    the run, so a run that paused and was approved reports what it spent in
    total rather than restarting them.
    """

    task: TaskResponse
    output: str
    detail: str
    steps: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    pending_tool_calls: list[str] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, result: AgentRunResult) -> RunTaskResponse:
        return cls(
            task=TaskResponse.from_domain(result.task),
            output=result.output,
            detail=result.detail,
            steps=result.steps,
            tool_calls=result.tool_calls,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            pending_tool_calls=[call.name for call in result.pending_calls],
        )


def get_task_repository(request: Request) -> TaskRepository:
    repository: TaskRepository = request.app.state.task_repository
    return repository


def get_agent_runner(request: Request) -> AgentRunner:
    runner: AgentRunner = request.app.state.agent_runner
    return runner


RepositoryDep = Annotated[TaskRepository, Depends(get_task_repository)]
RunnerDep = Annotated[AgentRunner, Depends(get_agent_runner)]


@router.post("", status_code=HTTPStatus.CREATED)
async def create_task(body: CreateTaskRequest, repository: RepositoryDep) -> TaskResponse:
    task = AgentTask.create(goal=body.goal, requested_by=body.requested_by)
    await repository.add(task)
    logger.info("task.created", extra={"task_id": str(task.id)})
    return TaskResponse.from_domain(task)


@router.get("")
async def list_tasks(
    repository: RepositoryDep,
    status: TaskStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[TaskResponse]:
    tasks = await repository.list_tasks(status=status, limit=limit)
    return [TaskResponse.from_domain(task) for task in tasks]


@router.get("/{task_id}")
async def get_task(task_id: UUID, repository: RepositoryDep) -> TaskResponse:
    try:
        task = await repository.get(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    return TaskResponse.from_domain(task)


@router.post("/{task_id}/run")
async def run_task(task_id: UUID, runner: RunnerDep) -> RunTaskResponse:
    """Execute a pending task with the agent loop.

    The caller waits for the run to finish. That is honest for the current
    single-process deployment and keeps the API easy to reason about; because
    progress is recorded on the task rather than in this response, moving
    execution to a queue and returning 202 is a change of entrypoint, not of
    domain logic. A task that is already running or finished returns 409 rather
    than starting a second agent on the same goal.
    """
    try:
        result = await runner.run(task_id, request_id=get_request_id())
    except TaskNotFoundError as exc:
        raise HTTPException(HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except (InvalidTransitionError, ConcurrentUpdateError) as exc:
        raise HTTPException(HTTPStatus.CONFLICT, detail=str(exc)) from exc

    return RunTaskResponse.from_domain(result)


@router.get("/{task_id}/approval")
async def get_approval(
    task_id: UUID, repository: RepositoryDep, runner: RunnerDep
) -> ApprovalResponse:
    """Show what a paused run is waiting to do.

    Separate from `GET /tasks/{id}` because the paused conversation is not part
    of the task resource: it holds model output and tool results, and the only
    slice of it anyone needs is the calls awaiting a decision.
    """
    try:
        task = await repository.get(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    if task.status is not TaskStatus.AWAITING_APPROVAL:
        raise HTTPException(
            HTTPStatus.CONFLICT,
            detail=f"task '{task_id}' is '{task.status}', not awaiting approval",
        )
    return ApprovalResponse.from_domain(task, runner.pending_approval(task))


@router.post("/{task_id}/approve")
async def approve_task(
    task_id: UUID, body: ApproveTaskRequest, runner: RunnerDep
) -> RunTaskResponse:
    """Authorise the held tool calls and continue the run.

    Like `/run`, the caller waits for the continuation. The approval itself is
    durable the moment the task transitions, so the run resumed here is the one
    the approver authorised — a second approver racing this one gets 409 rather
    than a second agent replaying the same critical calls.
    """
    try:
        result = await runner.resume(task_id, approved_by=body.approved_by, note=body.note)
    except TaskNotFoundError as exc:
        raise HTTPException(HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except (InvalidTransitionError, ConcurrentUpdateError, NotResumableError) as exc:
        raise HTTPException(HTTPStatus.CONFLICT, detail=str(exc)) from exc

    return RunTaskResponse.from_domain(result)


@router.post("/{task_id}/reject")
async def reject_task(task_id: UUID, body: RejectTaskRequest, runner: RunnerDep) -> TaskResponse:
    """Deny the held tool calls; the task is cancelled, not retried."""
    try:
        task = await runner.reject(task_id, rejected_by=body.rejected_by, reason=body.reason)
    except TaskNotFoundError as exc:
        raise HTTPException(HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except (InvalidTransitionError, ConcurrentUpdateError) as exc:
        raise HTTPException(HTTPStatus.CONFLICT, detail=str(exc)) from exc
    return TaskResponse.from_domain(task)


@router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: UUID,
    repository: RepositoryDep,
    body: CancelTaskRequest | None = None,
) -> TaskResponse:
    try:
        task = await transition_task(
            repository,
            task_id,
            TaskStatus.CANCELLED,
            reason=body.reason if body else None,
        )
    except TaskNotFoundError as exc:
        raise HTTPException(HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except (InvalidTransitionError, ConcurrentUpdateError) as exc:
        raise HTTPException(HTTPStatus.CONFLICT, detail=str(exc)) from exc
    return TaskResponse.from_domain(task)
