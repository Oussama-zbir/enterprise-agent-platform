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
from enterprise_agent_platform.auth.dependencies import ApprovalPolicyDep, requires
from enterprise_agent_platform.auth.models import Principal, Scope
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


class CancelTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(default=None, max_length=1000)


class ApproveTaskRequest(BaseModel):
    """Why the held tool calls are being released, optionally.

    Who is releasing them is not a field. An approval an unauthenticated caller
    can sign with any name is not an audit trail, so the approver is the
    authenticated subject and the body cannot contradict it — ``extra="forbid"``
    turns an attempt to send ``approved_by`` into a 422 rather than into a
    silently ignored claim.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    note: str | None = Field(default=None, max_length=1000)


class RejectTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

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

# Identity comes from the credential, never from the body. Every route below is
# closed: reading a task exposes the goal, the model's reasoning about it and
# the arguments of the tools it wanted to run, which is not less sensitive than
# starting one.
ReaderDep = Annotated[Principal, Depends(requires(Scope.TASKS_READ))]
WriterDep = Annotated[Principal, Depends(requires(Scope.TASKS_WRITE))]
ApproverDep = Annotated[Principal, Depends(requires(Scope.TASKS_APPROVE))]
# Seeing a decision is not making it, so either scope opens the approval view.
ApprovalViewerDep = Annotated[Principal, Depends(requires(Scope.TASKS_READ, Scope.TASKS_APPROVE))]


@router.post("", status_code=HTTPStatus.CREATED)
async def create_task(
    body: CreateTaskRequest, repository: RepositoryDep, principal: WriterDep
) -> TaskResponse:
    """Record a task on behalf of the authenticated caller.

    ``requested_by`` is taken from the credential rather than the body: it is
    half of the separation-of-duties check at the approval gate, so a caller
    that could name themselves could name someone else and approve their own
    task in two calls.
    """
    task = AgentTask.create(goal=body.goal, requested_by=principal.subject)
    await repository.add(task)
    logger.info("task.created", extra={"task_id": str(task.id)})
    return TaskResponse.from_domain(task)


@router.get("")
async def list_tasks(
    repository: RepositoryDep,
    principal: ReaderDep,
    status: TaskStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[TaskResponse]:
    tasks = await repository.list_tasks(status=status, limit=limit)
    return [TaskResponse.from_domain(task) for task in tasks]


@router.get("/{task_id}")
async def get_task(task_id: UUID, repository: RepositoryDep, principal: ReaderDep) -> TaskResponse:
    try:
        task = await repository.get(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    return TaskResponse.from_domain(task)


@router.post("/{task_id}/run")
async def run_task(task_id: UUID, runner: RunnerDep, principal: WriterDep) -> RunTaskResponse:
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
    task_id: UUID, repository: RepositoryDep, runner: RunnerDep, principal: ApprovalViewerDep
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
    task_id: UUID,
    runner: RunnerDep,
    repository: RepositoryDep,
    policy: ApprovalPolicyDep,
    principal: ApproverDep,
    body: ApproveTaskRequest | None = None,
) -> RunTaskResponse:
    """Authorise the held tool calls and continue the run.

    Like `/run`, the caller waits for the continuation. The approval itself is
    durable the moment the task transitions, so the run resumed here is the one
    the approver authorised — a second approver racing this one gets 409 rather
    than a second agent replaying the same critical calls.

    Holding `tasks:approve` is not sufficient: by default the approver must be
    someone other than the subject that requested the task. The check reads the
    task first, which is safe to do outside the versioned update because
    `requested_by` is set at creation and no transition ever rewrites it.
    """
    try:
        task = await repository.get(task_id)
    except TaskNotFoundError as exc:
        raise HTTPException(HTTPStatus.NOT_FOUND, detail=str(exc)) from exc

    if not policy.may_approve(principal, requested_by=task.requested_by):
        logger.warning(
            "task.approval.refused",
            extra={"task_id": str(task_id), "reason": "requester_is_approver"},
        )
        raise HTTPException(
            HTTPStatus.FORBIDDEN,
            detail="A task must be approved by someone other than the subject that requested it.",
        )

    try:
        result = await runner.resume(
            task_id, approved_by=principal.subject, note=body.note if body else None
        )
    except TaskNotFoundError as exc:
        raise HTTPException(HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except (InvalidTransitionError, ConcurrentUpdateError, NotResumableError) as exc:
        raise HTTPException(HTTPStatus.CONFLICT, detail=str(exc)) from exc

    return RunTaskResponse.from_domain(result)


@router.post("/{task_id}/reject")
async def reject_task(
    task_id: UUID,
    runner: RunnerDep,
    principal: ApproverDep,
    body: RejectTaskRequest | None = None,
) -> TaskResponse:
    """Deny the held tool calls; the task is cancelled, not retried.

    No separation-of-duties check: withholding a capability needs no second
    opinion, and requiring one would leave a requester who spotted their own
    mistake unable to stop the agent acting on it.
    """
    try:
        task = await runner.reject(
            task_id, rejected_by=principal.subject, reason=body.reason if body else None
        )
    except TaskNotFoundError as exc:
        raise HTTPException(HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except (InvalidTransitionError, ConcurrentUpdateError) as exc:
        raise HTTPException(HTTPStatus.CONFLICT, detail=str(exc)) from exc
    return TaskResponse.from_domain(task)


@router.post("/{task_id}/cancel")
async def cancel_task(
    task_id: UUID,
    repository: RepositoryDep,
    principal: WriterDep,
    body: CancelTaskRequest | None = None,
) -> TaskResponse:
    """Stop a task, recording who stopped it.

    Like approval and rejection, the actor comes from the credential: a history
    that names who asked and who approved but not who cancelled leaves the one
    transition anyone disputes unattributed. No separation-of-duties check —
    cancelling is withholding, and a requester must be able to stop their own
    task.
    """
    try:
        detail = f"cancelled by {principal.subject}"
        reason = body.reason if body else None
        task = await transition_task(
            repository,
            task_id,
            TaskStatus.CANCELLED,
            reason=f"{detail}: {reason}" if reason else detail,
        )
    except TaskNotFoundError as exc:
        raise HTTPException(HTTPStatus.NOT_FOUND, detail=str(exc)) from exc
    except (InvalidTransitionError, ConcurrentUpdateError) as exc:
        raise HTTPException(HTTPStatus.CONFLICT, detail=str(exc)) from exc
    return TaskResponse.from_domain(task)
