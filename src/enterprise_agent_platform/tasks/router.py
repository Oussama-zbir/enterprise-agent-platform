"""HTTP routes for agent tasks.

API schemas are kept separate from the domain model so the persisted shape and
the public contract can evolve independently.
"""

from __future__ import annotations

import logging
from datetime import datetime
from http import HTTPStatus
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from enterprise_agent_platform.agent.runner import AgentRunner
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    goal: str = Field(min_length=1, max_length=4000)
    requested_by: str = Field(min_length=1, max_length=256)


class CancelTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str | None = Field(default=None, max_length=1000)


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
        return cls(**task.model_dump(), is_terminal=task.is_terminal)


class RunTaskResponse(BaseModel):
    """The outcome of one agent run.

    The task is the durable record; the rest is run telemetry the caller would
    otherwise have to dig out of the logs. ``pending_tool_calls`` names the tools
    a paused run wants to use — their arguments are surfaced by the approval API
    in the next milestone, where a human can act on them.
    """

    task: TaskResponse
    output: str
    detail: str
    steps: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    pending_tool_calls: list[str] = Field(default_factory=list)


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

    return RunTaskResponse(
        task=TaskResponse.from_domain(result.task),
        output=result.output,
        detail=result.detail,
        steps=result.steps,
        tool_calls=result.tool_calls,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        pending_tool_calls=[call.name for call in result.pending_calls],
    )


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
