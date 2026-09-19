"""Task use cases shared by the HTTP layer and the future orchestrator."""

from __future__ import annotations

import logging
from uuid import UUID

from enterprise_agent_platform.tasks.models import AgentTask, RunCheckpoint, TaskStatus
from enterprise_agent_platform.tasks.repository import ConcurrentUpdateError, TaskRepository

logger = logging.getLogger(__name__)


async def transition_task(
    repository: TaskRepository,
    task_id: UUID,
    target: TaskStatus,
    *,
    reason: str | None = None,
    checkpoint: RunCheckpoint | None = None,
    expected_version: int | None = None,
) -> AgentTask:
    """Load a task, apply a lifecycle transition, and persist it atomically.

    ``checkpoint`` is the paused run state to store with the new status; leaving
    it unset clears whatever the task was holding.

    ``expected_version`` extends the repository's version check back to a read
    the caller made earlier. Resuming a paused run reads the checkpoint first
    and transitions second; without this guard the transition would re-read a
    task that had meanwhile been approved by someone else and happily resume
    from a conversation that is no longer current.

    Raises:
        TaskNotFoundError: if the task does not exist.
        InvalidTransitionError: if the lifecycle forbids the change.
        ConcurrentUpdateError: if the task changed after it was loaded, or since
            ``expected_version`` was read.
    """
    current = await repository.get(task_id)
    if expected_version is not None and current.version != expected_version:
        raise ConcurrentUpdateError(task_id, expected_version, current.version)
    updated = current.transition_to(target, reason=reason, checkpoint=checkpoint)
    await repository.update(updated, expected_version=current.version)

    logger.info(
        "task.transitioned",
        extra={
            "task_id": str(task_id),
            "from_status": current.status.value,
            "to_status": target.value,
            "version": updated.version,
        },
    )
    return updated
