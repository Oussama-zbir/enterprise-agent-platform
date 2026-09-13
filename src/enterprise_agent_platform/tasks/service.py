"""Task use cases shared by the HTTP layer and the future orchestrator."""

from __future__ import annotations

import logging
from uuid import UUID

from enterprise_agent_platform.tasks.models import AgentTask, TaskStatus
from enterprise_agent_platform.tasks.repository import TaskRepository

logger = logging.getLogger(__name__)


async def transition_task(
    repository: TaskRepository,
    task_id: UUID,
    target: TaskStatus,
    *,
    reason: str | None = None,
) -> AgentTask:
    """Load a task, apply a lifecycle transition, and persist it atomically.

    Raises:
        TaskNotFoundError: if the task does not exist.
        InvalidTransitionError: if the lifecycle forbids the change.
        ConcurrentUpdateError: if the task changed after it was loaded.
    """
    current = await repository.get(task_id)
    updated = current.transition_to(target, reason=reason)
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
