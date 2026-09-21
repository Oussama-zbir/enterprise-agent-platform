"""Task persistence.

``TaskRepository`` is the storage port used by the API and, later, the agent
orchestrator. Updates use optimistic concurrency: callers pass the version they
read, and the write is rejected if the stored task has moved on. This prevents
lost updates when, for example, a human approval and a cancellation race.

``InMemoryTaskRepository`` is the reference adapter for local development and
tests; a database-backed adapter can replace it without changing callers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from enterprise_agent_platform.tasks.models import AgentTask, TaskStatus


class TaskNotFoundError(Exception):
    def __init__(self, task_id: UUID) -> None:
        super().__init__(f"task '{task_id}' not found")
        self.task_id = task_id


class ConcurrentUpdateError(Exception):
    """Raised when a task changed between being read and being written."""

    def __init__(self, task_id: UUID, expected_version: int, actual_version: int) -> None:
        super().__init__(
            f"task '{task_id}' was modified concurrently "
            f"(expected version {expected_version}, found {actual_version})"
        )
        self.task_id = task_id
        self.expected_version = expected_version
        self.actual_version = actual_version


class TaskRepository(Protocol):
    async def add(self, task: AgentTask) -> None: ...

    async def get(self, task_id: UUID) -> AgentTask: ...

    async def update(self, task: AgentTask, *, expected_version: int) -> None: ...

    async def list_tasks(
        self, *, status: TaskStatus | None = None, limit: int = 50
    ) -> list[AgentTask]: ...


@runtime_checkable
class ManagedRepository(Protocol):
    """A repository that owns a resource the application must open and close.

    Kept separate from ``TaskRepository`` so the port stays about storage: a
    dict needs no connection pool, and callers doing task work should not have
    to care which kind they were handed. Only the application lifespan checks
    for this.
    """

    async def connect(self) -> None: ...

    async def aclose(self) -> None: ...


class InMemoryTaskRepository:
    """Dict-backed repository.

    Each operation completes without an ``await`` point, so it is atomic within
    a single event loop. The version check still enforces the same contract a
    database adapter must honour (e.g. ``UPDATE ... WHERE version = :expected``).
    """

    def __init__(self) -> None:
        self._tasks: dict[UUID, AgentTask] = {}

    async def add(self, task: AgentTask) -> None:
        if task.id in self._tasks:
            raise ValueError(f"task '{task.id}' already exists")
        self._tasks[task.id] = task

    async def get(self, task_id: UUID) -> AgentTask:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(task_id) from None

    async def update(self, task: AgentTask, *, expected_version: int) -> None:
        current = self._tasks.get(task.id)
        if current is None:
            raise TaskNotFoundError(task.id)
        if current.version != expected_version:
            raise ConcurrentUpdateError(task.id, expected_version, current.version)
        self._tasks[task.id] = task

    async def list_tasks(
        self, *, status: TaskStatus | None = None, limit: int = 50
    ) -> list[AgentTask]:
        matching = (t for t in self._tasks.values() if status is None or t.status is status)
        # The id breaks ties on identical timestamps so the order is total and
        # matches the SQL adapter's `ORDER BY created_at DESC, id DESC`. Without
        # it, two tasks created in the same instant could swap places between
        # calls — which is exactly what makes keyset pagination unsound.
        return sorted(matching, key=lambda t: (t.created_at, t.id), reverse=True)[:limit]
