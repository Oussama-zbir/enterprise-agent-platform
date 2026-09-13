"""Tests for the in-memory task repository contract."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from enterprise_agent_platform.tasks.models import AgentTask, TaskStatus
from enterprise_agent_platform.tasks.repository import (
    ConcurrentUpdateError,
    InMemoryTaskRepository,
    TaskNotFoundError,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def make_task(goal: str = "goal", *, minutes: int = 0) -> AgentTask:
    return AgentTask.create(goal=goal, requested_by="u", now=T0 + timedelta(minutes=minutes))


async def test_add_then_get_round_trips() -> None:
    repository = InMemoryTaskRepository()
    task = make_task()

    await repository.add(task)

    assert await repository.get(task.id) == task


async def test_add_rejects_duplicate_id() -> None:
    repository = InMemoryTaskRepository()
    task = make_task()
    await repository.add(task)

    with pytest.raises(ValueError, match="already exists"):
        await repository.add(task)


async def test_get_unknown_task_raises() -> None:
    with pytest.raises(TaskNotFoundError):
        await InMemoryTaskRepository().get(uuid4())


async def test_update_unknown_task_raises() -> None:
    with pytest.raises(TaskNotFoundError):
        await InMemoryTaskRepository().update(make_task(), expected_version=1)


async def test_update_persists_when_version_matches() -> None:
    repository = InMemoryTaskRepository()
    task = make_task()
    await repository.add(task)

    running = task.transition_to(TaskStatus.RUNNING)
    await repository.update(running, expected_version=task.version)

    assert (await repository.get(task.id)).status is TaskStatus.RUNNING


async def test_stale_update_is_rejected_to_prevent_lost_updates() -> None:
    repository = InMemoryTaskRepository()
    task = make_task()
    await repository.add(task)
    running = task.transition_to(TaskStatus.RUNNING)
    await repository.update(running, expected_version=task.version)

    # Two actors read the same version and race: an approval request and a cancel.
    awaiting = running.transition_to(TaskStatus.AWAITING_APPROVAL)
    cancelled = running.transition_to(TaskStatus.CANCELLED)
    await repository.update(awaiting, expected_version=running.version)

    with pytest.raises(ConcurrentUpdateError) as exc_info:
        await repository.update(cancelled, expected_version=running.version)

    assert exc_info.value.actual_version == awaiting.version
    assert (await repository.get(task.id)).status is TaskStatus.AWAITING_APPROVAL


async def test_list_filters_by_status_orders_newest_first_and_limits() -> None:
    repository = InMemoryTaskRepository()
    oldest, middle, newest = (make_task(f"g{i}", minutes=i) for i in range(3))
    for task in (middle, newest, oldest):
        await repository.add(task)
    await repository.update(middle.transition_to(TaskStatus.RUNNING), expected_version=1)

    assert [t.id for t in await repository.list_tasks()] == [newest.id, middle.id, oldest.id]
    assert [t.id for t in await repository.list_tasks(limit=2)] == [newest.id, middle.id]
    pending = await repository.list_tasks(status=TaskStatus.PENDING)
    assert [t.id for t in pending] == [newest.id, oldest.id]
