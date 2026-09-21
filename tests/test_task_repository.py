"""Contract tests every task repository adapter must satisfy.

The port only pays for itself if swapping the adapter cannot change behaviour,
so the in-memory and PostgreSQL adapters run the same tests rather than each
having its own. Anything an adapter is allowed to differ on is not in here.

The Postgres parameter is skipped unless ``EAP_TEST_DATABASE_URL`` names a
database to use, so the default suite stays offline; CI runs it against a real
server. Stubbing the driver instead would only test the stub — the behaviour
under test (``UPDATE ... WHERE version = $n`` resolving a race under a row
lock) is the database's, not ours.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from enterprise_agent_platform.llm.models import Message, Role, TokenUsage, ToolCall
from enterprise_agent_platform.tasks.models import AgentTask, RunCheckpoint, TaskStatus
from enterprise_agent_platform.tasks.repository import (
    ConcurrentUpdateError,
    InMemoryTaskRepository,
    TaskNotFoundError,
    TaskRepository,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
POSTGRES_DSN = os.environ.get("EAP_TEST_DATABASE_URL")


def make_task(goal: str = "goal", *, minutes: int = 0) -> AgentTask:
    return AgentTask.create(goal=goal, requested_by="u", now=T0 + timedelta(minutes=minutes))


def make_checkpoint() -> RunCheckpoint:
    """A paused turn: the conversation plus the budget already spent."""
    return RunCheckpoint(
        messages=(
            Message(role=Role.USER, content="Settle invoice INV-1"),
            Message(
                role=Role.ASSISTANT,
                content="Paying it now.",
                tool_calls=(
                    ToolCall(id="call_1", name="pay_invoice", arguments={"invoice_id": "INV-1"}),
                ),
            ),
        ),
        steps=2,
        tool_calls=1,
        usage=TokenUsage(input_tokens=812, output_tokens=96),
    )


async def _postgres_repository() -> AsyncIterator[TaskRepository]:
    if not POSTGRES_DSN:
        pytest.skip("set EAP_TEST_DATABASE_URL to run the PostgreSQL contract tests")
    asyncpg = pytest.importorskip("asyncpg")
    from enterprise_agent_platform.tasks.postgres import PostgresTaskRepository

    repository = PostgresTaskRepository(POSTGRES_DSN, max_size=4)
    await repository.connect(create_schema=True)
    # Each test starts from an empty table: these tests assert on listings, so a
    # leftover row from a previous test would make failures depend on order.
    connection = await asyncpg.connect(POSTGRES_DSN)
    try:
        await connection.execute("TRUNCATE agent_tasks")
    finally:
        await connection.close()
    try:
        yield repository
    finally:
        await repository.aclose()


@pytest.fixture(params=["memory", "postgres"])
async def repository(request: pytest.FixtureRequest) -> AsyncIterator[TaskRepository]:
    if request.param == "memory":
        yield InMemoryTaskRepository()
        return
    async for postgres in _postgres_repository():
        yield postgres


async def test_add_then_get_round_trips(repository: TaskRepository) -> None:
    task = make_task()

    await repository.add(task)

    assert await repository.get(task.id) == task


async def test_add_rejects_duplicate_id(repository: TaskRepository) -> None:
    task = make_task()
    await repository.add(task)

    with pytest.raises(ValueError, match="already exists"):
        await repository.add(task)


async def test_get_unknown_task_raises(repository: TaskRepository) -> None:
    with pytest.raises(TaskNotFoundError):
        await repository.get(uuid4())


async def test_update_unknown_task_raises(repository: TaskRepository) -> None:
    with pytest.raises(TaskNotFoundError):
        await repository.update(make_task(), expected_version=1)


async def test_update_persists_when_version_matches(repository: TaskRepository) -> None:
    task = make_task()
    await repository.add(task)

    running = task.transition_to(TaskStatus.RUNNING)
    await repository.update(running, expected_version=task.version)

    assert (await repository.get(task.id)).status is TaskStatus.RUNNING


async def test_stale_update_is_rejected_to_prevent_lost_updates(
    repository: TaskRepository,
) -> None:
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


async def test_concurrent_writers_on_one_version_produce_exactly_one_winner(
    repository: TaskRepository,
) -> None:
    """The version check must hold when the two writes actually overlap.

    The sequential test above cannot catch a check-then-write implemented as two
    statements; issued together on separate connections, that would let both
    approvals through and run the held tool calls twice.
    """
    task = make_task()
    await repository.add(task)
    running = task.transition_to(TaskStatus.RUNNING)
    await repository.update(running, expected_version=task.version)

    outcomes = await asyncio.gather(
        repository.update(
            running.transition_to(TaskStatus.COMPLETED, reason="approver A"),
            expected_version=running.version,
        ),
        repository.update(
            running.transition_to(TaskStatus.CANCELLED, reason="approver B"),
            expected_version=running.version,
        ),
        return_exceptions=True,
    )

    rejected = [o for o in outcomes if isinstance(o, ConcurrentUpdateError)]
    assert len(rejected) == 1, outcomes
    stored = await repository.get(task.id)
    assert stored.version == running.version + 1
    assert stored.is_terminal


async def test_checkpoint_and_history_survive_a_round_trip(repository: TaskRepository) -> None:
    """A paused run must come back exactly as it was stored.

    This is the whole point of a durable store: an approval that arrives after a
    restart resumes one run rather than starting a second. Comparing the full
    task also covers the audit history and the timestamps.
    """
    task = make_task()
    await repository.add(task)
    running = task.transition_to(TaskStatus.RUNNING)
    await repository.update(running, expected_version=task.version)
    awaiting = running.transition_to(
        TaskStatus.AWAITING_APPROVAL,
        reason="approval required for: pay_invoice",
        checkpoint=make_checkpoint(),
    )

    await repository.update(awaiting, expected_version=running.version)

    stored = await repository.get(task.id)
    assert stored == awaiting
    assert stored.checkpoint is not None
    assert stored.checkpoint.pending_calls[0].arguments == {"invoice_id": "INV-1"}
    assert [t.to_status for t in stored.history] == [
        TaskStatus.RUNNING,
        TaskStatus.AWAITING_APPROVAL,
    ]


async def test_resuming_clears_the_stored_checkpoint(repository: TaskRepository) -> None:
    """No finished or resumed task may keep a replayable conversation behind."""
    task = make_task()
    await repository.add(task)
    running = task.transition_to(TaskStatus.RUNNING)
    await repository.update(running, expected_version=task.version)
    awaiting = running.transition_to(TaskStatus.AWAITING_APPROVAL, checkpoint=make_checkpoint())
    await repository.update(awaiting, expected_version=running.version)

    resumed = awaiting.transition_to(TaskStatus.RUNNING)
    await repository.update(resumed, expected_version=awaiting.version)

    assert (await repository.get(task.id)).checkpoint is None


async def test_list_filters_by_status_orders_newest_first_and_limits(
    repository: TaskRepository,
) -> None:
    oldest, middle, newest = (make_task(f"g{i}", minutes=i) for i in range(3))
    for task in (middle, newest, oldest):
        await repository.add(task)
    await repository.update(middle.transition_to(TaskStatus.RUNNING), expected_version=1)

    assert [t.id for t in await repository.list_tasks()] == [newest.id, middle.id, oldest.id]
    assert [t.id for t in await repository.list_tasks(limit=2)] == [newest.id, middle.id]
    pending = await repository.list_tasks(status=TaskStatus.PENDING)
    assert [t.id for t in pending] == [newest.id, oldest.id]


async def test_list_breaks_timestamp_ties_on_id_for_a_stable_order(
    repository: TaskRepository,
) -> None:
    """Tasks created in the same instant must still have one fixed order.

    Two tasks can share a `created_at` under load; if their relative order can
    change between calls, a client paging through the list can miss or repeat
    one. Both adapters therefore sort by (created_at, id).
    """
    same_instant = [make_task(f"g{i}") for i in range(4)]
    for task in same_instant:
        await repository.add(task)

    listed = [t.id for t in await repository.list_tasks()]

    assert listed == sorted((t.id for t in same_instant), reverse=True)
    assert listed == [t.id for t in await repository.list_tasks()]


async def test_repository_rejects_use_before_it_is_connected() -> None:
    """A pooled adapter fails loudly rather than lazily opening a pool mid-request."""
    pytest.importorskip("asyncpg")
    from enterprise_agent_platform.tasks.postgres import PostgresTaskRepository

    repository = PostgresTaskRepository("postgresql://unused/unused")

    with pytest.raises(RuntimeError, match="not connected"):
        await repository.get(uuid4())


async def test_database_refuses_a_checkpoint_on_a_task_that_is_not_paused() -> None:
    """The domain invariant is also a table constraint.

    The model refuses to build such a task, so this writes the row directly:
    the point is that a future writer bypassing the model still cannot leave a
    stale conversation on a finished task.
    """
    if not POSTGRES_DSN:
        pytest.skip("set EAP_TEST_DATABASE_URL to run the PostgreSQL contract tests")
    asyncpg = pytest.importorskip("asyncpg")
    from enterprise_agent_platform.tasks.postgres import PostgresTaskRepository, apply_schema

    repository = PostgresTaskRepository(POSTGRES_DSN)
    await repository.connect(create_schema=True)
    await repository.aclose()

    connection = await asyncpg.connect(POSTGRES_DSN)
    try:
        await apply_schema(connection)
        with pytest.raises(asyncpg.CheckViolationError):
            await connection.execute(
                """
                INSERT INTO agent_tasks (id, goal, requested_by, status, version,
                                         created_at, updated_at, history, checkpoint)
                VALUES ($1, 'g', 'u', 'completed', 2, $2, $2, '[]'::jsonb, '{}'::jsonb)
                """,
                uuid4(),
                T0,
            )
    finally:
        await connection.close()
