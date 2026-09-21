"""PostgreSQL adapter for the task repository port.

The in-memory adapter is atomic because every operation runs to completion
without an ``await`` point inside one event loop. Nothing about that survives a
restart or a second replica, and since Milestone 4 a paused task carries the
conversation an approval resumes from — so losing task state now means losing a
human decision's context, not just a row.

This adapter keeps the same contract with different machinery: the version
check that was a dict comparison becomes ``UPDATE ... WHERE version = $expected``,
which Postgres evaluates under a row lock, so two racing writers on two
connections still produce exactly one winner.

**Storage shape.** ``history`` and ``checkpoint`` are ``jsonb`` columns rather
than child tables. Both are only ever read with their task and never queried
across tasks, and keeping them inline means a transition is a single statement —
the same statement that performs the version check. Child tables would buy
queries nobody makes at the cost of a multi-statement write whose atomicity
would then depend on getting the transaction right.

**Typing.** asyncpg ships no type information, so pool and connection handles
are ``Any`` inside this module. That stops at the module boundary: everything
this class accepts and returns is a typed domain object.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg

from enterprise_agent_platform.tasks.models import AgentTask, TaskStatus
from enterprise_agent_platform.tasks.repository import ConcurrentUpdateError, TaskNotFoundError

# Generated from the domain enum so the database's notion of a valid status
# cannot drift from the state machine's.
_STATUS_VALUES = ", ".join(f"'{status.value}'" for status in TaskStatus)

SCHEMA_STATEMENTS: tuple[str, ...] = (
    f"""
    CREATE TABLE IF NOT EXISTS agent_tasks (
        id            uuid PRIMARY KEY,
        goal          text NOT NULL,
        requested_by  text NOT NULL,
        status        text NOT NULL CHECK (status IN ({_STATUS_VALUES})),
        version       integer NOT NULL CHECK (version >= 1),
        created_at    timestamptz NOT NULL,
        updated_at    timestamptz NOT NULL,
        history       jsonb NOT NULL,
        checkpoint    jsonb,
        -- The domain forbids a task that is not paused from carrying run state
        -- (a stale conversation is a replay hazard). Restating it here means a
        -- bug in a future writer fails at the database rather than leaving a
        -- row the domain model refuses to load.
        CONSTRAINT agent_tasks_checkpoint_only_when_paused
            CHECK (checkpoint IS NULL OR status = 'awaiting_approval')
    )
    """,
    # Matches the list query's sort exactly, including the id tiebreaker that
    # makes the order total (and therefore paginable).
    """
    CREATE INDEX IF NOT EXISTS agent_tasks_created_at_idx
        ON agent_tasks (created_at DESC, id DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS agent_tasks_status_created_at_idx
        ON agent_tasks (status, created_at DESC, id DESC)
    """,
)

_INSERT = """
    INSERT INTO agent_tasks
        (id, goal, requested_by, status, version, created_at, updated_at, history, checkpoint)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
"""

_SELECT_COLUMNS = """
    id, goal, requested_by, status, version, created_at, updated_at, history, checkpoint
"""

_SELECT_BY_ID = f"SELECT {_SELECT_COLUMNS} FROM agent_tasks WHERE id = $1"

# Only the columns a transition can change are written: id, goal, requested_by
# and created_at are immutable on the domain entity, so leaving them out of the
# statement makes that structural rather than a convention to remember.
_UPDATE_IF_VERSION_MATCHES = """
    UPDATE agent_tasks
       SET status = $2, version = $3, updated_at = $4, history = $5, checkpoint = $6
     WHERE id = $1 AND version = $7
    RETURNING version
"""

_SELECT_ALL = f"""
    SELECT {_SELECT_COLUMNS} FROM agent_tasks
     ORDER BY created_at DESC, id DESC
     LIMIT $1
"""

_SELECT_BY_STATUS = f"""
    SELECT {_SELECT_COLUMNS} FROM agent_tasks
     WHERE status = $1
     ORDER BY created_at DESC, id DESC
     LIMIT $2
"""


async def apply_schema(connection: Any) -> None:
    """Create the table and indexes if they are absent.

    The statements are idempotent, which is enough for local development and
    tests. A real deployment runs schema changes as a migration with its own
    review and its own database role — application startup is the wrong place to
    hold DDL privileges — so this is never called implicitly.
    """
    for statement in SCHEMA_STATEMENTS:
        await connection.execute(statement)


class PostgresTaskRepository:
    """``TaskRepository`` backed by Postgres over an asyncpg pool.

    The pool is opened in ``connect`` rather than ``__init__`` because it needs
    a running event loop and because the application must be able to construct
    its wiring before startup. ``aclose`` returns the connections at shutdown.
    """

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
        command_timeout: float = 10.0,
    ) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._command_timeout = command_timeout
        self._pool: Any = None

    async def connect(self, *, create_schema: bool = False) -> None:
        """Open the connection pool.

        ``command_timeout`` bounds every statement: a task write blocked behind
        a lock fails the request instead of holding an API worker until the
        client gives up.
        """
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            command_timeout=self._command_timeout,
        )
        if create_schema:
            async with self._acquire() as connection:
                await apply_schema(connection)

    async def aclose(self) -> None:
        if self._pool is None:
            return
        pool, self._pool = self._pool, None
        await pool.close()

    def _acquire(self) -> Any:
        if self._pool is None:
            raise RuntimeError("repository is not connected; call connect() first")
        return self._pool.acquire()

    async def add(self, task: AgentTask) -> None:
        async with self._acquire() as connection:
            try:
                await connection.execute(
                    _INSERT,
                    task.id,
                    task.goal,
                    task.requested_by,
                    task.status.value,
                    task.version,
                    task.created_at,
                    task.updated_at,
                    _dump(_history_of(task)),
                    _dump_optional(_checkpoint_of(task)),
                )
            except asyncpg.UniqueViolationError:
                # Same contract as the in-memory adapter: creating a task twice
                # is a caller bug, not a concurrency outcome to retry.
                raise ValueError(f"task '{task.id}' already exists") from None

    async def get(self, task_id: UUID) -> AgentTask:
        async with self._acquire() as connection:
            row: Any = await connection.fetchrow(_SELECT_BY_ID, task_id)
        if row is None:
            raise TaskNotFoundError(task_id)
        return _to_task(row)

    async def update(self, task: AgentTask, *, expected_version: int) -> None:
        """Persist ``task`` only if the stored row is still at ``expected_version``.

        The check and the write are one statement, so the decision is made under
        the row lock and exactly one of two racing transitions can win. The
        follow-up read exists only to explain a rejection: by the time it runs
        the row may have moved again, so ``actual_version`` is diagnostic. What
        is guaranteed is that this write did not land.
        """
        async with self._acquire() as connection:
            updated: Any = await connection.fetchval(
                _UPDATE_IF_VERSION_MATCHES,
                task.id,
                task.status.value,
                task.version,
                task.updated_at,
                _dump(_history_of(task)),
                _dump_optional(_checkpoint_of(task)),
                expected_version,
            )
            if updated is not None:
                return
            actual: Any = await connection.fetchval(
                "SELECT version FROM agent_tasks WHERE id = $1", task.id
            )
        if actual is None:
            raise TaskNotFoundError(task.id)
        raise ConcurrentUpdateError(task.id, expected_version, int(actual))

    async def list_tasks(
        self, *, status: TaskStatus | None = None, limit: int = 50
    ) -> list[AgentTask]:
        async with self._acquire() as connection:
            rows: list[Any] = (
                await connection.fetch(_SELECT_ALL, limit)
                if status is None
                else await connection.fetch(_SELECT_BY_STATUS, status.value, limit)
            )
        return [_to_task(row) for row in rows]


async def _create_schema_from_settings() -> None:
    """``python -m enterprise_agent_platform.tasks.postgres`` — create the schema.

    A convenience for local development and CI, kept out of application startup
    so the service itself never needs DDL privileges. A real deployment runs a
    migration tool here instead.
    """
    from enterprise_agent_platform.config import get_settings

    settings = get_settings()
    if settings.database_url is None:
        raise SystemExit("EAP_DATABASE_URL is not set")
    connection = await asyncpg.connect(settings.database_url.get_secret_value())
    try:
        await apply_schema(connection)
    finally:
        await connection.close()


def _history_of(task: AgentTask) -> list[dict[str, Any]]:
    return [transition.model_dump(mode="json") for transition in task.history]


def _checkpoint_of(task: AgentTask) -> dict[str, Any] | None:
    return None if task.checkpoint is None else task.checkpoint.model_dump(mode="json")


def _dump(value: list[dict[str, Any]] | dict[str, Any]) -> str:
    # asyncpg passes jsonb parameters as text unless a codec is registered;
    # serialising here keeps the encoding explicit and local to this module.
    return json.dumps(value)


def _dump_optional(value: dict[str, Any] | None) -> str | None:
    return None if value is None else _dump(value)


def _to_task(row: Any) -> AgentTask:
    """Rebuild a task from a row, validating it through the domain model.

    Rows are validated rather than trusted: a task written by an older version
    of the service, or edited by hand, must still satisfy the state machine's
    invariants before anything acts on it.
    """
    checkpoint: str | None = row["checkpoint"]
    return AgentTask.model_validate(
        {
            "id": row["id"],
            "goal": row["goal"],
            "requested_by": row["requested_by"],
            "status": row["status"],
            "version": row["version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "history": json.loads(row["history"]),
            "checkpoint": None if checkpoint is None else json.loads(checkpoint),
        }
    )


if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(_create_schema_from_settings())
