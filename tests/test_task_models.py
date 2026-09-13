"""Tests for the agent task lifecycle state machine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from enterprise_agent_platform.tasks.models import (
    ALLOWED_TRANSITIONS,
    AgentTask,
    InvalidTransitionError,
    TaskStatus,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def make_task() -> AgentTask:
    return AgentTask.create(goal="Summarise open invoices", requested_by="analyst-1", now=T0)


def test_create_starts_pending_with_consistent_timestamps() -> None:
    task = make_task()

    assert task.status is TaskStatus.PENDING
    assert task.version == 1
    assert task.created_at == task.updated_at == T0
    assert task.history == ()
    assert not task.is_terminal


def test_goal_is_stripped_and_blank_goal_rejected() -> None:
    assert AgentTask.create(goal="  do it  ", requested_by="u").goal == "do it"

    with pytest.raises(ValidationError):
        AgentTask.create(goal="   ", requested_by="u")


def test_approval_lifecycle_records_history_and_versions() -> None:
    t1, t2, t3, t4 = (T0 + timedelta(minutes=i) for i in range(1, 5))

    task = (
        make_task()
        .transition_to(TaskStatus.RUNNING, now=t1)
        .transition_to(TaskStatus.AWAITING_APPROVAL, reason="refund > limit", now=t2)
        .transition_to(TaskStatus.RUNNING, reason="approved", now=t3)
        .transition_to(TaskStatus.COMPLETED, now=t4)
    )

    assert task.status is TaskStatus.COMPLETED
    assert task.is_terminal
    assert task.version == 5
    assert task.updated_at == t4
    assert task.created_at == T0
    assert [(h.from_status, h.to_status) for h in task.history] == [
        (TaskStatus.PENDING, TaskStatus.RUNNING),
        (TaskStatus.RUNNING, TaskStatus.AWAITING_APPROVAL),
        (TaskStatus.AWAITING_APPROVAL, TaskStatus.RUNNING),
        (TaskStatus.RUNNING, TaskStatus.COMPLETED),
    ]
    assert task.history[1].reason == "refund > limit"


def test_transition_does_not_mutate_original() -> None:
    task = make_task()

    task.transition_to(TaskStatus.RUNNING)

    assert task.status is TaskStatus.PENDING
    assert task.version == 1


def test_task_is_immutable() -> None:
    task = make_task()

    with pytest.raises(ValidationError):
        task.status = TaskStatus.COMPLETED


def test_pending_cannot_complete_without_running() -> None:
    with pytest.raises(InvalidTransitionError) as exc_info:
        make_task().transition_to(TaskStatus.COMPLETED)

    assert exc_info.value.current is TaskStatus.PENDING
    assert exc_info.value.target is TaskStatus.COMPLETED


@pytest.mark.parametrize(
    "terminal", [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
)
def test_terminal_states_allow_no_transitions(terminal: TaskStatus) -> None:
    assert ALLOWED_TRANSITIONS[terminal] == frozenset()
    task = make_task().model_copy(update={"status": terminal})

    for target in TaskStatus:
        with pytest.raises(InvalidTransitionError):
            task.transition_to(target)


def test_every_status_has_a_transition_rule() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(TaskStatus)
