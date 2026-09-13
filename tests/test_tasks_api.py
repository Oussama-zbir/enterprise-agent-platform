"""HTTP-level tests for the /tasks routes."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from enterprise_agent_platform.main import create_app
from enterprise_agent_platform.tasks.models import AgentTask
from enterprise_agent_platform.tasks.repository import (
    ConcurrentUpdateError,
    InMemoryTaskRepository,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(task_repository=InMemoryTaskRepository()))


def create(client: TestClient, goal: str = "Reconcile supplier payments") -> dict[str, object]:
    response = client.post("/tasks", json={"goal": goal, "requested_by": "analyst-1"})
    assert response.status_code == 201
    body: dict[str, object] = response.json()
    return body


def test_create_task_returns_pending_task(client: TestClient) -> None:
    body = create(client)

    assert body["status"] == "pending"
    assert body["version"] == 1
    assert body["is_terminal"] is False
    assert body["history"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"goal": "   ", "requested_by": "u"},
        {"goal": "x"},
        {"goal": "x", "requested_by": "u", "status": "completed"},
    ],
)
def test_create_task_rejects_invalid_payload(client: TestClient, payload: dict[str, str]) -> None:
    assert client.post("/tasks", json=payload).status_code == 422


def test_get_task_round_trips_and_unknown_is_404(client: TestClient) -> None:
    created = create(client)

    assert client.get(f"/tasks/{created['id']}").json() == created
    assert client.get(f"/tasks/{uuid4()}").status_code == 404
    assert client.get("/tasks/not-a-uuid").status_code == 422


def test_list_tasks_filters_by_status(client: TestClient) -> None:
    kept = create(client, "first")
    cancelled = create(client, "second")
    client.post(f"/tasks/{cancelled['id']}/cancel")

    pending = client.get("/tasks", params={"status": "pending"}).json()

    assert [t["id"] for t in pending] == [kept["id"]]
    assert len(client.get("/tasks").json()) == 2
    assert client.get("/tasks", params={"limit": 0}).status_code == 422


def test_cancel_records_reason_and_is_terminal(client: TestClient) -> None:
    created = create(client)

    response = client.post(f"/tasks/{created['id']}/cancel", json={"reason": "duplicate request"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "cancelled"
    assert body["is_terminal"] is True
    assert body["version"] == 2
    assert body["history"][0]["reason"] == "duplicate request"


def test_cancel_without_body_is_accepted(client: TestClient) -> None:
    created = create(client)

    assert client.post(f"/tasks/{created['id']}/cancel").status_code == 200


def test_cancelling_terminal_task_is_conflict(client: TestClient) -> None:
    created = create(client)
    client.post(f"/tasks/{created['id']}/cancel")

    response = client.post(f"/tasks/{created['id']}/cancel")

    assert response.status_code == 409
    assert "cannot transition" in response.json()["detail"]


def test_cancel_unknown_task_is_404(client: TestClient) -> None:
    assert client.post(f"/tasks/{uuid4()}/cancel").status_code == 404


class _RacingRepository(InMemoryTaskRepository):
    """Simulates another writer committing between our read and our write."""

    async def update(self, task: AgentTask, *, expected_version: int) -> None:
        raise ConcurrentUpdateError(task.id, expected_version, expected_version + 1)


def test_concurrent_modification_is_conflict() -> None:
    client = TestClient(create_app(task_repository=_RacingRepository()))
    created = create(client)

    response = client.post(f"/tasks/{created['id']}/cancel")

    assert response.status_code == 409
    assert "modified concurrently" in response.json()["detail"]
