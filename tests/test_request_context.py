"""Tests for request correlation IDs and their propagation into logs."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from enterprise_agent_platform.main import create_app
from enterprise_agent_platform.request_context import REQUEST_ID_HEADER, get_request_id


@pytest.fixture
def app() -> FastAPI:
    app = create_app()

    async def probe() -> dict[str, str | None]:
        await asyncio.sleep(0.01)  # yield so concurrent requests interleave
        return {"request_id": get_request_id()}

    async def boom() -> None:
        raise RuntimeError("tool backend exploded")

    app.add_api_route("/_probe", probe)
    app.add_api_route("/_boom", boom)
    return app


@pytest.fixture
def isolated_root_logger() -> Iterator[None]:
    """Let the app's lifespan install its JSON handler, then restore pytest's."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def json_logs(output: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def test_generates_request_id_when_absent(app: FastAPI) -> None:
    response = TestClient(app).get("/health")

    UUID(response.headers[REQUEST_ID_HEADER])


def test_well_formed_inbound_request_id_is_propagated(app: FastAPI) -> None:
    response = TestClient(app).get("/_probe", headers={REQUEST_ID_HEADER: "gw-7f3a:01"})

    assert response.headers[REQUEST_ID_HEADER] == "gw-7f3a:01"
    assert response.json() == {"request_id": "gw-7f3a:01"}


@pytest.mark.parametrize("candidate", ["", "has space", "a" * 129, '"}{"level":"CRITICAL"'])
def test_malformed_inbound_request_id_is_replaced(app: FastAPI, candidate: str) -> None:
    response = TestClient(app).get("/_probe", headers={REQUEST_ID_HEADER: candidate})

    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id != candidate
    UUID(request_id)
    assert response.json() == {"request_id": request_id}


def test_request_id_is_set_on_error_responses(app: FastAPI) -> None:
    response = TestClient(app).get("/tasks/not-a-uuid", headers={REQUEST_ID_HEADER: "req-422"})

    assert response.status_code == 422
    assert response.headers[REQUEST_ID_HEADER] == "req-422"


async def test_concurrent_requests_keep_their_own_request_id(app: FastAPI) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        ids = [f"req-{n}" for n in range(20)]
        responses = await asyncio.gather(
            *(client.get("/_probe", headers={REQUEST_ID_HEADER: rid}) for rid in ids)
        )

        assert [r.json()["request_id"] for r in responses] == ids

        # httpx runs the app in the caller's task, so a missing reset would leak here.
        await client.get("/_probe")
        assert get_request_id() is None


@pytest.mark.usefixtures("isolated_root_logger")
def test_logs_emitted_during_request_carry_request_id(
    app: FastAPI, capsys: pytest.CaptureFixture[str]
) -> None:
    with TestClient(app) as client:
        client.post(
            "/tasks",
            json={"goal": "Reconcile supplier payments", "requested_by": "analyst-1"},
            headers={REQUEST_ID_HEADER: "req-abc"},
        )

    records = {r["message"]: r for r in json_logs(capsys.readouterr().out)}

    assert records["task.created"]["request_id"] == "req-abc"
    completed = records["request.completed"]
    assert completed["request_id"] == "req-abc"
    assert (completed["method"], completed["path"], completed["status_code"]) == (
        "POST",
        "/tasks",
        201,
    )
    assert completed["duration_ms"] >= 0
    assert "request_id" not in records["service.startup"]


@pytest.mark.usefixtures("isolated_root_logger")
def test_unhandled_exception_returns_500_correlated_with_logged_traceback(
    app: FastAPI, capsys: pytest.CaptureFixture[str]
) -> None:
    with TestClient(app) as client:
        response = client.get("/_boom", headers={REQUEST_ID_HEADER: "req-500"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error"}
    assert response.headers[REQUEST_ID_HEADER] == "req-500"

    records = {r["message"]: r for r in json_logs(capsys.readouterr().out)}
    failed = records["request.failed"]
    assert failed["request_id"] == "req-500"
    assert "tool backend exploded" in failed["exc_info"]
    assert records["request.completed"]["status_code"] == 500
