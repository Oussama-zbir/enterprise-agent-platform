"""Tests for the health endpoint and application factory."""

from __future__ import annotations

from fastapi.testclient import TestClient

from enterprise_agent_platform import __version__
from enterprise_agent_platform.main import create_app


def test_health_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "enterprise-agent-platform"
    assert body["version"] == __version__


def test_health_reports_configured_environment() -> None:
    client = TestClient(create_app())

    body = client.get("/health").json()

    assert body["environment"] in {"development", "staging", "production", "test"}
