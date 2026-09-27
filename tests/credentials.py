"""Credentials the HTTP tests authenticate with.

Shared so every suite exercises the same wiring a deployment gets, rather than
each one inventing its own way around the gate.

The three subjects are the three interesting shapes: a requester who cannot
approve, an approver who cannot start work, and a single credential holding
both — which is what makes the separation-of-duties check testable, since a
principal has to be able to reach `/approve` before it can be refused there for
being the requester.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from enterprise_agent_platform.auth.models import APIClient, Scope
from enterprise_agent_platform.auth.tokens import TokenAuthenticator


def client_credential(subject: str, token: str, *scopes: Scope) -> APIClient:
    return APIClient(subject=subject, token=SecretStr(token), scopes=frozenset(scopes))


ANALYST = client_credential(
    "analyst-1", "analyst-token-aaaaaaaaaaaaaaaaaaaa", Scope.TASKS_READ, Scope.TASKS_WRITE
)
MANAGER = client_credential(
    "finance-manager-2", "manager-token-bbbbbbbbbbbbbbbbbbbb", Scope.TASKS_READ, Scope.TASKS_APPROVE
)
OTHER_MANAGER = client_credential(
    "finance-manager-3",
    "manager3-token-cccccccccccccccccccc",
    Scope.TASKS_READ,
    Scope.TASKS_APPROVE,
)
OPERATOR = client_credential(
    "solo-operator-4",
    "operator-token-dddddddddddddddddddd",
    Scope.TASKS_READ,
    Scope.TASKS_WRITE,
    Scope.TASKS_APPROVE,
)

AUTHENTICATOR = TokenAuthenticator((ANALYST, MANAGER, OTHER_MANAGER, OPERATOR))


def bearer(client: APIClient) -> dict[str, str]:
    return {"Authorization": f"Bearer {client.token.get_secret_value()}"}


def authorized(app: FastAPI, client: APIClient = ANALYST) -> TestClient:
    """A test client that presents ``client``'s credential on every request."""
    return TestClient(app, headers=bearer(client))
