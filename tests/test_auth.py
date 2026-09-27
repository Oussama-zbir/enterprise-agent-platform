"""Tests for bearer-token authentication and the authority to approve.

The question these answer is whether the approval gate is a gate. A platform
that lets an agent move money on a human's say-so has to know whose say-so it
was, and has to refuse an approval from the person who asked for the work.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from enterprise_agent_platform.auth.models import (
    APIClient,
    ApprovalPolicy,
    Principal,
    Scope,
)
from enterprise_agent_platform.auth.tokens import TokenAuthenticator
from enterprise_agent_platform.config import Settings
from enterprise_agent_platform.llm.client import LLMClient
from enterprise_agent_platform.llm.models import Completion, StopReason, TokenUsage, ToolCall
from enterprise_agent_platform.llm.provider import FakeLLMProvider
from enterprise_agent_platform.main import create_app
from enterprise_agent_platform.tasks.repository import InMemoryTaskRepository
from enterprise_agent_platform.tools.models import RiskLevel, Tool
from enterprise_agent_platform.tools.registry import ToolRegistry
from tests.credentials import (
    ANALYST,
    AUTHENTICATOR,
    MANAGER,
    OPERATOR,
    OTHER_MANAGER,
    authorized,
    bearer,
    client_credential,
)

# --- the authenticator -----------------------------------------------------


def test_a_known_token_resolves_to_its_principal() -> None:
    principal = AUTHENTICATOR.authenticate(ANALYST.token.get_secret_value())

    assert principal == Principal(subject="analyst-1", scopes=ANALYST.scopes)


@pytest.mark.parametrize("presented", ["", "not-a-token", "analyst-token-aaaaaaaaaaaaaaaaaaa"])
def test_an_unknown_token_resolves_to_nobody(presented: str) -> None:
    # The last case is the real token minus its final character: a near miss is
    # as wrong as a miss, and costs the same work to reject.
    assert AUTHENTICATOR.authenticate(presented) is None


def test_the_plaintext_token_is_not_retained_anywhere_in_the_index() -> None:
    secret = ANALYST.token.get_secret_value()

    assert secret not in repr(AUTHENTICATOR.__dict__)


def test_two_clients_sharing_a_token_is_refused() -> None:
    shared = "shared-token-eeeeeeeeeeeeeeeeeeee"
    with pytest.raises(ValueError, match="share a token"):
        TokenAuthenticator(
            (
                client_credential("a", shared, Scope.TASKS_WRITE),
                client_credential("b", shared, Scope.TASKS_WRITE),
            )
        )


def test_an_empty_authenticator_reports_itself_unconfigured() -> None:
    empty = TokenAuthenticator()

    assert empty.is_configured is False
    assert empty.authenticate("anything") is None


@pytest.mark.parametrize(
    "overrides",
    [
        # A secret short enough to be guessed offline is not a credential.
        {"token": "too-short"},
        # Subjects are written into logs and into the immutable task history.
        {"subject": "bad subject"},
        {"subject": ""},
        # A credential that authorises nothing can still authenticate, which is
        # a principal no route will ever accept and nobody will think to look at.
        {"scopes": []},
    ],
)
def test_a_credential_that_cannot_be_trusted_is_refused(overrides: dict[str, object]) -> None:
    fields: dict[str, object] = {
        "subject": "svc-ingest",
        "token": "a-perfectly-fine-token-ffffffff",
        "scopes": ["tasks:write"],
    }
    with pytest.raises(ValidationError):
        APIClient.model_validate({**fields, **overrides})


ENTRY = '{"subject":"dup","token":"a-perfectly-fine-token-gggggggggg","scopes":["tasks:read"]}'


def test_duplicate_subjects_are_refused_by_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Through the environment rather than as a keyword, because the JSON parse
    # is part of what is being checked: this is how an operator supplies it.
    monkeypatch.setenv("EAP_API_CLIENTS", f"[{ENTRY},{ENTRY}]")

    with pytest.raises(ValidationError, match="unique subjects"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_a_production_deployment_must_issue_at_least_one_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nowhere else: no credentials is what closed-by-default means, and the
    # suite relies on it. In production it is a service that answers only 401.
    for name, value in (
        ("EAP_ENVIRONMENT", "production"),
        ("EAP_LLM_PROVIDER", "anthropic"),
        ("EAP_TASK_STORE", "postgres"),
        ("EAP_DATABASE_URL", "postgresql://eap:eap@localhost:5432/eap"),
    ):
        monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError, match="EAP_API_CLIENTS is required in production"):
        Settings(_env_file=None)  # type: ignore[call-arg]

    monkeypatch.setenv("EAP_API_CLIENTS", f"[{ENTRY}]")
    assert Settings(_env_file=None).api_clients[0].subject == "dup"  # type: ignore[call-arg]


# --- the HTTP surface ------------------------------------------------------


def app_client() -> TestClient:
    return authorized(
        create_app(task_repository=InMemoryTaskRepository(), authenticator=AUTHENTICATOR)
    )


def test_health_stays_open_so_probes_need_no_credential() -> None:
    client = TestClient(create_app(authenticator=AUTHENTICATOR))

    assert client.get("/health").status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/tasks"),
        ("GET", "/tasks"),
        ("GET", "/tasks/00000000-0000-0000-0000-000000000000"),
        ("POST", "/tasks/00000000-0000-0000-0000-000000000000/run"),
        ("POST", "/tasks/00000000-0000-0000-0000-000000000000/cancel"),
        ("GET", "/tasks/00000000-0000-0000-0000-000000000000/approval"),
        ("POST", "/tasks/00000000-0000-0000-0000-000000000000/approve"),
        ("POST", "/tasks/00000000-0000-0000-0000-000000000000/reject"),
    ],
)
def test_every_task_route_is_closed_without_a_credential(method: str, path: str) -> None:
    client = TestClient(create_app(authenticator=AUTHENTICATOR))

    response = client.request(method, path, json={})

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "header",
    ["", "Bearer", "Basic dXNlcjpwYXNz", "Token analyst-token-aaaaaaaaaaaaaaaaaaaa"],
)
def test_a_credential_this_api_does_not_accept_is_401(header: str) -> None:
    client = TestClient(create_app(authenticator=AUTHENTICATOR))

    assert client.get("/tasks", headers={"Authorization": header}).status_code == 401


def test_an_unconfigured_deployment_authenticates_nobody() -> None:
    # Fail closed: with no credentials issued there is no principal to be, so
    # the API is shut rather than open.
    client = TestClient(create_app(task_repository=InMemoryTaskRepository()))

    assert client.get("/tasks", headers=bearer(ANALYST)).status_code == 401


def test_authentication_runs_before_request_validation() -> None:
    client = TestClient(create_app(authenticator=AUTHENTICATOR))

    # An anonymous caller learns nothing about the shape of the API from a 422.
    assert client.get("/tasks/not-a-uuid").status_code == 401
    assert client.get("/tasks/not-a-uuid", headers=bearer(ANALYST)).status_code == 422


def test_a_scope_the_token_does_not_hold_is_403_and_not_a_challenge() -> None:
    client = authorized(
        create_app(task_repository=InMemoryTaskRepository(), authenticator=AUTHENTICATOR),
        MANAGER,
    )

    # The manager may approve and read; starting work is not theirs to do.
    response = client.post("/tasks", json={"goal": "Settle invoice INV-1"})

    assert response.status_code == 403
    assert "tasks:write" in response.json()["detail"]
    # Re-authenticating cannot fix a scope problem, so no challenge is offered.
    assert "WWW-Authenticate" not in response.headers


def test_the_requester_is_the_authenticated_subject_not_the_body() -> None:
    client = app_client()

    body = client.post("/tasks", json={"goal": "Settle invoice INV-1"}).json()

    assert body["requested_by"] == ANALYST.subject


def test_a_body_that_tries_to_name_the_requester_is_refused() -> None:
    client = app_client()

    response = client.post(
        "/tasks", json={"goal": "Settle invoice INV-1", "requested_by": "finance-manager-2"}
    )

    assert response.status_code == 422


# --- separation of duties --------------------------------------------------


class PayArgs(BaseModel):
    invoice_id: str


PAY_CALL = Completion(
    text="Paying the outstanding invoice.",
    model="fake-model",
    stop_reason=StopReason.TOOL_USE,
    usage=TokenUsage(input_tokens=9, output_tokens=6),
    tool_calls=(ToolCall(id="call_1", name="pay_invoice", arguments={"invoice_id": "INV-1"}),),
)
ANSWER = Completion(
    text="INV-1 is settled.",
    model="fake-model",
    stop_reason=StopReason.END_TURN,
    usage=TokenUsage(input_tokens=11, output_tokens=4),
)


class Paused(NamedTuple):
    """An app holding one task paused on a critical tool, plus what it paid."""

    app: FastAPI
    paid: list[str]
    task_id: str


def paused_app(policy: ApprovalPolicy | None = None, requester: APIClient = ANALYST) -> Paused:
    paid: list[str] = []

    async def pay(args: PayArgs) -> str:
        paid.append(args.invoice_id)
        return f"paid {args.invoice_id}"

    app = create_app(
        task_repository=InMemoryTaskRepository(),
        llm_client=LLMClient(FakeLLMProvider([PAY_CALL, ANSWER]), timeout_seconds=5),
        tool_registry=ToolRegistry(
            [
                Tool(
                    name="pay_invoice",
                    description="Pay an invoice.",
                    arguments=PayArgs,
                    risk=RiskLevel.CRITICAL,
                    handler=pay,
                )
            ]
        ),
        authenticator=AUTHENTICATOR,
        approval_policy=policy,
    )
    client = authorized(app, requester)
    task_id = client.post("/tasks", json={"goal": "Settle invoice INV-1"}).json()["id"]
    assert client.post(f"/tasks/{task_id}/run").json()["task"]["status"] == "awaiting_approval"
    return Paused(app, paid, task_id)


def test_the_requester_cannot_approve_their_own_task() -> None:
    # The operator holds `tasks:approve`, so scope is not what stops it: this is
    # authority over *this* task, which a scope cannot express.
    paused = paused_app(requester=OPERATOR)

    response = authorized(paused.app, OPERATOR).post(f"/tasks/{paused.task_id}/approve")

    # The requester already decided they wanted this; their approval adds no
    # second judgement, so the held call must not run.
    assert response.status_code == 403
    assert "someone other than" in response.json()["detail"]
    assert paused.paid == []


def test_a_second_person_can_approve_it() -> None:
    paused = paused_app()

    response = authorized(paused.app, MANAGER).post(
        f"/tasks/{paused.task_id}/approve", json={"note": "supplier verified"}
    )

    assert response.status_code == 200
    assert response.json()["task"]["status"] == "completed"
    assert paused.paid == ["INV-1"]


def test_the_approver_recorded_in_history_is_the_authenticated_subject() -> None:
    paused = paused_app()
    approver = authorized(paused.app, MANAGER)
    approver.post(f"/tasks/{paused.task_id}/approve")

    history = approver.get(f"/tasks/{paused.task_id}").json()["history"]

    assert history[2]["reason"] == f"approved by {MANAGER.subject}"


def test_the_rejecter_recorded_in_history_is_the_authenticated_subject() -> None:
    paused = paused_app()
    approver = authorized(paused.app, OTHER_MANAGER)
    approver.post(f"/tasks/{paused.task_id}/reject", json={"reason": "supplier unverified"})

    history = approver.get(f"/tasks/{paused.task_id}").json()["history"]

    assert history[-1]["reason"] == f"rejected by {OTHER_MANAGER.subject}: supplier unverified"
    assert paused.paid == []


def test_a_body_that_tries_to_name_the_approver_is_refused() -> None:
    paused = paused_app()

    response = authorized(paused.app, MANAGER).post(
        f"/tasks/{paused.task_id}/approve", json={"approved_by": ANALYST.subject}
    )

    # Silently ignoring the field would leave a caller believing it took effect;
    # `extra="forbid"` makes the API say so.
    assert response.status_code == 422
    assert paused.paid == []


def test_a_deployment_may_switch_separation_of_duties_off() -> None:
    # A single-operator deployment that turns it off has made a choice; one that
    # never had the check has an approval gate in name only.
    paused = paused_app(ApprovalPolicy(requires_second_person=False), requester=OPERATOR)

    response = authorized(paused.app, OPERATOR).post(f"/tasks/{paused.task_id}/approve")

    assert response.status_code == 200
    assert paused.paid == ["INV-1"]


def test_the_requester_may_still_reject_their_own_task() -> None:
    paused = paused_app(requester=OPERATOR)

    # Withholding a capability needs no second opinion; requiring one would
    # leave a requester unable to stop an agent acting on their own mistake.
    response = authorized(paused.app, OPERATOR).post(f"/tasks/{paused.task_id}/reject")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert paused.paid == []


def test_the_policy_decides_on_subjects_rather_than_on_scopes() -> None:
    policy = ApprovalPolicy()
    requester = Principal(subject="analyst-1", scopes=frozenset({Scope.TASKS_APPROVE}))

    assert policy.may_approve(requester, requested_by="analyst-1") is False
    assert policy.may_approve(requester, requested_by=OTHER_MANAGER.subject) is True
