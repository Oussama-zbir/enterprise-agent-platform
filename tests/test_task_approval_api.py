"""HTTP-level tests for the human-in-the-loop approval routes.

The question these answer is whether a human can actually act on a paused run:
see what it wants to do, authorise it and have it continue where it stopped, or
deny it and have the held actions never happen.
"""

from __future__ import annotations

from typing import NamedTuple
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import BaseModel

from enterprise_agent_platform.llm.client import LLMClient
from enterprise_agent_platform.llm.models import Completion, StopReason, TokenUsage, ToolCall
from enterprise_agent_platform.llm.provider import FakeLLMProvider
from enterprise_agent_platform.main import create_app
from enterprise_agent_platform.tasks.repository import InMemoryTaskRepository
from enterprise_agent_platform.tools.models import RiskLevel, Tool
from enterprise_agent_platform.tools.registry import ToolRegistry


class PayArgs(BaseModel):
    invoice_id: str


PAY_CALL = Completion(
    text="Paying the outstanding invoice.",
    model="fake-model",
    stop_reason=StopReason.TOOL_USE,
    usage=TokenUsage(input_tokens=9, output_tokens=6),
    tool_calls=(ToolCall(id="call_1", name="pay_invoice", arguments={"invoice_id": "INV-1"}),),
)


def answer(text: str) -> Completion:
    return Completion(
        text=text,
        model="fake-model",
        stop_reason=StopReason.END_TURN,
        usage=TokenUsage(input_tokens=11, output_tokens=4),
    )


class Fixture(NamedTuple):
    client: TestClient
    paid: list[str]


def build(script: list[Completion]) -> Fixture:
    """A client whose only tool is critical, so any use of it pauses the run."""
    paid: list[str] = []

    async def pay(args: PayArgs) -> str:
        paid.append(args.invoice_id)
        return f"paid {args.invoice_id}"

    tool = Tool(
        name="pay_invoice",
        description="Pay an invoice.",
        arguments=PayArgs,
        risk=RiskLevel.CRITICAL,
        handler=pay,
    )
    client = TestClient(
        create_app(
            task_repository=InMemoryTaskRepository(),
            llm_client=LLMClient(FakeLLMProvider(script), timeout_seconds=5),
            tool_registry=ToolRegistry([tool]),
        )
    )
    return Fixture(client, paid)


def create(client: TestClient) -> str:
    response = client.post(
        "/tasks", json={"goal": "Settle invoice INV-1", "requested_by": "analyst-1"}
    )
    assert response.status_code == 201
    task_id: str = response.json()["id"]
    return task_id


def pause(fixture: Fixture) -> str:
    """Create a task and run it up to its approval gate."""
    task_id = create(fixture.client)
    assert fixture.client.post(f"/tasks/{task_id}/run").json()["task"]["status"] == (
        "awaiting_approval"
    )
    return task_id


def test_the_approval_view_shows_what_the_agent_wants_to_do() -> None:
    f = build([PAY_CALL])
    task_id = pause(f)

    body = f.client.get(f"/tasks/{task_id}/approval").json()

    assert body["goal"] == "Settle invoice INV-1"
    assert body["detail"] == "approval required for: pay_invoice"
    assert body["pending_tool_calls"] == [
        {
            "id": "call_1",
            "name": "pay_invoice",
            # Arguments, not just the name: paying an invoice and paying *this*
            # invoice are different decisions.
            "arguments": {"invoice_id": "INV-1"},
            "risk": "critical",
            "needs_approval": True,
        }
    ]


def test_approving_runs_the_held_call_and_finishes_the_task() -> None:
    f = build([PAY_CALL, answer("INV-1 is settled.")])
    task_id = pause(f)

    response = f.client.post(
        f"/tasks/{task_id}/approve", json={"approved_by": "manager-2", "note": "verified"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"]["status"] == "completed"
    assert body["output"] == "INV-1 is settled."
    assert f.paid == ["INV-1"]
    # Counters cover the whole run, including what the paused half spent.
    assert (body["steps"], body["tool_calls"]) == (2, 1)
    assert body["input_tokens"] == 20


def test_the_approval_is_written_into_the_task_history() -> None:
    f = build([PAY_CALL, answer("done")])
    task_id = pause(f)
    f.client.post(f"/tasks/{task_id}/approve", json={"approved_by": "manager-2"})

    history = f.client.get(f"/tasks/{task_id}").json()["history"]

    assert [step["to_status"] for step in history] == [
        "running",
        "awaiting_approval",
        "running",
        "completed",
    ]
    assert history[2]["reason"] == "approved by manager-2"


def test_rejecting_cancels_the_task_and_the_held_call_never_runs() -> None:
    f = build([PAY_CALL, answer("done")])
    task_id = pause(f)

    response = f.client.post(
        f"/tasks/{task_id}/reject",
        json={"rejected_by": "manager-2", "reason": "supplier not verified"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "cancelled"
    assert body["is_terminal"] is True
    assert body["history"][-1]["reason"] == "rejected by manager-2: supplier not verified"
    assert f.paid == []


def test_a_rejected_task_cannot_then_be_approved() -> None:
    f = build([PAY_CALL, answer("done")])
    task_id = pause(f)
    f.client.post(f"/tasks/{task_id}/reject", json={"rejected_by": "manager-2"})

    response = f.client.post(f"/tasks/{task_id}/approve", json={"approved_by": "manager-3"})

    assert response.status_code == 409
    assert f.paid == []


def test_a_task_cannot_be_approved_twice() -> None:
    f = build([PAY_CALL, answer("done")])
    task_id = pause(f)
    assert (
        f.client.post(f"/tasks/{task_id}/approve", json={"approved_by": "m-2"}).status_code == 200
    )

    response = f.client.post(f"/tasks/{task_id}/approve", json={"approved_by": "m-3"})

    assert response.status_code == 409
    assert f.paid == ["INV-1"]  # the critical call is not replayed


def test_approving_a_task_that_never_paused_is_409() -> None:
    f = build([answer("nothing to do")])
    task_id = create(f.client)

    response = f.client.post(f"/tasks/{task_id}/approve", json={"approved_by": "manager-2"})

    assert response.status_code == 409
    assert "cannot transition task from 'pending'" in response.json()["detail"]


def test_the_approval_view_is_409_when_the_task_is_not_waiting() -> None:
    f = build([answer("nothing to do")])
    task_id = create(f.client)

    response = f.client.get(f"/tasks/{task_id}/approval")

    assert response.status_code == 409
    assert "not awaiting approval" in response.json()["detail"]


def test_approving_an_unknown_task_is_404() -> None:
    f = build([answer("done")])

    response = f.client.post(f"/tasks/{uuid4()}/approve", json={"approved_by": "manager-2"})

    assert response.status_code == 404


def test_an_approval_without_an_approver_is_rejected() -> None:
    f = build([PAY_CALL])
    task_id = pause(f)

    # An approval with nobody attached to it is not an audit trail.
    assert f.client.post(f"/tasks/{task_id}/approve", json={}).status_code == 422
    assert f.client.post(f"/tasks/{task_id}/approve", json={"approved_by": " "}).status_code == 422
    assert f.paid == []


def test_the_paused_conversation_is_not_exposed_on_the_task_resource() -> None:
    f = build([PAY_CALL])
    task_id = pause(f)

    body = f.client.get(f"/tasks/{task_id}").json()

    # It holds model output and tool results; the approval route serves the one
    # slice anyone needs to act on.
    assert "checkpoint" not in body
