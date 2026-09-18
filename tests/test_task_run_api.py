"""HTTP-level tests for POST /tasks/{id}/run."""

from __future__ import annotations

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


async def _pay(args: PayArgs) -> str:
    return f"paid {args.invoice_id}"


PAY_TOOL = Tool(
    name="pay_invoice",
    description="Pay an invoice.",
    arguments=PayArgs,
    risk=RiskLevel.CRITICAL,
    handler=_pay,
)


def build_client(
    script: list[Completion], *, tools: list[Tool[PayArgs]] | None = None
) -> TestClient:
    return TestClient(
        create_app(
            task_repository=InMemoryTaskRepository(),
            llm_client=LLMClient(FakeLLMProvider(script), timeout_seconds=5),
            tool_registry=ToolRegistry(tools or []),
        )
    )


def answer(text: str) -> Completion:
    return Completion(
        text=text,
        model="fake-model",
        stop_reason=StopReason.END_TURN,
        usage=TokenUsage(input_tokens=11, output_tokens=4),
    )


def create(client: TestClient) -> str:
    response = client.post(
        "/tasks", json={"goal": "Settle invoice INV-1", "requested_by": "analyst-1"}
    )
    assert response.status_code == 201
    task_id: str = response.json()["id"]
    return task_id


def test_running_a_task_returns_the_answer_and_the_completed_task() -> None:
    client = build_client([answer("INV-1 settled.")])
    task_id = create(client)

    response = client.post(f"/tasks/{task_id}/run")

    assert response.status_code == 200
    body = response.json()
    assert body["output"] == "INV-1 settled."
    assert body["task"]["status"] == "completed"
    assert body["task"]["is_terminal"] is True
    assert (body["steps"], body["tool_calls"]) == (1, 0)
    assert (body["input_tokens"], body["output_tokens"]) == (11, 4)


def test_a_run_shows_up_in_the_task_history() -> None:
    client = build_client([answer("done")])
    task_id = create(client)
    client.post(f"/tasks/{task_id}/run")

    history = client.get(f"/tasks/{task_id}").json()["history"]

    assert [step["to_status"] for step in history] == ["running", "completed"]


def test_a_task_cannot_be_run_twice() -> None:
    client = build_client([answer("done")])
    task_id = create(client)
    assert client.post(f"/tasks/{task_id}/run").status_code == 200

    response = client.post(f"/tasks/{task_id}/run")

    assert response.status_code == 409
    assert "cannot transition" in response.json()["detail"]


def test_running_an_unknown_task_is_404() -> None:
    client = build_client([answer("done")])

    assert client.post(f"/tasks/{uuid4()}/run").status_code == 404


def test_a_run_that_needs_approval_reports_the_tools_it_is_waiting_on() -> None:
    tool_call = Completion(
        text="",
        model="fake-model",
        stop_reason=StopReason.TOOL_USE,
        usage=TokenUsage(input_tokens=9, output_tokens=6),
        tool_calls=(ToolCall(id="call_1", name="pay_invoice", arguments={"invoice_id": "INV-1"}),),
    )
    client = build_client([tool_call], tools=[PAY_TOOL])
    task_id = create(client)

    body = client.post(f"/tasks/{task_id}/run").json()

    assert body["task"]["status"] == "awaiting_approval"
    assert body["pending_tool_calls"] == ["pay_invoice"]
    assert body["task"]["is_terminal"] is False


def test_a_malformed_task_id_is_422() -> None:
    client = build_client([answer("done")])

    assert client.post("/tasks/not-a-uuid/run").status_code == 422
