"""The shipped end-to-end demo: run, pause for approval, approve, resume.

It drives the real application over HTTP — same routes, same middleware, same
agent runner, same risk gate — through an in-process ASGI transport, so there
is no port to bind, no key to supply and no network to reach. What it prints is
the transcript of those calls, so every request in the README is one a reader
can reproduce and one this repository re-executes on every CI run.

The walkthrough also checks its own story: the run must pause on the critical
tool, the approval must resume the same run rather than start a new one, and
the ledger must show exactly one payment at the end. If any of that stops being
true, the demo exits non-zero instead of printing a transcript that has quietly
stopped matching the system.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, TextIO

import httpx
from fastapi import FastAPI

from enterprise_agent_platform.demo.data import Ledger
from enterprise_agent_platform.demo.provider import DEFAULT_INVOICE_ID, DemoLLMProvider
from enterprise_agent_platform.demo.tools import build_demo_tools
from enterprise_agent_platform.llm.client import LLMClient
from enterprise_agent_platform.main import create_app
from enterprise_agent_platform.tools.registry import ToolRegistry

BASE_URL = "http://127.0.0.1:8000"
DEMO_GOAL = f"Settle invoice {DEFAULT_INVOICE_ID} with the supplier, in full."
REQUESTED_BY = "analyst-1"
APPROVED_BY = "finance-manager-2"
DEMO_TIMEOUT_SECONDS = 5.0
"""A deadline the scripted provider can never need; it answers synchronously."""


class WalkthroughError(AssertionError):
    """The demo did not do what the demo says the platform does."""


@dataclass(frozen=True, slots=True)
class WalkthroughResult:
    """What the run did, for the caller (a test, or an exit code) to check."""

    task_id: str
    paused_on: tuple[str, ...]
    final_status: str
    steps: int
    tool_calls: int
    ledger: Ledger


def build_demo_app(ledger: Ledger) -> FastAPI:
    """Build the app the demo drives.

    The backend and the tool set are wired explicitly rather than read from the
    environment, so the transcript cannot be changed by a stray ``.env``;
    ``EAP_LLM_PROVIDER=demo EAP_DEMO_TOOLS=true`` produces the same wiring under
    uvicorn. The ledger is injected so the walkthrough can check the payment
    against the system of record rather than against the agent's account of
    itself.
    """
    # The transcript's own HTTP client would otherwise log a line per call into
    # the middle of the service's logs, which is noise from the demo harness
    # rather than from the platform.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return create_app(
        llm_client=LLMClient(DemoLLMProvider(), timeout_seconds=DEMO_TIMEOUT_SECONDS),
        tool_registry=ToolRegistry(build_demo_tools(ledger)),
    )


async def run_walkthrough(
    app: FastAPI, ledger: Ledger, *, goal: str = DEMO_GOAL, out: TextIO = sys.stdout
) -> WalkthroughResult:
    """Drive one task from creation to a resumed, completed run."""
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client,
    ):
        transcript = _Transcript(client, out)

        task = await transcript.call(
            "POST",
            "/tasks",
            body={"goal": goal, "requested_by": REQUESTED_BY},
            heading="1. Create the task. Nothing runs yet; it is 'pending'.",
            expect=201,
        )
        task_id = str(task["id"])

        run = await transcript.call(
            "POST",
            f"/tasks/{task_id}/run",
            heading=(
                "2. Run it. The agent reads the invoice unattended, then asks to pay it —\n"
                "#    'pay_invoice' is CRITICAL, above EAP_AGENT_AUTO_APPROVE_UP_TO=read,\n"
                "#    so the whole turn stops and the task parks in 'awaiting_approval'."
            ),
        )
        _expect(
            run["task"]["status"] == "awaiting_approval",
            f"the run should have paused for approval, got '{run['task']['status']}'",
        )
        _expect(
            not ledger.payments,
            "nothing should have been paid before a human approved it",
        )
        paused_on = tuple(run["pending_tool_calls"])

        await transcript.call(
            "GET",
            f"/tasks/{task_id}/approval",
            heading=(
                "3. Show the approver the decision: which calls are held, with the\n"
                "#    arguments they would run with and the risk that stopped them."
            ),
        )

        resumed = await transcript.call(
            "POST",
            f"/tasks/{task_id}/approve",
            body={"approved_by": APPROVED_BY, "note": "supplier and amount verified"},
            heading=(
                "4. Approve. The held call runs and the *same* run continues from its\n"
                "#    checkpoint: the counters below cover the whole run, pause included."
            ),
        )
        _expect(
            resumed["task"]["status"] == "completed",
            f"the approved run should have completed, got '{resumed['task']['status']}'",
        )
        _expect(
            resumed["steps"] > run["steps"],
            "the resumed run should continue the paused run's step budget, not reset it",
        )

        await transcript.call(
            "GET",
            f"/tasks/{task_id}",
            heading=(
                "5. The audit history: who asked, what paused it, who approved it, and\n"
                "#    the version at every step."
            ),
        )

        _expect(
            list(ledger.payments) == [DEFAULT_INVOICE_ID],
            f"exactly one invoice should have been paid, ledger holds {list(ledger.payments)}",
        )
        return WalkthroughResult(
            task_id=task_id,
            paused_on=paused_on,
            final_status=str(resumed["task"]["status"]),
            steps=int(resumed["steps"]),
            tool_calls=int(resumed["tool_calls"]),
            ledger=ledger,
        )


async def main(out: TextIO = sys.stdout) -> int:
    """Run the walkthrough and report whether the platform still behaves."""
    ledger = Ledger()
    try:
        result = await run_walkthrough(build_demo_app(ledger), ledger, out=out)
    except WalkthroughError as exc:
        print(f"\nDEMO FAILED: {exc}", file=out)
        return 1
    payment = ledger.payments[DEFAULT_INVOICE_ID]
    print(
        f"\n# Done. Task {result.task_id} is '{result.final_status}' after {result.steps} model "
        f"calls and {result.tool_calls} tool calls,\n"
        f"# pausing on {', '.join(result.paused_on)} for a human. The ledger records one "
        f"payment: {payment.amount_eur} EUR under {payment.reference}.",
        file=out,
    )
    return 0


class _Transcript:
    """Makes each call and prints it as the curl a reader can paste."""

    def __init__(self, client: httpx.AsyncClient, out: TextIO) -> None:
        self._client = client
        self._out = out

    async def call(
        self,
        method: str,
        path: str,
        *,
        heading: str,
        body: dict[str, Any] | None = None,
        expect: int = 200,
    ) -> dict[str, Any]:
        print(f"\n# {heading}", file=self._out)
        print(_as_curl(method, path, body), file=self._out)
        response = await self._client.request(method, path, json=body)
        payload = response.json()
        print(json.dumps(payload, indent=2), file=self._out)
        _expect(
            response.status_code == expect,
            f"{method} {path} returned {response.status_code}, expected {expect}",
        )
        return dict(payload)


def _as_curl(method: str, path: str, body: dict[str, Any] | None) -> str:
    command = f"$ curl -s -X {method} {BASE_URL}{path}"
    if body is None:
        return command
    return f"{command} \\\n    -H 'Content-Type: application/json' \\\n    -d '{json.dumps(body)}'"


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise WalkthroughError(message)
