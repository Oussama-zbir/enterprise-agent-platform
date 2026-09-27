"""The shipped end-to-end demo: run, pause for approval, approve, resume.

It drives the real application over HTTP — same routes, same middleware, same
agent runner, same risk gate, same bearer-token authentication — through an
in-process ASGI transport, so there is no port to bind, no key to supply and no
network to reach. What it prints is the transcript of those calls, so every
request in the README is one a reader can reproduce and one this repository
re-executes on every CI run.

The walkthrough also checks its own story: the run must pause on the critical
tool, the requester's attempt to approve their own task must be refused, the
approval must resume the same run rather than start a new one, and the ledger
must show exactly one payment at the end. If any of that stops being
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

from enterprise_agent_platform.auth.models import APIClient
from enterprise_agent_platform.auth.tokens import TokenAuthenticator
from enterprise_agent_platform.demo.credentials import (
    DEMO_ANALYST,
    DEMO_API_CLIENTS,
    DEMO_MANAGER,
)
from enterprise_agent_platform.demo.data import Ledger
from enterprise_agent_platform.demo.provider import DEFAULT_INVOICE_ID, DemoLLMProvider
from enterprise_agent_platform.demo.tools import build_demo_tools
from enterprise_agent_platform.llm.client import LLMClient
from enterprise_agent_platform.main import create_app
from enterprise_agent_platform.tools.registry import ToolRegistry

BASE_URL = "http://127.0.0.1:8000"
DEMO_GOAL = f"Settle invoice {DEFAULT_INVOICE_ID} with the supplier, in full."
DEMO_TIMEOUT_SECONDS = 5.0
"""A deadline the scripted provider can never need; it answers synchronously."""

ANALYST_VAR = "EAP_ANALYST_TOKEN"
MANAGER_VAR = "EAP_MANAGER_TOKEN"
"""Shell variables the transcript exports, so no line repeats a token."""


class WalkthroughError(AssertionError):
    """The demo did not do what the demo says the platform does."""


@dataclass(frozen=True, slots=True)
class WalkthroughResult:
    """What the run did, for the caller (a test, or an exit code) to check."""

    task_id: str
    paused_on: tuple[str, ...]
    requester_approval_status: int
    """What the requester got when they tried to release their own task."""
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

    The two demo credentials are issued the same way — the transcript would
    otherwise be a sequence of 401s, and the point of running the whole app
    rather than calling the runner directly is that nothing is bypassed.
    """
    # The transcript's own HTTP client would otherwise log a line per call into
    # the middle of the service's logs, which is noise from the demo harness
    # rather than from the platform.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return create_app(
        llm_client=LLMClient(DemoLLMProvider(), timeout_seconds=DEMO_TIMEOUT_SECONDS),
        tool_registry=ToolRegistry(build_demo_tools(ledger)),
        authenticator=TokenAuthenticator(DEMO_API_CLIENTS),
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
        transcript.preamble()

        task = await transcript.call(
            "POST",
            "/tasks",
            body={"goal": goal},
            heading=(
                "1. Create the task as the analyst. Nothing runs yet; it is 'pending'.\n"
                "#    'requested_by' is not a field — it is the token's subject, because it\n"
                "#    is half of the separation-of-duties check at the approval gate."
            ),
            as_=DEMO_ANALYST,
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
            as_=DEMO_ANALYST,
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
            as_=DEMO_MANAGER,
        )

        await transcript.call(
            "POST",
            f"/tasks/{task_id}/approve",
            heading=(
                "4. The analyst tries to release the task they asked for. 403: their token\n"
                "#    carries 'tasks:write', not 'tasks:approve' — the authority to make an\n"
                "#    agent act is not the authority to release what it wants to do. A token\n"
                "#    holding both is refused too, by the separation-of-duties check, since\n"
                "#    the subject that requested a task cannot be the one that approves it."
            ),
            as_=DEMO_ANALYST,
            expect=403,
        )
        # Read now rather than at the end: it is this call's outcome, and the
        # steps below overwrite it.
        requester_approval_status = transcript.last_status
        _expect(
            not ledger.payments,
            "a refused approval must not have run the held call",
        )

        resumed = await transcript.call(
            "POST",
            f"/tasks/{task_id}/approve",
            body={"note": "supplier and amount verified"},
            heading=(
                "5. The manager approves. The held call runs and the *same* run continues\n"
                "#    from its checkpoint: the counters below cover the whole run, pause\n"
                "#    included. Who approved comes from the token, not from the body."
            ),
            as_=DEMO_MANAGER,
        )
        _expect(
            resumed["task"]["status"] == "completed",
            f"the approved run should have completed, got '{resumed['task']['status']}'",
        )
        _expect(
            resumed["steps"] > run["steps"],
            "the resumed run should continue the paused run's step budget, not reset it",
        )

        final = await transcript.call(
            "GET",
            f"/tasks/{task_id}",
            heading=(
                "6. The audit history: who asked, what paused it, who approved it, and\n"
                "#    the version at every step. Both names are authenticated subjects."
            ),
            as_=DEMO_ANALYST,
        )
        _expect(
            final["requested_by"] == DEMO_ANALYST.subject,
            f"the requester should be the analyst's subject, got '{final['requested_by']}'",
        )
        _expect(
            any(
                str(step["reason"]).startswith(f"approved by {DEMO_MANAGER.subject}")
                for step in final["history"]
            ),
            "the history should record the manager as the approver",
        )

        _expect(
            list(ledger.payments) == [DEFAULT_INVOICE_ID],
            f"exactly one invoice should have been paid, ledger holds {list(ledger.payments)}",
        )
        return WalkthroughResult(
            task_id=task_id,
            paused_on=paused_on,
            requester_approval_status=requester_approval_status,
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
    """Makes each call as one of the demo's subjects, and prints it as curl."""

    def __init__(self, client: httpx.AsyncClient, out: TextIO) -> None:
        self._client = client
        self._out = out
        self.last_status = 0
        """The status of the most recent call, for a step whose outcome *is* the
        status rather than the body. Read it before making the next call."""

    def preamble(self) -> None:
        """Name the credentials the transcript below presents.

        The tokens are printed once, as shell variables, so each request shows
        which subject made it without repeating a secret on every line — and so
        the curl commands can be pasted against a running service unchanged.
        """
        print("# 0. The two credentials this deployment issues.", file=self._out)
        for credential, variable in ((DEMO_ANALYST, ANALYST_VAR), (DEMO_MANAGER, MANAGER_VAR)):
            scopes = ", ".join(sorted(credential.scopes))
            print(
                f"$ export {variable}={credential.token.get_secret_value()}"
                f"  # {credential.subject}: {scopes}",
                file=self._out,
            )

    async def call(
        self,
        method: str,
        path: str,
        *,
        heading: str,
        as_: APIClient,
        body: dict[str, Any] | None = None,
        expect: int = 200,
    ) -> dict[str, Any]:
        print(f"\n# {heading}", file=self._out)
        print(_as_curl(method, path, body, as_), file=self._out)
        response = await self._client.request(method, path, json=body, headers=_bearer(as_))
        self.last_status = response.status_code
        payload = response.json()
        print(json.dumps(payload, indent=2), file=self._out)
        _expect(
            response.status_code == expect,
            f"{method} {path} as {as_.subject} returned {response.status_code}, expected {expect}",
        )
        return dict(payload)


def _bearer(credential: APIClient) -> dict[str, str]:
    return {"Authorization": f"Bearer {credential.token.get_secret_value()}"}


def _as_curl(method: str, path: str, body: dict[str, Any] | None, as_: APIClient) -> str:
    variable = ANALYST_VAR if as_ is DEMO_ANALYST else MANAGER_VAR
    lines = [
        f"$ curl -s -X {method} {BASE_URL}{path}",
        f'    -H "Authorization: Bearer ${variable}"',
    ]
    if body is not None:
        lines.append("    -H 'Content-Type: application/json'")
        lines.append(f"    -d '{json.dumps(body)}'")
    return " \\\n".join(lines)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise WalkthroughError(message)
