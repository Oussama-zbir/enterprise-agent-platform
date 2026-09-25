"""A deterministic stand-in for a model, so the platform can be run offline.

``DemoLLMProvider`` is not a model and does not imitate one. It matches a task
goal against a small table of scenarios and replays that scenario's tool calls
one turn at a time, reading each turn's arguments out of the previous turn's
tool results — so a run is still driven by what the tools actually returned,
not by a transcript recorded in advance. When the plan runs out, or a tool
reports a failure the plan cannot continue past, it answers with a summary of
what ran.

This is a deliberate substitution, not a shortcut. Everything a demo is meant
to show belongs to the platform rather than to whatever produced the tool
calls: the risk gate, the pause, the checkpoint, the resumed step budget, the
audit history. A scripted producer demonstrates all of it exactly as well as a
real model would, while staying free, instant, reproducible, and runnable in CI
with no API key and no network. Point ``EAP_LLM_PROVIDER`` at ``anthropic`` and
the same tools run against a real model with nothing else changed.

It is rejected in production for the same reason ``fake`` is.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from enterprise_agent_platform.llm.models import (
    Completion,
    CompletionRequest,
    Message,
    Role,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolResult,
)

GOAL_PREFIX = "Task goal:"
INVOICE_ID = re.compile(r"INV-\d{4}")
MAX_SUMMARY_CHARS = 240
DEFAULT_INVOICE_ID = "INV-1043"


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """One completed tool call from the conversation so far."""

    call: ToolCall
    result: ToolResult

    @property
    def failed(self) -> bool:
        return self.result.is_error

    def payload(self) -> dict[str, Any]:
        """The result parsed as JSON, or ``{}`` if it is not JSON.

        The demo tools return JSON, but an error result is prose, and a tool
        added later may not. Reading a result is therefore allowed to come up
        empty rather than to raise: this stands where a model's own reading of a
        tool result stands, and that can fail too.
        """
        try:
            payload = json.loads(self.result.content)
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}


Step = Callable[[str, tuple[ToolOutcome, ...]], tuple[ToolCall, ...]]
"""One planned assistant turn: the tool calls to make, given the goal and what
earlier calls returned. Returning no calls ends the run with a summary — the
step's way of saying the facts it needed did not arrive."""


@dataclass(frozen=True, slots=True)
class DemoScenario:
    """A goal the demo provider knows how to pursue."""

    name: str
    goal: str
    keywords: tuple[str, ...]
    steps: tuple[Step, ...]

    def matches(self, goal: str) -> bool:
        lowered = goal.lower()
        return all(keyword in lowered for keyword in self.keywords)


def _settle_invoice_lookup(goal: str, outcomes: tuple[ToolOutcome, ...]) -> tuple[ToolCall, ...]:
    return (
        ToolCall(
            id="demo_1_0",
            name="lookup_invoice",
            arguments={"invoice_id": _invoice_id(goal)},
        ),
    )


def _settle_invoice_payment(goal: str, outcomes: tuple[ToolOutcome, ...]) -> tuple[ToolCall, ...]:
    """Pay the invoice the lookup returned, for the amount the lookup reported.

    If the lookup failed, this returns nothing: the run ends with the failure
    reported instead of paying an amount nobody established. That is the same
    judgement the system prompt asks a real model for, and it keeps the demo's
    error path (an unknown invoice id) honest.
    """
    invoice = _latest_payload(outcomes, "lookup_invoice")
    invoice_id = invoice.get("invoice_id")
    amount = _amount(invoice.get("amount_eur"))
    if not isinstance(invoice_id, str) or amount is None:
        return ()
    return (
        ToolCall(
            id="demo_2_0",
            name="pay_invoice",
            arguments={"invoice_id": invoice_id, "amount_eur": str(amount)},
        ),
    )


def _overdue_review(goal: str, outcomes: tuple[ToolOutcome, ...]) -> tuple[ToolCall, ...]:
    return (ToolCall(id="demo_1_0", name="list_overdue_invoices", arguments={"limit": 5}),)


def _overdue_note(goal: str, outcomes: tuple[ToolOutcome, ...]) -> tuple[ToolCall, ...]:
    """Annotate the most overdue invoice the listing returned."""
    listing = _latest_payload(outcomes, "list_overdue_invoices")
    invoices = listing.get("invoices") or []
    if not invoices:
        return ()
    worst = invoices[0]
    return (
        ToolCall(
            id="demo_2_0",
            name="add_invoice_note",
            arguments={
                "invoice_id": worst["invoice_id"],
                "note": f"Flagged for payment review: {worst['days_overdue']} days overdue.",
            },
        ),
    )


SCENARIOS: tuple[DemoScenario, ...] = (
    DemoScenario(
        name="settle_invoice",
        goal=f"Settle invoice {DEFAULT_INVOICE_ID} with the supplier",
        keywords=("settle", "invoice"),
        steps=(_settle_invoice_lookup, _settle_invoice_payment),
    ),
    DemoScenario(
        name="overdue_review",
        goal="Review the overdue supplier invoices and flag the worst one",
        keywords=("overdue",),
        steps=(_overdue_review, _overdue_note),
    ),
)


class DemoLLMProvider:
    """Replays a scenario's tool calls against whatever the tools return."""

    name = "demo"

    def __init__(
        self,
        scenarios: Sequence[DemoScenario] = SCENARIOS,
        *,
        model: str = "demo-scripted",
    ) -> None:
        self._scenarios = tuple(scenarios)
        self._model = model

    async def complete(self, request: CompletionRequest) -> Completion:
        goal = _goal(request.messages)
        scenario = next((s for s in self._scenarios if s.matches(goal)), None)
        if scenario is None:
            return self._answer(request, _unknown_goal(self._scenarios))

        outcomes = _tool_outcomes(request.messages)
        turn = sum(1 for message in request.messages if message.role is Role.ASSISTANT)
        calls = scenario.steps[turn](goal, outcomes) if turn < len(scenario.steps) else ()
        if not calls:
            return self._answer(request, _summarise(outcomes))
        return self._request_tools(request, calls)

    async def aclose(self) -> None:
        """No resources to release."""

    def _answer(self, request: CompletionRequest, text: str) -> Completion:
        return Completion(
            text=text,
            model=self._model,
            stop_reason=StopReason.END_TURN,
            usage=_usage(request, output=text),
        )

    def _request_tools(self, request: CompletionRequest, calls: tuple[ToolCall, ...]) -> Completion:
        return Completion(
            text="",
            model=self._model,
            stop_reason=StopReason.TOOL_USE,
            usage=_usage(request, output=json.dumps([call.arguments for call in calls])),
            tool_calls=calls,
        )


def _goal(messages: Sequence[Message]) -> str:
    """The task goal, as the runner phrased it in the opening user turn."""
    opening = messages[0].content
    return opening.removeprefix(GOAL_PREFIX).strip()


def _invoice_id(goal: str) -> str:
    match = INVOICE_ID.search(goal)
    return match.group(0) if match else DEFAULT_INVOICE_ID


def _amount(raw: Any) -> Decimal | None:
    if not isinstance(raw, str):
        return None
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _tool_outcomes(messages: Sequence[Message]) -> tuple[ToolOutcome, ...]:
    """Pair every tool call in the conversation with the result it received.

    Calls and results live in different turns and are joined by call id, so a
    pause and a resume — which put them several messages apart — make no
    difference to what a later step can read.
    """
    calls: dict[str, ToolCall] = {}
    results: dict[str, ToolResult] = {}
    for message in messages:
        calls.update({call.id: call for call in message.tool_calls})
        results.update({result.call_id: result for result in message.tool_results})
    return tuple(
        ToolOutcome(call=call, result=results[call_id])
        for call_id, call in calls.items()
        if call_id in results
    )


def _latest_payload(outcomes: Sequence[ToolOutcome], tool: str) -> dict[str, Any]:
    """The parsed result of the most recent successful call to ``tool``."""
    for outcome in reversed(outcomes):
        if outcome.call.name == tool and not outcome.failed:
            return outcome.payload()
    return {}


def _summarise(outcomes: Sequence[ToolOutcome]) -> str:
    """Report what ran, including what failed.

    A real model would write prose here. This writes a ledger, because a demo
    whose final answer is invented reads as a demo whose *whole* run might be.
    """
    if not outcomes:
        return "No tools were called, so there is nothing to report."
    lines = ["Done. Tool calls made, in order:"]
    lines += [
        f"- {outcome.call.name}: {'failed' if outcome.failed else 'ok'} — "
        f"{_clip(outcome.result.content)}"
        for outcome in outcomes
    ]
    return "\n".join(lines)


def _unknown_goal(scenarios: Sequence[DemoScenario]) -> str:
    known = "\n".join(f'- "{scenario.goal}"' for scenario in scenarios)
    return (
        "The demo provider has no plan for this goal. It is a scripted stand-in for a "
        "model, not a model; set EAP_LLM_PROVIDER=anthropic to run arbitrary goals. "
        f"Goals it does handle:\n{known}"
    )


def _clip(content: str) -> str:
    collapsed = " ".join(content.split())
    if len(collapsed) <= MAX_SUMMARY_CHARS:
        return collapsed
    return collapsed[: MAX_SUMMARY_CHARS - 1] + "…"


def _usage(request: CompletionRequest, *, output: str) -> TokenUsage:
    """Token counts estimated from character length, at roughly four per token.

    Estimated rather than measured: no tokenizer runs here. They are reported so
    a demo run exercises the same per-run accounting a real backend feeds, and
    so the numbers in a demo transcript are produced rather than written.
    """
    prompt = [request.system or ""]
    prompt += [tool.description for tool in request.tools]
    for message in request.messages:
        prompt.append(message.content)
        prompt += [json.dumps(call.arguments) for call in message.tool_calls]
        prompt += [result.content for result in message.tool_results]
    return TokenUsage(input_tokens=_estimate(prompt), output_tokens=_estimate([output]))


def _estimate(texts: Sequence[str]) -> int:
    return sum(len(text) for text in texts) // 4
