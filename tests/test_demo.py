"""Tests for the shipped demo: its tools, its scripted provider, and the run.

The walkthrough test is the one that matters. It is the repository's only
end-to-end coverage — HTTP in, agent loop, risk gate, approval pause,
checkpoint, resumed run, audit history — and it is the same code path a reader
executes with ``python -m enterprise_agent_platform.demo``, so a transcript in
the README cannot drift from the system without a test going red.
"""

from __future__ import annotations

import io
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from enterprise_agent_platform.agent.runner import AgentRunner
from enterprise_agent_platform.config import Settings
from enterprise_agent_platform.demo.data import AS_OF, Ledger, LedgerError
from enterprise_agent_platform.demo.provider import (
    DEFAULT_INVOICE_ID,
    DemoLLMProvider,
    ToolOutcome,
    _tool_outcomes,
)
from enterprise_agent_platform.demo.tools import build_demo_tools
from enterprise_agent_platform.demo.walkthrough import (
    WalkthroughError,
    build_demo_app,
    run_walkthrough,
)
from enterprise_agent_platform.llm.factory import build_llm_client
from enterprise_agent_platform.llm.models import (
    CompletionRequest,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolResult,
)
from enterprise_agent_platform.main import create_app
from enterprise_agent_platform.tools.models import RiskLevel
from enterprise_agent_platform.tools.registry import ToolRegistry


def settings(**overrides: Any) -> Settings:
    """Settings built from explicit values only, ignoring any local ``.env``."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def goal_request(goal: str, *turns: Message) -> CompletionRequest:
    return CompletionRequest(
        messages=(Message(role=Role.USER, content=f"Task goal:\n{goal}"), *turns)
    )


# --- the ledger ------------------------------------------------------------


def test_overdue_lists_the_oldest_unpaid_invoice_first() -> None:
    ledger = Ledger()

    overdue = ledger.overdue()

    assert [invoice.id for invoice in overdue] == ["INV-1051", "INV-1043", "INV-1052"]
    assert all(invoice.due_on < AS_OF for invoice in overdue)


def test_paying_an_invoice_removes_it_from_the_overdue_list() -> None:
    ledger = Ledger()

    ledger.pay(DEFAULT_INVOICE_ID, Decimal("1284.50"))

    assert DEFAULT_INVOICE_ID not in {invoice.id for invoice in ledger.overdue()}


def test_an_invoice_cannot_be_paid_twice() -> None:
    ledger = Ledger()
    ledger.pay(DEFAULT_INVOICE_ID, Decimal("1284.50"))

    with pytest.raises(LedgerError, match="already paid"):
        ledger.pay(DEFAULT_INVOICE_ID, Decimal("1284.50"))


def test_a_payment_must_match_the_invoice_exactly() -> None:
    ledger = Ledger()

    with pytest.raises(LedgerError, match="does not match"):
        ledger.pay(DEFAULT_INVOICE_ID, Decimal("1000.00"))

    assert not ledger.payments


# --- the tools -------------------------------------------------------------


def test_the_demo_tools_span_the_risk_range() -> None:
    risks = {tool.name: tool.risk for tool in build_demo_tools()}

    assert risks["lookup_invoice"] is RiskLevel.READ
    assert risks["add_invoice_note"] is RiskLevel.WRITE
    assert risks["pay_invoice"] is RiskLevel.CRITICAL


async def test_a_missing_invoice_comes_back_as_a_recoverable_tool_error() -> None:
    registry = ToolRegistry(build_demo_tools(Ledger()))

    result = await registry.execute(
        ToolCall(id="c1", name="lookup_invoice", arguments={"invoice_id": "INV-9999"})
    )

    assert result.is_error
    assert "no invoice with id 'INV-9999'" in result.content


async def test_a_payment_the_model_got_wrong_is_refused_by_the_tool() -> None:
    ledger = Ledger()
    registry = ToolRegistry(build_demo_tools(ledger))

    result = await registry.execute(
        ToolCall(
            id="c1",
            name="pay_invoice",
            arguments={"invoice_id": DEFAULT_INVOICE_ID, "amount_eur": "1.00"},
        )
    )

    assert result.is_error
    assert not ledger.payments


def test_money_is_offered_to_the_model_as_a_constrained_string() -> None:
    """A JSON number is a double; a payment amount must survive the round trip."""
    pay = next(tool for tool in build_demo_tools() if tool.name == "pay_invoice")

    amount = pay.spec().input_schema["properties"]["amount_eur"]

    assert amount["type"] == "string"
    assert amount["pattern"] == r"^\d{1,9}\.\d{2}$"


def test_an_amount_the_schema_rejects_never_reaches_the_ledger() -> None:
    pay = next(tool for tool in build_demo_tools() if tool.name == "pay_invoice")

    with pytest.raises(ValidationError):
        pay.arguments.model_validate({"invoice_id": DEFAULT_INVOICE_ID, "amount_eur": "1284.5"})


# --- the scripted provider -------------------------------------------------


async def test_the_provider_opens_by_reading_the_invoice_named_in_the_goal() -> None:
    completion = await DemoLLMProvider().complete(goal_request("Settle invoice INV-1051 today"))

    assert completion.stop_reason is StopReason.TOOL_USE
    assert [call.name for call in completion.tool_calls] == ["lookup_invoice"]
    assert completion.tool_calls[0].arguments == {"invoice_id": "INV-1051"}


async def test_the_provider_pays_the_amount_the_lookup_returned() -> None:
    lookup = ToolCall(id="c1", name="lookup_invoice", arguments={"invoice_id": DEFAULT_INVOICE_ID})
    conversation = (
        Message(role=Role.ASSISTANT, tool_calls=(lookup,)),
        Message.with_tool_results(
            [
                ToolResult(
                    call_id="c1",
                    content=f'{{"invoice_id":"{DEFAULT_INVOICE_ID}","amount_eur":"1284.50"}}',
                )
            ]
        ),
    )

    completion = await DemoLLMProvider().complete(
        goal_request(f"Settle invoice {DEFAULT_INVOICE_ID}", *conversation)
    )

    assert completion.tool_calls[0].name == "pay_invoice"
    assert completion.tool_calls[0].arguments["amount_eur"] == "1284.50"


async def test_a_failed_lookup_stops_the_run_instead_of_paying_an_invented_amount() -> None:
    lookup = ToolCall(id="c1", name="lookup_invoice", arguments={"invoice_id": "INV-9999"})
    conversation = (
        Message(role=Role.ASSISTANT, tool_calls=(lookup,)),
        Message.with_tool_results(
            [ToolResult(call_id="c1", content="no invoice with id 'INV-9999'", is_error=True)]
        ),
    )

    completion = await DemoLLMProvider().complete(
        goal_request("Settle invoice INV-9999", *conversation)
    )

    assert completion.stop_reason is StopReason.END_TURN
    assert not completion.tool_calls
    assert "failed" in completion.text


OVERDUE_GOAL = "Review the overdue supplier invoices and flag the worst one"


async def test_the_provider_opens_an_overdue_review_by_listing_the_invoices() -> None:
    completion = await DemoLLMProvider().complete(goal_request(OVERDUE_GOAL))

    assert completion.stop_reason is StopReason.TOOL_USE
    assert [call.name for call in completion.tool_calls] == ["list_overdue_invoices"]


async def test_the_provider_annotates_the_invoice_the_listing_ranked_worst() -> None:
    """The note's content comes out of the tool result, not out of the script."""
    listing = ToolCall(id="c1", name="list_overdue_invoices", arguments={"limit": 5})
    conversation = (
        Message(role=Role.ASSISTANT, tool_calls=(listing,)),
        Message.with_tool_results(
            [
                ToolResult(
                    call_id="c1",
                    content=(
                        '{"count":1,"invoices":[{"invoice_id":"INV-1051","days_overdue":31}]}'
                    ),
                )
            ]
        ),
    )

    completion = await DemoLLMProvider().complete(goal_request(OVERDUE_GOAL, *conversation))

    assert completion.tool_calls[0].name == "add_invoice_note"
    assert completion.tool_calls[0].arguments["invoice_id"] == "INV-1051"
    assert "31 days overdue" in completion.tool_calls[0].arguments["note"]


async def test_an_empty_overdue_listing_ends_the_run_rather_than_annotating_nothing() -> None:
    listing = ToolCall(id="c1", name="list_overdue_invoices", arguments={"limit": 5})
    conversation = (
        Message(role=Role.ASSISTANT, tool_calls=(listing,)),
        Message.with_tool_results([ToolResult(call_id="c1", content='{"count":0,"invoices":[]}')]),
    )

    completion = await DemoLLMProvider().complete(goal_request(OVERDUE_GOAL, *conversation))

    assert completion.stop_reason is StopReason.END_TURN
    assert not completion.tool_calls


async def test_an_unplanned_goal_is_answered_rather_than_improvised() -> None:
    completion = await DemoLLMProvider().complete(goal_request("Write next year's budget"))

    assert completion.stop_reason is StopReason.END_TURN
    assert "EAP_LLM_PROVIDER=anthropic" in completion.text


def test_calls_and_results_are_joined_across_the_pause() -> None:
    """A checkpoint puts several messages between a call and its result."""
    call = ToolCall(id="c1", name="lookup_invoice", arguments={})
    messages = (
        Message(role=Role.USER, content="Task goal:\nSettle invoice INV-1043"),
        Message(role=Role.ASSISTANT, tool_calls=(call,)),
        Message(role=Role.USER, content="…"),
        Message.with_tool_results([ToolResult(call_id="c1", content='{"amount_eur":"1.00"}')]),
    )

    outcomes = _tool_outcomes(messages)

    assert outcomes == (
        ToolOutcome(call=call, result=ToolResult(call_id="c1", content='{"amount_eur":"1.00"}')),
    )


# --- wiring ----------------------------------------------------------------


def test_the_demo_backend_is_selected_by_configuration() -> None:
    assert build_llm_client(settings(llm_provider="demo")).provider_name == "demo"


def test_no_tools_are_registered_unless_the_deployment_asks() -> None:
    app = create_app()

    assert app.state.tool_registry.names == ()


def test_the_demo_tools_are_registered_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from enterprise_agent_platform.config import get_settings

    monkeypatch.setenv("EAP_DEMO_TOOLS", "true")
    get_settings.cache_clear()
    try:
        app = create_app()
        with TestClient(app):
            assert "pay_invoice" in app.state.tool_registry.names
    finally:
        get_settings.cache_clear()


def test_the_demo_is_rejected_in_production() -> None:
    with pytest.raises(ValidationError, match="EAP_LLM_PROVIDER=demo"):
        settings(environment="production", llm_provider="demo", task_store="postgres")

    with pytest.raises(ValidationError, match="EAP_DEMO_TOOLS=true"):
        settings(
            environment="production",
            llm_provider="anthropic",
            task_store="postgres",
            database_url="postgresql://eap:secret@localhost:5432/eap",
            demo_tools=True,
        )


# --- the walkthrough -------------------------------------------------------


async def test_the_walkthrough_pauses_on_the_payment_and_finishes_once_approved() -> None:
    ledger = Ledger()

    result = await run_walkthrough(build_demo_app(ledger), ledger, out=io.StringIO())

    assert result.paused_on == ("pay_invoice",)
    assert result.final_status == "completed"
    # Three model calls: read the invoice, ask to pay it, answer. The pause costs
    # one of them, so a resumed run reporting two would mean the budget reset.
    assert (result.steps, result.tool_calls) == (3, 2)
    assert ledger.payments[DEFAULT_INVOICE_ID].amount_eur == Decimal("1284.50")


async def test_the_walkthrough_fails_loudly_if_the_gate_stops_gating() -> None:
    """The demo is a check, not just a script: it must notice a missing pause."""
    ledger = Ledger()
    app = build_demo_app(ledger)
    # A deployment that auto-approves everything, which is exactly the
    # misconfiguration the transcript would otherwise keep quiet about.
    app.state.agent_runner = AgentRunner(
        app.state.llm_client,
        app.state.tool_registry,
        app.state.task_repository,
        auto_approve_up_to=RiskLevel.CRITICAL,
    )

    with pytest.raises(WalkthroughError, match="should have paused"):
        await run_walkthrough(app, ledger, out=io.StringIO())
