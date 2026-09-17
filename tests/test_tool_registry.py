"""Tests for tool definitions and the registry that runs them.

The behaviour that matters here is what the model is allowed to see and what it
is protected from: recoverable mistakes must come back as tool results it can
act on, while handler internals must not.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import BaseModel, Field

from enterprise_agent_platform.llm.models import ToolCall
from enterprise_agent_platform.tools.models import RiskLevel, Tool, requires_approval
from enterprise_agent_platform.tools.registry import ToolRegistry


class InvoiceArgs(BaseModel):
    invoice_id: str
    amount: float = Field(gt=0)


class ReportArgs(BaseModel):
    period: str = "month"


async def _lookup(args: InvoiceArgs) -> str:
    return f"invoice {args.invoice_id} for {args.amount}"


def invoice_tool(
    *,
    risk: RiskLevel = RiskLevel.READ,
    handler: object = None,
) -> Tool[InvoiceArgs]:
    return Tool(
        name="lookup_invoice",
        description="Look up an invoice by id.",
        arguments=InvoiceArgs,
        risk=risk,
        handler=handler or _lookup,  # type: ignore[arg-type]
    )


def call(**arguments: object) -> ToolCall:
    return ToolCall(id="call_1", name="lookup_invoice", arguments=dict(arguments))


def test_tool_declares_its_arguments_as_a_json_schema() -> None:
    spec = invoice_tool().spec()

    assert spec.name == "lookup_invoice"
    assert spec.input_schema["properties"]["invoice_id"]["type"] == "string"
    assert spec.input_schema["required"] == ["invoice_id", "amount"]


def test_a_tool_without_a_description_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs a description"):
        Tool(
            name="lookup_invoice",
            description="  ",
            arguments=InvoiceArgs,
            risk=RiskLevel.READ,
            handler=_lookup,
        )


def test_a_name_the_model_cannot_emit_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid tool name"):
        Tool(
            name="lookup invoice!",
            description="Look up an invoice.",
            arguments=InvoiceArgs,
            risk=RiskLevel.READ,
            handler=_lookup,
        )


def test_registering_the_same_name_twice_is_a_programmer_error() -> None:
    registry = ToolRegistry([invoice_tool()])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(invoice_tool())


def test_specs_cover_every_registered_tool() -> None:
    registry = ToolRegistry(
        [
            invoice_tool(),
            Tool(
                name="build_report",
                description="Build a report.",
                arguments=ReportArgs,
                risk=RiskLevel.WRITE,
                handler=lambda args: asyncio.sleep(0, result=args.period),
            ),
        ]
    )

    assert registry.names == ("lookup_invoice", "build_report")
    assert [spec.name for spec in registry.specs()] == ["lookup_invoice", "build_report"]


async def test_a_valid_call_runs_the_handler_with_a_typed_model() -> None:
    received: list[InvoiceArgs] = []

    async def handler(args: InvoiceArgs) -> str:
        received.append(args)
        return "paid"

    registry = ToolRegistry([invoice_tool(handler=handler)])

    result = await registry.execute(call(invoice_id="INV-1", amount=10))

    assert result.call_id == "call_1"
    assert result.content == "paid"
    assert not result.is_error
    assert received == [InvoiceArgs(invoice_id="INV-1", amount=10)]


async def test_an_unknown_tool_is_reported_back_instead_of_raising() -> None:
    registry = ToolRegistry([invoice_tool()])

    result = await registry.execute(ToolCall(id="call_2", name="delete_everything"))

    assert result.is_error
    assert result.call_id == "call_2"
    # The model is told what it may call instead, so it can recover this turn.
    assert "lookup_invoice" in result.content


async def test_invalid_arguments_tell_the_model_which_field_is_wrong() -> None:
    registry = ToolRegistry([invoice_tool()])

    result = await registry.execute(call(invoice_id="INV-1", amount=-5))

    assert result.is_error
    assert "amount" in result.content


async def test_invalid_arguments_do_not_reach_the_handler() -> None:
    calls: list[InvoiceArgs] = []

    async def handler(args: InvoiceArgs) -> str:
        calls.append(args)
        return "ok"

    registry = ToolRegistry([invoice_tool(handler=handler)])

    await registry.execute(call(invoice_id="INV-1"))

    assert calls == []


async def test_a_failing_handler_does_not_leak_its_internals_to_the_model() -> None:
    async def handler(args: InvoiceArgs) -> str:
        raise RuntimeError("postgres://user:secret@ledger-db/internal")

    registry = ToolRegistry([invoice_tool(handler=handler)])

    result = await registry.execute(call(invoice_id="INV-1", amount=10))

    assert result.is_error
    assert "secret" not in result.content
    assert "lookup_invoice" in result.content


async def test_a_hanging_tool_is_abandoned_at_its_timeout() -> None:
    async def handler(args: InvoiceArgs) -> str:
        await asyncio.sleep(10)
        return "too late"

    registry = ToolRegistry([invoice_tool(handler=handler)], timeout_seconds=0.05)

    result = await registry.execute(call(invoice_id="INV-1", amount=10))

    assert result.is_error
    assert "timed out" in result.content


async def test_an_oversized_result_is_truncated_before_it_reaches_the_prompt() -> None:
    async def handler(args: InvoiceArgs) -> str:
        return "x" * 10_000

    registry = ToolRegistry([invoice_tool(handler=handler)], max_result_chars=100)

    result = await registry.execute(call(invoice_id="INV-1", amount=10))

    assert len(result.content) == 100
    assert result.content.endswith("[truncated]")


async def test_calls_from_one_turn_run_concurrently_and_stay_in_order() -> None:
    async def slow(args: InvoiceArgs) -> str:
        await asyncio.sleep(0.05)
        return args.invoice_id

    registry = ToolRegistry([invoice_tool(handler=slow)])
    calls = [
        ToolCall(
            id=f"call_{index}",
            name="lookup_invoice",
            arguments={"invoice_id": f"INV-{index}", "amount": 1},
        )
        for index in range(3)
    ]

    started = time.perf_counter()
    results = await registry.execute_all(calls)
    elapsed = time.perf_counter() - started

    assert [result.content for result in results] == ["INV-0", "INV-1", "INV-2"]
    assert elapsed < 0.12  # sequential execution would take at least 0.15s


@pytest.mark.parametrize(
    ("risk", "threshold", "expected"),
    [
        (RiskLevel.READ, RiskLevel.READ, False),
        (RiskLevel.WRITE, RiskLevel.READ, True),
        (RiskLevel.WRITE, RiskLevel.WRITE, False),
        (RiskLevel.CRITICAL, RiskLevel.WRITE, True),
        (RiskLevel.CRITICAL, RiskLevel.CRITICAL, False),
    ],
)
def test_approval_is_decided_by_a_risk_threshold(
    risk: RiskLevel, threshold: RiskLevel, expected: bool
) -> None:
    assert requires_approval(invoice_tool(risk=risk), auto_approve_up_to=threshold) is expected
