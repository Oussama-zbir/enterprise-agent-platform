"""Tests for the Anthropic adapter.

A mock HTTP transport stands in for the API, so these exercise the real SDK
request building, response parsing, and exception types without a network call
or a key. Two things matter here: the request the platform actually sends, and
that every vendor failure lands on the right side of the ``retryable`` line.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from pydantic import BaseModel

from enterprise_agent_platform.llm.anthropic_provider import AnthropicProvider, strict_json_schema
from enterprise_agent_platform.llm.errors import (
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from enterprise_agent_platform.llm.models import (
    Completion,
    CompletionRequest,
    Message,
    Role,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolSpec,
)

Handler = Callable[[httpx2.Request], httpx2.Response]

PROMPT = "Reconcile supplier payments for March"


class Step(BaseModel):
    name: str


class Plan(BaseModel):
    steps: list[Step]
    requires_approval: bool = False


class InvoiceArgs(BaseModel):
    invoice_id: str


class ReportArgs(BaseModel):
    """Every field optional — Pydantic then omits ``required`` entirely."""

    period: str = "month"


INVOICE_TOOL = ToolSpec(
    name="lookup_invoice",
    description="Look up an invoice by id.",
    input_schema=InvoiceArgs.model_json_schema(),
)


def make_request(**overrides: Any) -> CompletionRequest:
    return CompletionRequest(messages=(Message(role=Role.USER, content=PROMPT),), **overrides)


def message_response(
    *,
    content: list[dict[str, object]] | None = None,
    stop_reason: str | None = "end_turn",
    model: str = "claude-opus-5",
    input_tokens: int = 31,
    output_tokens: int = 7,
) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content if content is not None else [{"type": "text", "text": "ok"}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    )


def error_response(status: int, *, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(
        status,
        headers=headers or {},
        json={"type": "error", "error": {"type": "invalid_request_error", "message": PROMPT}},
    )


def build_provider(handler: Handler) -> AnthropicProvider:
    client = AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicProvider(client, model="claude-opus-5")


def responding(response: httpx2.Response) -> tuple[AnthropicProvider, list[httpx2.Request]]:
    """A provider that always answers with ``response``, recording what it was sent."""
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return response

    return build_provider(handler), sent


def failing(response: httpx2.Response) -> AnthropicProvider:
    return responding(response)[0]


async def test_request_is_translated_to_the_messages_api() -> None:
    provider, sent = responding(message_response())

    await provider.complete(make_request(system="You are an operations agent", max_tokens=256))

    [request] = sent
    body = json.loads(request.content)
    assert request.url.path == "/v1/messages"
    assert body["model"] == "claude-opus-5"
    assert body["max_tokens"] == 256
    assert body["system"] == "You are an operations agent"
    assert body["messages"] == [{"role": "user", "content": PROMPT}]
    assert "output_config" not in body


async def test_response_is_mapped_to_a_provider_neutral_completion() -> None:
    provider, _ = responding(message_response(input_tokens=31, output_tokens=7))

    result = await provider.complete(make_request())

    assert result == Completion(
        text="ok",
        model="claude-opus-5",
        stop_reason=StopReason.END_TURN,
        usage=TokenUsage(input_tokens=31, output_tokens=7),
    )


async def test_only_text_blocks_contribute_to_the_completion_text() -> None:
    provider, _ = responding(
        message_response(
            content=[
                {"type": "thinking", "thinking": "internal reasoning", "signature": "sig"},
                {"type": "text", "text": "first "},
                {"type": "text", "text": "second"},
            ]
        )
    )

    result = await provider.complete(make_request())

    assert result.text == "first second"


@pytest.mark.parametrize(
    ("api_stop_reason", "expected"),
    [
        ("end_turn", StopReason.END_TURN),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("stop_sequence", StopReason.STOP_SEQUENCE),
        ("refusal", StopReason.REFUSAL),
    ],
)
async def test_stop_reasons_are_mapped(api_stop_reason: str, expected: StopReason) -> None:
    provider, _ = responding(message_response(stop_reason=api_stop_reason))

    result = await provider.complete(make_request())

    assert result.stop_reason is expected


@pytest.mark.parametrize("stop_reason", ["model_context_window_exceeded", "pause_turn", None])
async def test_unsupported_stop_reason_fails_loudly(stop_reason: str | None) -> None:
    provider, _ = responding(message_response(stop_reason=stop_reason))

    with pytest.raises(LLMRequestError, match="unsupported stop reason") as exc_info:
        await provider.complete(make_request())

    assert not exc_info.value.retryable


async def test_output_schema_is_sent_as_a_closed_json_schema() -> None:
    provider, sent = responding(message_response(content=[{"type": "text", "text": "{}"}]))

    await provider.complete(make_request(output_schema=Plan.model_json_schema()))

    schema_format = json.loads(sent[0].content)["output_config"]["format"]
    assert schema_format["type"] == "json_schema"
    assert schema_format["schema"]["additionalProperties"] is False
    assert schema_format["schema"]["$defs"]["Step"]["additionalProperties"] is False


def test_strict_json_schema_keeps_an_explicit_additional_properties() -> None:
    schema = strict_json_schema({"type": "object", "additionalProperties": True})

    assert schema["additionalProperties"] is True


def test_strict_json_schema_does_not_change_optionality() -> None:
    schema = strict_json_schema(Plan.model_json_schema())

    assert schema["required"] == ["steps"]


@pytest.mark.parametrize(
    ("status", "expected", "retryable"),
    [
        (400, LLMRequestError, False),
        (401, LLMRequestError, False),
        (404, LLMRequestError, False),
        (422, LLMRequestError, False),
        (408, LLMTimeoutError, True),
        (500, LLMUnavailableError, True),
        (529, LLMUnavailableError, True),
    ],
)
async def test_http_status_is_mapped_to_the_error_taxonomy(
    status: int, expected: type[LLMError], retryable: bool
) -> None:
    provider = failing(error_response(status))

    with pytest.raises(expected) as exc_info:
        await provider.complete(make_request())

    assert exc_info.value.retryable is retryable


async def test_provider_error_message_does_not_echo_request_content() -> None:
    provider = failing(error_response(400))

    with pytest.raises(LLMRequestError) as exc_info:
        await provider.complete(make_request())

    assert PROMPT not in str(exc_info.value)


async def test_rate_limit_carries_the_retry_after_delay() -> None:
    provider = failing(error_response(429, headers={"retry-after": "7"}))

    with pytest.raises(LLMRateLimitError) as exc_info:
        await provider.complete(make_request())

    assert exc_info.value.retry_after_seconds == 7.0
    assert exc_info.value.retryable


async def test_rate_limit_tolerates_a_non_numeric_retry_after() -> None:
    provider = failing(
        error_response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    )

    with pytest.raises(LLMRateLimitError) as exc_info:
        await provider.complete(make_request())

    assert exc_info.value.retry_after_seconds is None


async def test_connection_failure_is_retryable_unavailability() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("no route to host", request=request)

    with pytest.raises(LLMUnavailableError) as exc_info:
        await build_provider(handler).complete(make_request())

    assert exc_info.value.retryable


async def test_transport_timeout_is_a_timeout_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("too slow", request=request)

    with pytest.raises(LLMTimeoutError) as exc_info:
        await build_provider(handler).complete(make_request())

    assert exc_info.value.retryable


async def test_a_failed_call_is_not_retried_inside_the_adapter() -> None:
    attempts: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(request)
        return error_response(500)

    with pytest.raises(LLMUnavailableError):
        await build_provider(handler).complete(make_request())

    assert len(attempts) == 1


async def test_tools_are_declared_with_closed_strict_schemas() -> None:
    provider, sent = responding(message_response())

    await provider.complete(make_request(tools=(INVOICE_TOOL,)))

    [tool] = json.loads(sent[0].content)["tools"]
    assert tool["name"] == "lookup_invoice"
    assert tool["description"] == "Look up an invoice by id."
    assert tool["input_schema"]["additionalProperties"] is False
    # Constrained decoding: the model can only produce arguments that validate.
    assert tool["strict"] is True


def test_strict_json_schema_declares_required_even_when_nothing_is_required() -> None:
    schema = strict_json_schema(ReportArgs.model_json_schema())

    assert schema["required"] == []


async def test_a_tool_use_response_becomes_tool_calls() -> None:
    provider, _ = responding(
        message_response(
            content=[
                {"type": "text", "text": "checking the ledger"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup_invoice",
                    "input": {"invoice_id": "INV-1"},
                },
            ],
            stop_reason="tool_use",
        )
    )

    result = await provider.complete(make_request(tools=(INVOICE_TOOL,)))

    assert result.stop_reason is StopReason.TOOL_USE
    assert result.text == "checking the ledger"
    assert result.tool_calls == (
        ToolCall(id="toolu_1", name="lookup_invoice", arguments={"invoice_id": "INV-1"}),
    )


async def test_tool_calls_and_their_results_are_sent_back_as_content_blocks() -> None:
    provider, sent = responding(message_response())
    completion = Completion(
        text="checking",
        model="claude-opus-5",
        stop_reason=StopReason.TOOL_USE,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        tool_calls=(ToolCall(id="toolu_1", name="lookup_invoice", arguments={"invoice_id": "X"}),),
    )
    results = (ToolResult(call_id="toolu_1", content="no such invoice", is_error=True),)

    await provider.complete(
        CompletionRequest(
            messages=(
                Message(role=Role.USER, content=PROMPT),
                Message.from_completion(completion),
                Message.with_tool_results(results),
            ),
            tools=(INVOICE_TOOL,),
        )
    )

    messages = json.loads(sent[0].content)["messages"]
    assert messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "checking"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "lookup_invoice",
                "input": {"invoice_id": "X"},
            },
        ],
    }
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "no such invoice",
                "is_error": True,
            }
        ],
    }


async def test_aclose_releases_the_underlying_client() -> None:
    provider, _ = responding(message_response())

    await provider.aclose()

    # The connection pool is gone, so a later call reports the backend as
    # unreachable instead of quietly reopening one.
    with pytest.raises(LLMUnavailableError):
        await provider.complete(make_request())
