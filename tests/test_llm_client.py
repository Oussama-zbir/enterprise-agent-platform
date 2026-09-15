"""Tests for the provider-neutral LLM client, error taxonomy, and fake provider."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import BaseModel, ValidationError

from enterprise_agent_platform import request_context
from enterprise_agent_platform.llm.client import LLMClient
from enterprise_agent_platform.llm.errors import (
    LLMError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    StructuredOutputError,
)
from enterprise_agent_platform.llm.models import (
    Completion,
    CompletionRequest,
    Message,
    Role,
    StopReason,
    TokenUsage,
)
from enterprise_agent_platform.llm.provider import FakeLLMProvider
from enterprise_agent_platform.logging import RequestContextFilter

SECRET_PROMPT = "Summarise account 4111-1111 for supplier Acme"


class Plan(BaseModel):
    steps: list[str]
    requires_approval: bool


def make_request(content: str = SECRET_PROMPT) -> CompletionRequest:
    return CompletionRequest(messages=(Message(role=Role.USER, content=content),))


def completion(text: str, stop_reason: StopReason) -> Completion:
    return Completion(
        text=text,
        model="fake-model",
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=5, output_tokens=5),
    )


@pytest.fixture
def llm_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture LLM logs as they would be emitted inside request ``req-llm``."""
    caplog.set_level("INFO", logger="enterprise_agent_platform.llm")
    caplog.handler.addFilter(RequestContextFilter())
    token = request_context._request_id.set("req-llm")
    yield caplog
    request_context._request_id.reset(token)


def records(caplog: pytest.LogCaptureFixture, message: str) -> list[dict[str, object]]:
    return [r.__dict__ for r in caplog.records if r.getMessage() == message]


async def test_complete_logs_usage_and_request_id_without_prompt_text(
    llm_logs: pytest.LogCaptureFixture,
) -> None:
    provider = FakeLLMProvider(["three word answer"])
    client = LLMClient(provider, timeout_seconds=1)

    result = await client.complete(make_request(), operation="plan")

    assert result.text == "three word answer"
    assert provider.requests == [make_request()]
    [record] = records(llm_logs, "llm.call.completed")
    assert record["request_id"] == "req-llm"
    assert (record["provider"], record["model"], record["operation"]) == (
        "fake",
        "fake-model",
        "plan",
    )
    assert (record["input_tokens"], record["output_tokens"]) == (6, 3)
    assert record["stop_reason"] == "end_turn"
    assert isinstance(record["duration_ms"], float)
    assert SECRET_PROMPT not in llm_logs.text
    assert "4111" not in str(record)


async def test_slow_provider_raises_retryable_timeout(llm_logs: pytest.LogCaptureFixture) -> None:
    client = LLMClient(FakeLLMProvider(["late"], delay_seconds=1), timeout_seconds=0.01)

    with pytest.raises(LLMTimeoutError) as exc_info:
        await client.complete(make_request(), operation="plan")

    assert exc_info.value.retryable
    [record] = records(llm_logs, "llm.call.failed")
    assert (record["error_type"], record["retryable"]) == ("LLMTimeoutError", True)
    assert record["request_id"] == "req-llm"


async def test_provider_errors_propagate_unchanged_and_are_logged(
    llm_logs: pytest.LogCaptureFixture,
) -> None:
    rate_limited = LLMRateLimitError("slow down", retry_after_seconds=12.0)
    client = LLMClient(FakeLLMProvider([rate_limited]), timeout_seconds=1)

    with pytest.raises(LLMRateLimitError) as exc_info:
        await client.complete(make_request(), operation="plan")

    assert exc_info.value is rate_limited
    assert exc_info.value.retry_after_seconds == 12.0
    [record] = records(llm_logs, "llm.call.failed")
    assert (record["error_type"], record["retryable"]) == ("LLMRateLimitError", True)


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (LLMTimeoutError("t"), True),
        (LLMRateLimitError("r"), True),
        (LLMUnavailableError("u"), True),
        (LLMRequestError("bad request"), False),
        (LLMRefusalError("refused"), False),
        (StructuredOutputError("invalid", raw_text="{}"), False),
    ],
)
def test_retryable_classification(error: LLMError, retryable: bool) -> None:
    assert error.retryable is retryable


async def test_structured_output_is_validated_and_schema_sent_to_provider() -> None:
    provider = FakeLLMProvider(
        ['{"steps": ["fetch invoices", "match"], "requires_approval": true}']
    )
    client = LLMClient(provider, timeout_seconds=1)

    plan = await client.complete_structured(make_request(), Plan, operation="plan")

    assert plan == Plan(steps=["fetch invoices", "match"], requires_approval=True)
    assert provider.requests[0].output_schema == Plan.model_json_schema()


@pytest.mark.parametrize(
    "text",
    [
        "Sure! Here is the plan.",
        '{"steps": "not a list", "requires_approval": true}',
        '```json\n{"steps": [], "requires_approval": false}\n```',
    ],
)
async def test_non_conforming_structured_output_raises_with_raw_text(
    text: str, llm_logs: pytest.LogCaptureFixture
) -> None:
    client = LLMClient(FakeLLMProvider([text]), timeout_seconds=1)

    with pytest.raises(StructuredOutputError, match="does not match Plan") as exc_info:
        await client.complete_structured(make_request(), Plan, operation="plan")

    assert exc_info.value.raw_text == text
    [record] = records(llm_logs, "llm.structured_output.invalid")
    assert record["schema"] == "Plan"


async def test_truncated_structured_output_is_reported_as_truncation() -> None:
    truncated = completion('{"steps": ["fetch inv', StopReason.MAX_TOKENS)
    client = LLMClient(FakeLLMProvider([truncated]), timeout_seconds=1)

    with pytest.raises(StructuredOutputError, match="truncated at max_tokens=1024"):
        await client.complete_structured(make_request(), Plan, operation="plan")


async def test_refusal_is_not_parsed_as_structured_output() -> None:
    client = LLMClient(FakeLLMProvider([completion("", StopReason.REFUSAL)]), timeout_seconds=1)

    with pytest.raises(LLMRefusalError):
        await client.complete_structured(make_request(), Plan, operation="plan")


async def test_fake_provider_fails_loudly_when_script_is_exhausted() -> None:
    client = LLMClient(FakeLLMProvider([]), timeout_seconds=1)

    with pytest.raises(RuntimeError, match="script exhausted"):
        await client.complete(make_request(), operation="plan")


def test_client_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="positive"):
        LLMClient(FakeLLMProvider([]), timeout_seconds=0)


@pytest.mark.parametrize(
    "messages",
    [
        (),
        (Message(role=Role.ASSISTANT, content="I will start"),),
    ],
)
def test_request_requires_conversation_starting_with_user(
    messages: tuple[Message, ...],
) -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(messages=messages)
