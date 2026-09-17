"""LLM client: the single entry point agent code uses to call a model.

Wraps any ``LLMProvider`` with the concerns every call needs regardless of
vendor: a hard timeout, structured-output validation, and one log record per
call with provider, model, token usage, and latency. Records are emitted in the
caller's context, so calls made while handling an HTTP request carry its
``request_id``. Prompt and response text are never logged — they may contain
customer data — only their sizes.
"""

from __future__ import annotations

import asyncio
import logging
import time

from pydantic import BaseModel, ValidationError

from enterprise_agent_platform.llm.errors import (
    LLMError,
    LLMRefusalError,
    LLMTimeoutError,
    StructuredOutputError,
)
from enterprise_agent_platform.llm.models import Completion, CompletionRequest, StopReason
from enterprise_agent_platform.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, provider: LLMProvider, *, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._provider = provider
        self._timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def aclose(self) -> None:
        """Release the provider's resources; the client is unusable afterwards."""
        await self._provider.aclose()

    async def complete(self, request: CompletionRequest, *, operation: str) -> Completion:
        """Call the provider once.

        ``operation`` names the calling step (e.g. ``"plan"``) so logs can be
        aggregated per use case rather than per prompt.

        Raises:
            LLMTimeoutError: if the provider does not answer within the timeout.
            LLMError: any provider failure, unchanged.
        """
        log_fields: dict[str, object] = {
            "provider": self.provider_name,
            "operation": operation,
            "message_count": len(request.messages),
            "prompt_chars": sum(len(m.content) for m in request.messages),
            "tool_count": len(request.tools),
        }
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                completion = await self._provider.complete(request)
        except TimeoutError:
            error: Exception = LLMTimeoutError(
                f"{self.provider_name} did not respond within {self._timeout_seconds}s"
            )
            self._log_failure(error, started, log_fields)
            raise error from None
        except Exception as exc:
            self._log_failure(exc, started, log_fields)
            raise

        logger.info(
            "llm.call.completed",
            extra={
                **log_fields,
                "model": completion.model,
                "stop_reason": completion.stop_reason.value,
                "input_tokens": completion.usage.input_tokens,
                "output_tokens": completion.usage.output_tokens,
                "response_chars": len(completion.text),
                # Tool names are schema, not user data, so they are safe to log
                # and show which capabilities the model actually reaches for.
                "tool_calls": [call.name for call in completion.tool_calls],
                "duration_ms": _elapsed_ms(started),
            },
        )
        return completion

    async def complete_structured[T: BaseModel](
        self, request: CompletionRequest, schema: type[T], *, operation: str
    ) -> T:
        """Call the provider and validate the response as ``schema``.

        The schema is attached to the request so providers with native
        structured outputs can enforce it; validation here is the guarantee.

        Raises:
            ValueError: if the request also offers tools — a turn that ends in a
                tool call produces no JSON to validate, so the two modes are
                kept apart instead of failing confusingly at parse time.
            LLMRefusalError: if the model declined to answer.
            StructuredOutputError: if the output was truncated or does not match.
        """
        if request.tools:
            raise ValueError("complete_structured cannot be combined with tools")
        request = request.model_copy(update={"output_schema": schema.model_json_schema()})
        completion = await self.complete(request, operation=operation)

        if completion.stop_reason is StopReason.REFUSAL:
            raise LLMRefusalError(f"model refused structured output for '{operation}'")
        if completion.stop_reason is StopReason.MAX_TOKENS:
            raise StructuredOutputError(
                f"output for '{operation}' was truncated at max_tokens={request.max_tokens}",
                raw_text=completion.text,
            )
        try:
            return schema.model_validate_json(completion.text)
        except ValidationError as exc:
            logger.warning(
                "llm.structured_output.invalid",
                extra={
                    "operation": operation,
                    "schema": schema.__name__,
                    "error_count": exc.error_count(),
                },
            )
            raise StructuredOutputError(
                f"output for '{operation}' does not match {schema.__name__}: "
                f"{exc.error_count()} validation error(s)",
                raw_text=completion.text,
            ) from exc

    def _log_failure(self, error: Exception, started: float, fields: dict[str, object]) -> None:
        logger.warning(
            "llm.call.failed",
            extra={
                **fields,
                "error_type": type(error).__name__,
                "retryable": isinstance(error, LLMError) and error.retryable,
                "duration_ms": _elapsed_ms(started),
            },
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
