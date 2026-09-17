"""Anthropic adapter for the ``LLMProvider`` port.

Translates in both directions: provider-neutral requests into Messages API
calls, and SDK responses and exceptions into ``Completion`` values and the
platform's error taxonomy. Everything vendor-specific — model IDs, JSON Schema
dialect, HTTP status semantics — is confined to this module, so agent code
depends only on the port.

The SDK's built-in retry loop is disabled where the client is constructed
(``max_retries=0``). Retries are the caller's decision: only it knows the task,
its deadline, and its budget, and a hidden retry would also inflate the latency
recorded by ``LLMClient`` for a single logical call.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    AsyncAnthropicBedrockMantle,
    Omit,
    RateLimitError,
    omit,
)
from anthropic.types import (
    ContentBlockParam,
    MessageParam,
    OutputConfigParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)
from anthropic.types import Message as AnthropicMessage
from anthropic.types.tool_param import InputSchema

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
)

AsyncAnthropicClient = AsyncAnthropic | AsyncAnthropicBedrockMantle
"""Either Messages API backend; both expose the same ``messages`` resource."""

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "refusal": StopReason.REFUSAL,
    "tool_use": StopReason.TOOL_USE,
}
"""Stop reasons this platform can serve. Anything else (including
``model_context_window_exceeded``) is reported as a request error rather than
quietly mislabelled as a normal end of turn."""


class AnthropicProvider:
    """``LLMProvider`` backed by the Anthropic Messages API or Bedrock.

    ``client`` is injected rather than built here so deployments choose the
    backend and tests can supply a mock HTTP transport.
    """

    def __init__(
        self,
        client: AsyncAnthropicClient,
        *,
        model: str,
        name: str = "anthropic",
    ) -> None:
        self._client = client
        self._model = model
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: CompletionRequest) -> Completion:
        try:
            message = await self._client.messages.create(
                model=self._model,
                max_tokens=request.max_tokens,
                system=request.system if request.system is not None else omit,
                messages=[_message_param(message) for message in request.messages],
                output_config=_output_config(request),
                tools=_tool_params(request),
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"{self._name} request timed out") from exc
        except APIConnectionError as exc:
            raise LLMUnavailableError(f"cannot reach {self._name}") from exc
        except RateLimitError as exc:
            raise LLMRateLimitError(
                f"{self._name} rate limit exceeded",
                retry_after_seconds=_retry_after_seconds(exc),
            ) from exc
        except APIStatusError as exc:
            raise self._status_error(exc) from exc

        return self._to_completion(message)

    async def aclose(self) -> None:
        await self._client.close()

    def _status_error(self, exc: APIStatusError) -> LLMError:
        """Map an HTTP status to the taxonomy.

        Only the status code is carried over. Provider error bodies can echo
        parts of the request, and these errors are logged, so keeping them out
        preserves the guarantee that prompt content never reaches the logs.
        """
        if exc.status_code == 408:
            return LLMTimeoutError(f"{self._name} returned 408 Request Timeout")
        if exc.status_code >= 500:
            return LLMUnavailableError(f"{self._name} returned {exc.status_code}")
        return LLMRequestError(f"{self._name} rejected the request with {exc.status_code}")

    def _to_completion(self, message: AnthropicMessage) -> Completion:
        stop_reason = _STOP_REASONS.get(message.stop_reason or "")
        if stop_reason is None:
            raise LLMRequestError(
                f"{self._name} returned unsupported stop reason {message.stop_reason!r}"
            )
        return Completion(
            text="".join(block.text for block in message.content if block.type == "text"),
            model=str(message.model),
            stop_reason=stop_reason,
            usage=TokenUsage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
            tool_calls=tuple(
                self._to_tool_call(block) for block in message.content if block.type == "tool_use"
            ),
        )

    def _to_tool_call(self, block: ToolUseBlock) -> ToolCall:
        if not isinstance(block.input, dict):
            # The API documents tool input as a JSON object; anything else would
            # silently reach a handler as arguments it cannot validate.
            raise LLMRequestError(f"{self._name} returned non-object arguments for a tool call")
        return ToolCall(id=block.id, name=block.name, arguments=cast(dict[str, Any], block.input))


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Close every object in ``schema`` with ``additionalProperties: false``.

    The API enforces a JSON Schema — for structured outputs and for strict tool
    arguments — only when its objects are closed and state their ``required``
    fields; Pydantic emits open objects and omits ``required`` entirely when
    every field is optional. Adapting the dialect belongs here rather than in
    the shared port, which should not know one vendor's rules. Field optionality
    is left exactly as the model declared it.
    """
    return cast(dict[str, Any], _close_objects(schema))


def _close_objects(node: Any) -> Any:
    if isinstance(node, dict):
        closed = {key: _close_objects(value) for key, value in node.items()}
        if closed.get("type") == "object" and "additionalProperties" not in closed:
            closed["additionalProperties"] = False
        if closed.get("type") == "object" and "properties" in closed and "required" not in closed:
            closed["required"] = []
        return closed
    if isinstance(node, list):
        return [_close_objects(item) for item in node]
    return node


def _message_param(message: Message) -> MessageParam:
    """Render a turn as Messages API content.

    Plain text turns stay a bare string — the common case, and the form the API
    documents. Turns carrying tool calls or their results become block lists;
    all results for one assistant turn travel in a single user message, which is
    what keeps the model issuing parallel tool calls.
    """
    role: Literal["user", "assistant"] = "user" if message.role is Role.USER else "assistant"
    if not message.tool_calls and not message.tool_results:
        return {"role": role, "content": message.content}

    blocks: list[ContentBlockParam] = []
    for result in message.tool_results:
        tool_result: ToolResultBlockParam = {
            "type": "tool_result",
            "tool_use_id": result.call_id,
            "content": result.content,
            "is_error": result.is_error,
        }
        blocks.append(tool_result)
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    blocks.extend(
        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        for call in message.tool_calls
    )
    return {"role": role, "content": blocks}


def _tool_params(request: CompletionRequest) -> list[ToolParam] | Omit:
    """Declare the offered tools.

    ``strict`` asks the API to constrain decoding to the schema, so arguments
    arrive well-formed instead of costing a correction round trip. The tool
    layer validates them again regardless: the guarantee has to hold for every
    provider, not only the ones that support constrained decoding.
    """
    if not request.tools:
        return omit
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": cast(InputSchema, strict_json_schema(tool.input_schema)),
            "strict": True,
        }
        for tool in request.tools
    ]


def _output_config(request: CompletionRequest) -> OutputConfigParam | Omit:
    if request.output_schema is None:
        return omit
    return {"format": {"type": "json_schema", "schema": strict_json_schema(request.output_schema)}}


def _retry_after_seconds(exc: RateLimitError) -> float | None:
    """Read ``retry-after`` as a delay in seconds, ignoring HTTP-date form."""
    try:
        return float(exc.response.headers.get("retry-after", ""))
    except ValueError:
        return None
