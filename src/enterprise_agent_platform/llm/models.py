"""Provider-neutral LLM request and response types.

The orchestrator talks to models through these types rather than a vendor SDK's,
so providers (Anthropic API, Bedrock, a deterministic fake in tests) can be
swapped without touching agent logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    REFUSAL = "refusal"
    TOOL_USE = "tool_use"


TOOL_NAME_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"
"""Tool names are identifiers the model emits back verbatim, so they are kept to
a narrow character set that every provider accepts."""


class ToolSpec(BaseModel):
    """A tool as the model sees it: a name, what it does, and its argument schema."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    """A model's request to run one tool.

    ``arguments`` is whatever the model produced. It is *unvalidated* here — the
    tool layer parses it against the tool's own schema before anything runs.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of one tool call, as it is handed back to the model.

    ``is_error`` lets a failure be reported as data rather than by aborting the
    run, so the model can correct itself (fix an argument, pick another tool)
    instead of the whole task failing on a recoverable mistake.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str = Field(min_length=1)
    content: str
    is_error: bool = False


class Message(BaseModel):
    """One conversation turn.

    A turn carries text, tool calls (assistant), or tool results (user). The
    role rules are enforced here rather than in each adapter, because a
    misplaced block is rejected by providers with an opaque 400.
    """

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()

    @classmethod
    def from_completion(cls, completion: Completion) -> Message:
        """Build the assistant turn to append before answering its tool calls."""
        return cls(role=Role.ASSISTANT, content=completion.text, tool_calls=completion.tool_calls)

    @classmethod
    def with_tool_results(cls, results: Iterable[ToolResult]) -> Message:
        """Build the user turn that returns tool results to the model.

        All results for one assistant turn belong in a single message; splitting
        them across turns teaches the model to stop calling tools in parallel.
        """
        return cls(role=Role.USER, tool_results=tuple(results))

    @model_validator(mode="after")
    def _check_blocks(self) -> Message:
        if self.tool_calls and self.role is not Role.ASSISTANT:
            raise ValueError("only an assistant message can carry tool calls")
        if self.tool_results and self.role is not Role.USER:
            raise ValueError("only a user message can carry tool results")
        if not self.content and not self.tool_calls and not self.tool_results:
            raise ValueError("message must carry content, tool calls, or tool results")
        return self


class CompletionRequest(BaseModel):
    """A single model call.

    ``output_schema`` is a JSON Schema the provider may enforce natively
    (constrained decoding). The client validates the response against the
    Pydantic model either way, so providers without native support are safe.

    ``tools`` are the tools the model may call on this turn. The set is passed
    per request rather than held on the client, so an orchestrator can narrow it
    per task — a model cannot misuse a tool it was never offered.
    """

    model_config = ConfigDict(frozen=True)

    messages: tuple[Message, ...] = Field(min_length=1)
    system: str | None = None
    max_tokens: int = Field(default=1024, ge=1)
    output_schema: dict[str, Any] | None = None
    tools: tuple[ToolSpec, ...] = ()

    @field_validator("messages")
    @classmethod
    def _starts_with_user_turn(cls, messages: tuple[Message, ...]) -> tuple[Message, ...]:
        if messages[0].role is not Role.USER:
            raise ValueError("conversation must start with a user message")
        return messages

    @field_validator("tools")
    @classmethod
    def _tool_names_are_unique(cls, tools: tuple[ToolSpec, ...]) -> tuple[ToolSpec, ...]:
        names = [tool.name for tool in tools]
        if len(set(names)) != len(names):
            raise ValueError("tool names offered to the model must be unique")
        return tools


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class Completion(BaseModel):
    """What the model produced.

    ``tool_calls`` are only safe to run when ``stop_reason`` is
    ``TOOL_USE``: a turn cut short at ``max_tokens`` can carry a half-written
    call whose arguments were never finished.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    stop_reason: StopReason
    usage: TokenUsage
    tool_calls: tuple[ToolCall, ...] = ()

    @model_validator(mode="after")
    def _check_tool_calls(self) -> Completion:
        if self.stop_reason is StopReason.TOOL_USE and not self.tool_calls:
            raise ValueError("a tool_use completion must carry at least one tool call")
        call_ids = [call.id for call in self.tool_calls]
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("tool call ids must be unique within a completion")
        return self
