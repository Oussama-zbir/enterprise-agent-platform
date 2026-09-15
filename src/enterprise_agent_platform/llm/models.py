"""Provider-neutral LLM request and response types.

The orchestrator talks to models through these types rather than a vendor SDK's,
so providers (Anthropic API, Bedrock, a deterministic fake in tests) can be
swapped without touching agent logic.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    REFUSAL = "refusal"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = Field(min_length=1)


class CompletionRequest(BaseModel):
    """A single model call.

    ``output_schema`` is a JSON Schema the provider may enforce natively
    (constrained decoding). The client validates the response against the
    Pydantic model either way, so providers without native support are safe.
    """

    model_config = ConfigDict(frozen=True)

    messages: tuple[Message, ...] = Field(min_length=1)
    system: str | None = None
    max_tokens: int = Field(default=1024, ge=1)
    output_schema: dict[str, Any] | None = None

    @field_validator("messages")
    @classmethod
    def _starts_with_user_turn(cls, messages: tuple[Message, ...]) -> tuple[Message, ...]:
        if messages[0].role is not Role.USER:
            raise ValueError("conversation must start with a user message")
        return messages


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class Completion(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    stop_reason: StopReason
    usage: TokenUsage
