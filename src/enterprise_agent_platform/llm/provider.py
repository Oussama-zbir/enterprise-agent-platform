"""LLM provider port and a deterministic fake adapter."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from typing import Protocol

from enterprise_agent_platform.llm.errors import LLMError
from enterprise_agent_platform.llm.models import (
    Completion,
    CompletionRequest,
    StopReason,
    TokenUsage,
)


class LLMProvider(Protocol):
    """A model backend.

    Implementations must raise ``LLMError`` subclasses for provider failures and
    must not retry internally unless configured to; retry policy belongs to the
    caller, which can see the task and budget context.
    """

    @property
    def name(self) -> str: ...

    async def complete(self, request: CompletionRequest) -> Completion: ...

    async def aclose(self) -> None:
        """Release held resources, such as an HTTP connection pool."""
        ...


class FakeLLMProvider:
    """Scripted provider for tests and offline development.

    Each call consumes the next scripted item: a string becomes an ``end_turn``
    completion, a ``Completion`` is returned as-is, and an ``LLMError`` is
    raised. Received requests are recorded for assertions.
    """

    name = "fake"

    def __init__(
        self,
        script: Iterable[str | Completion | LLMError],
        *,
        model: str = "fake-model",
        delay_seconds: float = 0.0,
    ) -> None:
        self._script = deque(script)
        self._model = model
        self._delay_seconds = delay_seconds
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> Completion:
        self.requests.append(request)
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        if not self._script:
            raise RuntimeError("FakeLLMProvider script exhausted")

        item = self._script.popleft()
        if isinstance(item, LLMError):
            raise item
        if isinstance(item, Completion):
            return item
        return Completion(
            text=item,
            model=self._model,
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(
                input_tokens=sum(len(m.content.split()) for m in request.messages),
                output_tokens=len(item.split()),
            ),
        )

    async def aclose(self) -> None:
        """No resources to release."""
