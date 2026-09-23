"""Tool registry: the one place a model's tool call turns into real work.

Two rules shape this module.

*Model mistakes are data, not exceptions.* A hallucinated tool name, malformed
arguments, a handler that fails or hangs — each comes back as a ``ToolResult``
with ``is_error`` set, which the orchestrator hands to the model so it can
correct itself. Raising would end a task that is usually still recoverable.
Programmer mistakes (registering the same tool twice) still raise.

*What goes back to the model is bounded and scrubbed.* An unexpected handler
exception is logged and replaced with a generic message — its text can carry
connection strings or internal identifiers, and everything returned here
re-enters the prompt, where it is also a prompt-injection surface. Only
``ToolExecutionError``, which a handler raises deliberately to describe a
failure in its own domain, travels back verbatim. Results are truncated so one
chatty tool cannot consume the context window.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import ValidationError

from enterprise_agent_platform.llm.models import ToolCall, ToolResult, ToolSpec
from enterprise_agent_platform.tools.models import Tool, ToolExecutionError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_RESULT_CHARS = 8_000
_TRUNCATION_NOTE = "\n[truncated]"


class ToolRegistry:
    """The tools a deployment offers, and the only way to run them."""

    def __init__(
        self,
        tools: Iterable[Tool[Any]] = (),
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_result_chars <= len(_TRUNCATION_NOTE):
            raise ValueError("max_result_chars must leave room for the truncation note")
        self._tools: dict[str, Tool[Any]] = {}
        self._timeout_seconds = timeout_seconds
        self._max_result_chars = max_result_chars
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool[Any]) -> None:
        """Add a tool.

        Raises:
            ValueError: if the name is already taken. Two tools answering to one
                name is a wiring bug, and the model would have no way to say
                which it meant.
        """
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def specs(self) -> tuple[ToolSpec, ...]:
        """Declarations for every tool, to offer on a completion request."""
        return tuple(tool.spec() for tool in self._tools.values())

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call and describe the outcome to the model."""
        tool = self._tools.get(call.name)
        if tool is None:
            return self._failure(
                call,
                tool=None,
                outcome="unknown_tool",
                message=(
                    f"No tool named '{call.name}'. Available tools: "
                    f"{', '.join(self.names) or 'none'}."
                ),
            )

        try:
            arguments = tool.arguments.model_validate(call.arguments)
        except ValidationError as exc:
            # The model wrote these arguments, so telling it exactly which field
            # is wrong is what lets it retry successfully. Input values are left
            # out: they may have come from earlier tool output.
            return self._failure(
                call,
                tool=tool,
                outcome="invalid_arguments",
                message=(
                    f"Invalid arguments for '{tool.name}': "
                    f"{exc.errors(include_input=False, include_url=False)}"
                ),
            )

        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                output = await tool.handler(arguments)
        except ToolExecutionError as exc:
            # The handler is reporting an outcome, not a defect: "no invoice
            # with that id", "the remote server rejected this call". Its text is
            # part of the tool's contract, so it goes back to the model — the
            # same treatment invalid arguments get, and the reason a run can
            # recover instead of failing the task.
            return self._failure(
                call,
                tool=tool,
                outcome="tool_error",
                message=str(exc) or f"Tool '{tool.name}' reported a failure.",
                duration_ms=_elapsed_ms(started),
            )
        except TimeoutError:
            return self._failure(
                call,
                tool=tool,
                outcome="timeout",
                message=f"Tool '{tool.name}' timed out after {self._timeout_seconds}s.",
                duration_ms=_elapsed_ms(started),
            )
        except Exception:
            logger.exception(
                "tool.call.failed",
                extra={
                    "tool": tool.name,
                    "risk": tool.risk.value,
                    "outcome": "error",
                    "duration_ms": _elapsed_ms(started),
                },
            )
            return ToolResult(
                call_id=call.id,
                content=f"Tool '{tool.name}' failed to run.",
                is_error=True,
            )

        content = self._truncate(output)
        logger.info(
            "tool.call.completed",
            extra={
                "tool": tool.name,
                "risk": tool.risk.value,
                "outcome": "ok",
                "result_chars": len(content),
                "duration_ms": _elapsed_ms(started),
            },
        )
        return ToolResult(call_id=call.id, content=content)

    async def execute_all(self, calls: Sequence[ToolCall]) -> tuple[ToolResult, ...]:
        """Run every call from one assistant turn, concurrently.

        Models emit independent tool calls in a single turn precisely so they can
        run in parallel; running them in sequence would add up their latencies
        for no reason. Order is preserved because results are matched by call id.
        """
        return tuple(await asyncio.gather(*(self.execute(call) for call in calls)))

    def _failure(
        self,
        call: ToolCall,
        *,
        tool: Tool[Any] | None,
        outcome: str,
        message: str,
        duration_ms: float | None = None,
    ) -> ToolResult:
        logger.warning(
            "tool.call.rejected",
            extra={
                "tool": call.name,
                "risk": tool.risk.value if tool is not None else None,
                "outcome": outcome,
                "duration_ms": duration_ms,
            },
        )
        return ToolResult(call_id=call.id, content=self._truncate(message), is_error=True)

    def _truncate(self, content: str) -> str:
        if len(content) <= self._max_result_chars:
            return content
        keep = self._max_result_chars - len(_TRUNCATION_NOTE)
        return content[:keep] + _TRUNCATION_NOTE


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
