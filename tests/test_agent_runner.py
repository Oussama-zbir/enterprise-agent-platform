"""Tests for the agent run loop.

What matters here is not that a model can be called, but what the loop does with
what comes back: it must feed tool results to the model, keep a runaway run
inside its budget, record every outcome on the task, and lose a race against a
human who cancels.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from pydantic import BaseModel

from enterprise_agent_platform.agent.runner import AgentRunner, NotResumableError
from enterprise_agent_platform.llm.client import LLMClient
from enterprise_agent_platform.llm.errors import LLMError, LLMUnavailableError
from enterprise_agent_platform.llm.models import (
    Completion,
    CompletionRequest,
    Role,
    StopReason,
    TokenUsage,
    ToolCall,
)
from enterprise_agent_platform.llm.provider import FakeLLMProvider
from enterprise_agent_platform.request_context import get_request_id
from enterprise_agent_platform.tasks.models import AgentTask, InvalidTransitionError, TaskStatus
from enterprise_agent_platform.tasks.repository import InMemoryTaskRepository
from enterprise_agent_platform.tasks.service import transition_task
from enterprise_agent_platform.tools.models import RiskLevel, Tool
from enterprise_agent_platform.tools.registry import ToolRegistry


class LookupArgs(BaseModel):
    invoice_id: str


async def _lookup(args: LookupArgs) -> str:
    return f"invoice {args.invoice_id}: 1200.00 EUR, unpaid"


def lookup_tool(*, risk: RiskLevel = RiskLevel.READ, handler: object = None) -> Tool[LookupArgs]:
    return Tool(
        name="lookup_invoice",
        description="Look up an invoice by id.",
        arguments=LookupArgs,
        risk=risk,
        handler=handler or _lookup,  # type: ignore[arg-type]
    )


def tool_use(
    *,
    call_id: str = "call_1",
    name: str = "lookup_invoice",
    arguments: dict[str, object] | None = None,
) -> Completion:
    return Completion(
        text="",
        model="fake-model",
        stop_reason=StopReason.TOOL_USE,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        tool_calls=(
            ToolCall(id=call_id, name=name, arguments=arguments or {"invoice_id": "INV-1"}),
        ),
    )


def answer(
    text: str = "Invoice INV-1 is unpaid.", *, stop_reason: StopReason = StopReason.END_TURN
) -> Completion:
    return Completion(
        text=text,
        model="fake-model",
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=7, output_tokens=3),
    )


class Harness(NamedTuple):
    runner: AgentRunner
    repository: InMemoryTaskRepository
    provider: FakeLLMProvider
    task: AgentTask


async def harness(
    script: list[Completion | LLMError],
    *,
    tools: list[Tool[LookupArgs]] | None = None,
    max_steps: int = 8,
    auto_approve_up_to: RiskLevel = RiskLevel.READ,
    provider: FakeLLMProvider | None = None,
    repository: InMemoryTaskRepository | None = None,
) -> Harness:
    repository = repository or InMemoryTaskRepository()
    task = AgentTask.create(goal="Reconcile supplier payments", requested_by="analyst-1")
    await repository.add(task)
    provider = provider or FakeLLMProvider(script)
    runner = AgentRunner(
        LLMClient(provider, timeout_seconds=5),
        ToolRegistry(tools or []),
        repository,
        max_steps=max_steps,
        auto_approve_up_to=auto_approve_up_to,
    )
    return Harness(runner, repository, provider, task)


async def test_a_direct_answer_completes_the_task() -> None:
    h = await harness([answer("Nothing outstanding.")])

    result = await h.runner.run(h.task.id)

    assert result.status is TaskStatus.COMPLETED
    assert result.output == "Nothing outstanding."
    assert (result.steps, result.tool_calls) == (1, 0)


async def test_the_run_is_recorded_on_the_task_not_only_returned() -> None:
    h = await harness([answer()])

    await h.runner.run(h.task.id)

    stored = await h.repository.get(h.task.id)
    assert [step.to_status for step in stored.history] == [
        TaskStatus.RUNNING,
        TaskStatus.COMPLETED,
    ]
    assert stored.history[-1].reason == "agent run completed"
    assert stored.is_terminal


async def test_the_model_is_offered_the_registered_tools_and_the_goal() -> None:
    h = await harness([answer()], tools=[lookup_tool()])

    await h.runner.run(h.task.id)

    request: CompletionRequest = h.provider.requests[0]
    assert [spec.name for spec in request.tools] == ["lookup_invoice"]
    assert request.messages[0].role is Role.USER
    assert "Reconcile supplier payments" in request.messages[0].content
    # `requested_by` is unverified caller input and is deliberately not prompted.
    assert "analyst-1" not in request.messages[0].content


async def test_tool_output_is_fed_back_to_the_model_as_a_result_turn() -> None:
    h = await harness([tool_use(), answer("INV-1 is unpaid.")], tools=[lookup_tool()])

    result = await h.runner.run(h.task.id)

    assert result.status is TaskStatus.COMPLETED
    assert (result.steps, result.tool_calls) == (2, 1)
    follow_up = h.provider.requests[1].messages
    assert follow_up[1].role is Role.ASSISTANT
    assert [call.name for call in follow_up[1].tool_calls] == ["lookup_invoice"]
    assert follow_up[2].role is Role.USER
    assert "1200.00 EUR" in follow_up[2].tool_results[0].content


async def test_a_bad_tool_call_is_returned_for_correction_rather_than_failing_the_run() -> None:
    h = await harness(
        [
            tool_use(arguments={"invoice": "INV-1"}),  # wrong field name
            tool_use(call_id="call_2"),
            answer(),
        ],
        tools=[lookup_tool()],
    )

    result = await h.runner.run(h.task.id)

    assert result.status is TaskStatus.COMPLETED
    error_turn = h.provider.requests[1].messages[2].tool_results[0]
    assert error_turn.is_error and "invoice_id" in error_turn.content


async def test_token_usage_is_summed_over_the_whole_run() -> None:
    h = await harness([tool_use(), answer()], tools=[lookup_tool()])

    result = await h.runner.run(h.task.id)

    assert result.usage == TokenUsage(input_tokens=17, output_tokens=8)


async def test_a_runaway_tool_loop_is_stopped_by_the_step_budget() -> None:
    h = await harness(
        [tool_use(call_id=f"call_{i}") for i in range(10)], tools=[lookup_tool()], max_steps=3
    )

    result = await h.runner.run(h.task.id)

    assert result.status is TaskStatus.FAILED
    assert result.steps == 3
    assert len(h.provider.requests) == 3
    assert "step budget" in result.detail
    stored = await h.repository.get(h.task.id)
    assert stored.history[-1].reason == result.detail


async def test_a_provider_failure_fails_the_task_with_its_error_type() -> None:
    h = await harness([LLMUnavailableError("bedrock returned 503")])

    result = await h.runner.run(h.task.id)

    assert result.status is TaskStatus.FAILED
    assert "LLMUnavailableError" in result.detail
    # The provider's message may echo the prompt, so it stays out of the record.
    assert "503" not in result.detail


@pytest.mark.parametrize(
    ("stop_reason", "expected_detail"),
    [
        (StopReason.MAX_TOKENS, "truncated"),
        (StopReason.REFUSAL, "refused"),
    ],
)
async def test_an_unusable_answer_fails_the_task(
    stop_reason: StopReason, expected_detail: str
) -> None:
    h = await harness([answer(stop_reason=stop_reason)])

    result = await h.runner.run(h.task.id)

    assert result.status is TaskStatus.FAILED
    assert expected_detail in result.detail


async def test_a_tool_above_the_approval_threshold_pauses_the_run_before_it_acts() -> None:
    executed: list[LookupArgs] = []

    async def pay(args: LookupArgs) -> str:
        executed.append(args)
        return "paid"

    h = await harness(
        [tool_use(), answer()],
        tools=[lookup_tool(risk=RiskLevel.CRITICAL, handler=pay)],
        auto_approve_up_to=RiskLevel.WRITE,
    )

    result = await h.runner.run(h.task.id)

    assert result.status is TaskStatus.AWAITING_APPROVAL
    assert [call.name for call in result.pending_calls] == ["lookup_invoice"]
    assert "approval required for: lookup_invoice" in result.detail
    assert executed == []  # the point of the pause
    assert len(h.provider.requests) == 1  # and the run stops there


async def test_a_task_that_is_not_pending_cannot_be_run() -> None:
    h = await harness([answer()])
    await transition_task(h.repository, h.task.id, TaskStatus.CANCELLED)

    with pytest.raises(InvalidTransitionError):
        await h.runner.run(h.task.id)


async def test_a_cancellation_mid_run_stops_the_loop_before_the_next_model_call() -> None:
    repository = InMemoryTaskRepository()

    async def cancel_while_running(args: LookupArgs) -> str:
        await transition_task(repository, task_id, TaskStatus.CANCELLED, reason="user cancelled")
        return "invoice looked up"

    h = await harness(
        [tool_use(), answer()],
        tools=[lookup_tool(handler=cancel_while_running)],
        repository=repository,
    )
    task_id = h.task.id

    result = await h.runner.run(task_id)

    assert result.status is TaskStatus.CANCELLED
    assert "run stopped" in result.detail
    assert len(h.provider.requests) == 1  # the second step never happened
    stored = await repository.get(task_id)
    assert stored.history[-1].reason == "user cancelled"


async def test_an_outcome_that_arrives_after_a_cancellation_is_discarded() -> None:
    repository = InMemoryTaskRepository()

    class CancellingProvider(FakeLLMProvider):
        """Simulates a human cancelling while the model call is in flight."""

        async def complete(self, request: CompletionRequest) -> Completion:
            await transition_task(repository, task_id, TaskStatus.CANCELLED)
            return await super().complete(request)

    h = await harness(
        [], provider=CancellingProvider([answer("Reconciled.")]), repository=repository
    )
    task_id = h.task.id

    result = await h.runner.run(task_id)

    assert result.status is TaskStatus.CANCELLED
    assert result.output == ""  # the completed answer is not reported as the task's
    stored = await repository.get(task_id)
    assert stored.status is TaskStatus.CANCELLED


class Paused(NamedTuple):
    harness: Harness
    executed: list[str]


async def paused_run(*, follow_up: list[Completion] | None = None, max_steps: int = 8) -> Paused:
    """A run stopped at an approval gate, with the tool it wants to use."""
    executed: list[str] = []

    async def pay(args: LookupArgs) -> str:
        executed.append(args.invoice_id)
        return f"paid {args.invoice_id}"

    h = await harness(
        [tool_use(), *(follow_up or [])],
        tools=[lookup_tool(risk=RiskLevel.CRITICAL, handler=pay)],
        auto_approve_up_to=RiskLevel.WRITE,
        max_steps=max_steps,
    )
    result = await h.runner.run(h.task.id)
    assert result.status is TaskStatus.AWAITING_APPROVAL
    return Paused(h, executed)


async def test_a_pause_checkpoints_the_conversation_on_the_task() -> None:
    p = await paused_run()

    stored = await p.harness.repository.get(p.harness.task.id)

    assert stored.checkpoint is not None
    assert [m.role for m in stored.checkpoint.messages] == [Role.USER, Role.ASSISTANT]
    assert [c.name for c in stored.checkpoint.pending_calls] == ["lookup_invoice"]
    # The step already spent is carried, not forgotten.
    assert stored.checkpoint.steps == 1


async def test_approving_continues_the_paused_run_instead_of_restarting_it() -> None:
    p = await paused_run(follow_up=[answer("INV-1 settled.")])
    h = p.harness

    result = await h.runner.resume(h.task.id, approved_by="manager-2")

    assert result.status is TaskStatus.COMPLETED
    assert result.output == "INV-1 settled."
    assert p.executed == ["INV-1"]  # the held call ran, once
    # The model is asked to continue the same conversation, not to start over.
    follow_up = h.provider.requests[1].messages
    assert [m.role for m in follow_up] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert "Reconcile supplier payments" in follow_up[0].content
    assert follow_up[2].tool_results[0].content == "paid INV-1"
    assert (result.steps, result.tool_calls) == (2, 1)


async def test_the_checkpoint_is_dropped_once_the_task_moves_on() -> None:
    p = await paused_run(follow_up=[answer()])
    h = p.harness

    await h.runner.resume(h.task.id, approved_by="manager-2")

    assert (await h.repository.get(h.task.id)).checkpoint is None


async def test_the_approver_is_recorded_in_the_task_history() -> None:
    p = await paused_run(follow_up=[answer()])
    h = p.harness

    await h.runner.resume(h.task.id, approved_by="manager-2", note="supplier verified")

    stored = await h.repository.get(h.task.id)
    resumption = stored.history[-2]
    assert resumption.to_status is TaskStatus.RUNNING
    assert resumption.reason == "approved by manager-2: supplier verified"


async def test_approval_does_not_reset_the_step_budget() -> None:
    p = await paused_run(max_steps=1)
    h = p.harness

    result = await h.runner.resume(h.task.id, approved_by="manager-2")

    # Otherwise a run could be paused and approved its way around max_steps.
    assert result.status is TaskStatus.FAILED
    assert "step budget" in result.detail
    assert p.executed == ["INV-1"]  # the approved call still ran
    assert len(h.provider.requests) == 1


async def test_rejecting_cancels_the_task_and_drops_the_denied_calls() -> None:
    p = await paused_run(follow_up=[answer()])
    h = p.harness

    task = await h.runner.reject(h.task.id, rejected_by="manager-2", reason="supplier unverified")

    assert task.status is TaskStatus.CANCELLED
    assert task.checkpoint is None
    assert p.executed == []
    assert task.history[-1].reason == "rejected by manager-2: supplier unverified"


async def test_a_rejected_run_cannot_be_approved_afterwards() -> None:
    p = await paused_run(follow_up=[answer()])
    h = p.harness
    await h.runner.reject(h.task.id, rejected_by="manager-2")

    with pytest.raises(InvalidTransitionError):
        await h.runner.resume(h.task.id, approved_by="manager-3")


async def test_only_the_first_approval_takes_effect() -> None:
    p = await paused_run(follow_up=[answer()])
    h = p.harness
    await h.runner.resume(h.task.id, approved_by="manager-2")

    with pytest.raises(InvalidTransitionError):
        await h.runner.resume(h.task.id, approved_by="manager-3")

    assert p.executed == ["INV-1"]  # the critical call is not replayed


async def test_a_task_that_never_paused_cannot_be_approved() -> None:
    h = await harness([answer()])

    with pytest.raises(InvalidTransitionError):
        await h.runner.resume(h.task.id, approved_by="manager-2")

    assert h.provider.requests == []


async def test_a_paused_task_with_no_run_state_is_not_resumed_silently() -> None:
    h = await harness([answer()])
    await transition_task(h.repository, h.task.id, TaskStatus.RUNNING)
    await transition_task(h.repository, h.task.id, TaskStatus.AWAITING_APPROVAL)

    # Resuming from nothing would quietly restart the run instead of continuing it.
    with pytest.raises(NotResumableError):
        await h.runner.resume(h.task.id, approved_by="manager-2")


async def test_the_approval_view_marks_only_the_calls_that_tripped_the_gate() -> None:
    mixed_turn = Completion(
        text="",
        model="fake-model",
        stop_reason=StopReason.TOOL_USE,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        tool_calls=(
            ToolCall(id="call_1", name="list_invoices", arguments={"invoice_id": "INV-1"}),
            ToolCall(id="call_2", name="lookup_invoice", arguments={"invoice_id": "INV-2"}),
        ),
    )
    listing = Tool(
        name="list_invoices",
        description="List invoices.",
        arguments=LookupArgs,
        risk=RiskLevel.READ,
        handler=_lookup,
    )
    h = await harness(
        [mixed_turn],
        tools=[listing, lookup_tool(risk=RiskLevel.CRITICAL)],
        auto_approve_up_to=RiskLevel.WRITE,
    )
    await h.runner.run(h.task.id)

    pending = h.runner.pending_approval(await h.repository.get(h.task.id))

    # The whole turn is shown: approving one action while its siblings run
    # unexamined is not an informed decision.
    assert [(c.name, c.needs_approval) for c in pending] == [
        ("list_invoices", False),
        ("lookup_invoice", True),
    ]
    assert [c.risk for c in pending] == [RiskLevel.READ, RiskLevel.CRITICAL]
    assert pending[1].arguments == {"invoice_id": "INV-2"}


async def test_the_approval_view_is_empty_for_a_task_that_is_not_paused() -> None:
    h = await harness([answer()])

    assert h.runner.pending_approval(h.task) == ()


async def test_the_originating_request_id_follows_the_run_into_tool_execution() -> None:
    seen: list[str | None] = []

    async def record_request_id(args: LookupArgs) -> str:
        seen.append(get_request_id())
        return "ok"

    h = await harness([tool_use(), answer()], tools=[lookup_tool(handler=record_request_id)])

    await h.runner.run(h.task.id, request_id="req-42")

    assert seen == ["req-42"]
    assert get_request_id() is None  # the binding does not leak past the run
