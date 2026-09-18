# Enterprise Agent Platform

A production-oriented platform for building and operating enterprise AI agents.

> **Status: AI capability.** This repository is being built incrementally. It
> has an engineering foundation (config, structured logging with request
> correlation IDs, strict typing, tests, Docker, CI), a task domain (an explicit
> agent-task lifecycle, a storage port with optimistic concurrency, and a
> `/tasks` API), an LLM layer (a provider port with a real Anthropic / Bedrock
> adapter behind it, including tool calling), a tool layer (typed tools with risk
> metadata, validated arguments, and bounded execution), and the agent runner
> that drives a task through model and tool calls under a step budget. MCP
> integration, human-in-the-loop approval, and evaluation are planned milestones
> (see [Roadmap](#roadmap)).

## Problem statement

Most agentic AI work stops at a prototype: a prompt, a model call, a response.
That gap — between a notebook demo and a system an organization can actually
run — is where the hard engineering lives. This project exists to close it: to
demonstrate how an agent platform is built with the same rigor as any other
production service (configuration, observability, testing, reliability,
security, and deployment), not just clever prompting.

## Architecture (intended)

The foundation is a FastAPI service. Subsequent milestones layer domain and
agent capabilities on top without disturbing the operational base.

```
                +--------------------------------------------------+
                |                 FastAPI service                  |
                |                                                  |
  HTTP  ----->  |  /tasks router                                   |
                |      |                                           |
                |      v                                           |
                |  task service  -->  TaskRepository (port)        |
                |      |                  +--> in-memory adapter   |
                |      |                  +--> database (later)    |
                |      v                                           |
                |  AgentTask state machine                         |
                |      ^                                           |
                |      +-- AgentRunner (model <-> tool loop)       |
                |            +--> LLMClient --> provider port      |
                |            +--> ToolRegistry (typed tools)       |
                |            +--> MCP (later)                      |
                |            +--> human-in-the-loop (later)        |
                |                                                  |
                |  cross-cutting: config · JSON logging ·          |
                |  X-Request-ID correlation · tracing (later)      |
                +--------------------------------------------------+
```

Current modules:

| Module                                 | Responsibility                                  |
| -------------------------------------- | ----------------------------------------------- |
| `enterprise_agent_platform.config`     | Environment-based settings (Pydantic Settings)  |
| `enterprise_agent_platform.logging`    | Structured JSON logging to stdout               |
| `...request_context`                   | `X-Request-ID` middleware, request access logs  |
| `enterprise_agent_platform.main`       | App factory, lifespan, `/health`, router wiring |
| `...tasks.models`                      | `AgentTask` model and lifecycle state machine   |
| `...tasks.repository`                  | `TaskRepository` port + in-memory adapter       |
| `...tasks.service`                     | Load → transition → persist use case            |
| `...tasks.router`                      | `/tasks` HTTP routes and API schemas            |
| `...llm.models`                        | Provider-neutral request/completion types       |
| `...llm.errors`                        | LLM failure taxonomy with `retryable` flag      |
| `...llm.provider`                      | `LLMProvider` port + scripted fake provider     |
| `...llm.client`                        | Timeouts, structured output, LLM call logs      |
| `...llm.anthropic_provider`            | Anthropic / Bedrock adapter for the port        |
| `...llm.factory`                       | Backend selection from settings                 |
| `...tools.models`                      | Typed tool definitions and risk levels          |
| `...tools.registry`                    | Tool lookup, validation, bounded execution      |
| `...agent.runner`                      | The run loop: model calls, tools, budgets        |

### Task lifecycle

```
pending ──> running ──> completed
   │           │  ├───> failed
   │           │  └───> awaiting_approval ──> running   (approved)
   │           │               │
   └───────────┴───────────────┴──> cancelled           (cancel / rejected)
```

`completed`, `failed`, and `cancelled` are terminal. The allowed transitions
live in one table (`ALLOWED_TRANSITIONS`), so the API, the future orchestrator,
and the approval workflow cannot disagree about what is legal.

## Design decisions

- **Application factory (`create_app`)** rather than a single global app, so
  tests and future deployments can build isolated instances with overridden
  configuration.
- **Environment-based config with an `EAP_` prefix** via Pydantic Settings —
  twelve-factor style, validated at load time, no config scattered in code.
- **Structured JSON logging, dependency-free** — parseable in containers and
  cloud log aggregators today; a full OpenTelemetry tracing stack is deferred
  to the observability milestone rather than added prematurely.
- **Request correlation IDs via a `ContextVar`.** A pure ASGI middleware assigns
  each request an ID, returns it as `X-Request-ID`, and a logging filter stamps
  it on every record emitted while the request runs — including logs from deep
  in the service layer, without threading the ID through function signatures.
  Pure ASGI (not `BaseHTTPMiddleware`) keeps the context in the endpoint's task
  and avoids buffering responses. A caller-supplied ID is reused only if it is a
  short token of `[A-Za-z0-9._:-]`; anything else is replaced, so the header
  cannot be used for log injection. Unhandled exceptions are logged with the ID
  and returned as a JSON 500 that still carries the header.
- **`src/` layout** to keep the importable package separate from tooling and
  tests, and to catch packaging mistakes early.
- **Strict typing and linting from day one** so quality is enforced by CI
  before the codebase grows.
- **Explicit state machine over free-form status updates.** Agent tasks will be
  driven by several actors (orchestrator, human approvers, API clients). A
  single transition table rejects impossible states such as completing a task
  that never ran, and each change is appended to an audit history.
- **Immutable tasks with versioning.** Transitions return a new `AgentTask`
  with `version + 1` rather than mutating in place, which keeps history
  append-only and makes stale writes detectable.
- **Optimistic concurrency in the repository.** `update()` takes the version the
  caller read and fails with `ConcurrentUpdateError` (HTTP 409) if the task has
  moved on. This avoids lost updates when, say, an approval and a cancellation
  race, without holding locks across slow LLM or human steps. A database
  adapter maps this to `UPDATE ... WHERE id = :id AND version = :expected`.
- **Repository as a `Protocol` port, in-memory adapter first.** Persistence
  technology is deferred until orchestration shows the real access patterns;
  the adapter is injected through `create_app(task_repository=...)`.
- **API schemas separate from the domain model**, so the public contract and
  internal representation can evolve independently. Clients can create, read,
  list, and cancel tasks; running/approval transitions are reserved for the
  orchestrator and approval workflow rather than exposed as raw status writes.
- **LLM access behind a provider port.** Agent code depends on `LLMClient` and
  provider-neutral types, never on a vendor SDK, so the Anthropic API, Bedrock,
  and a deterministic fake are interchangeable. Adapters translate vendor
  exceptions into one taxonomy (`LLMTimeoutError`, `LLMRateLimitError`,
  `LLMUnavailableError`, `LLMRequestError`, `LLMRefusalError`,
  `StructuredOutputError`) with a `retryable` flag; whether to retry stays with
  the caller, which knows the task and its budget.
- **Structured outputs are validated, not trusted.** `complete_structured`
  sends the Pydantic model's JSON Schema so providers with native constrained
  decoding can enforce it, then validates the response regardless. Truncation
  (`max_tokens`) and refusals are reported as such instead of surfacing as
  confusing JSON parse errors.
- **The vendor SDK stops at the adapter.** `AnthropicProvider` is the only
  module that imports `anthropic`. It maps HTTP status codes to the taxonomy
  (408 → timeout, 429 → rate limit with the `retry-after` delay, 5xx →
  unavailable, other 4xx → request error), translates stop reasons, and rejects
  ones it cannot honour yet (`tool_use`) rather than silently mislabelling them
  as a normal end of turn. Error messages carry the status code but never the
  provider's error body, which can echo the prompt.
- **Provider SDK retries are disabled (`max_retries=0`).** The SDK would happily
  retry a 429 or 5xx inside a single `complete()` call, which both takes the
  retry decision away from the caller — the only party that knows the task, its
  deadline, and its budget — and inflates the latency recorded for what the logs
  present as one model call.
- **Schema dialects are adapted, not standardised.** The Messages API only
  enforces a JSON Schema when its objects are closed; Pydantic emits open ones.
  The adapter closes them on the way out, so the shared port stays free of one
  vendor's rules.
- **Misconfiguration fails at startup.** The default backend (`fake`) is offline
  only and is rejected when `EAP_ENVIRONMENT=production`, so a deployment that
  forgot to configure a model dies immediately instead of on the first customer
  request.
- **One log record per model call, without content.** `llm.call.completed` /
  `llm.call.failed` carry provider, model, operation, token usage, stop reason,
  latency, and the request ID, which is the raw material for cost and latency
  accounting. Prompt and response text are not logged, only their sizes,
  because they may contain customer data.
- **Tools are typed, and their arguments are validated twice.** A tool declares
  its arguments as a Pydantic model; that model generates the JSON Schema the
  model sees (sent with `strict: true`, so the provider constrains decoding) and
  validates what comes back. Handlers therefore receive a typed object, never a
  raw dictionary produced by a language model.
- **A model's mistakes are data; a programmer's are exceptions.** An unknown
  tool name, invalid arguments, a handler that fails or hangs — each returns a
  `ToolResult` with `is_error`, which the orchestrator feeds back so the model
  can correct itself. Aborting the task would throw away a run that is usually
  still recoverable. Registering two tools under one name still raises.
- **What returns to the model is bounded and scrubbed.** Handler exception
  messages are logged but never returned — they can carry connection strings or
  internal identifiers, and everything returned here re-enters the prompt, which
  is also a prompt-injection surface. Results are truncated so one chatty tool
  cannot consume the context window. Validation errors *are* returned: the model
  wrote those arguments, and naming the bad field is what lets it retry.
- **Risk is metadata on the tool; approval is policy.** Each tool declares
  `read`, `write`, or `critical`, and `requires_approval` compares that against
  a deployment-wide threshold. A stricter deployment lowers one threshold
  instead of editing every tool, and the approval workflow will read the same
  function the orchestrator does.
- **Tools are offered per request, not held on the client**, so an orchestrator
  can narrow the set per task: a model cannot misuse a tool it was never given.
- **Every run is bounded, and exhausting the budget is a real outcome.** A model
  that keeps calling tools is a common failure mode that costs money and latency
  until something stops it. `EAP_AGENT_MAX_STEPS` caps the model calls in one
  run; hitting the cap fails the task with that reason in its history rather
  than looping or silently returning a half-finished answer.
- **Starting a run is a state transition, so a task has at most one agent.**
  `pending -> running` goes through the repository's version check, so two
  callers racing to start the same task produce one run and one 409 — not two
  agents doing the same work with the same tools.
- **Cancellation is cooperative, and the human wins.** The task is re-read
  between steps, so a cancelled task stops before the next model call. If the
  cancellation lands while a step is in flight, the run's final transition is
  rejected by the lifecycle and its outcome is discarded rather than overwriting
  the human's decision.
- **An approval pause stops the whole turn, not the risky call.** When any tool
  in a turn exceeds `EAP_AGENT_AUTO_APPROVE_UP_TO`, nothing from that turn runs
  and the task moves to `awaiting_approval`. Executing the rest would mean
  reconstructing a partially applied turn on approval, and the remaining calls
  may depend on the paused one.
- **Agent failures are outcomes, not exceptions.** A provider error, a refusal,
  a truncated answer, an exhausted budget — each ends with the task in `failed`
  and a reason in its history; only genuine lifecycle errors (unknown task, task
  not startable) propagate as 404/409. The reason returned to the caller is the
  same string recorded in the audit trail, so the API and the history cannot
  disagree. Provider error text is not copied into it, since it can echo the
  prompt.
- **The prompt carries the goal, not the requester.** `requested_by` is
  unverified caller input, so it stays out of the prompt; the system prompt
  tells the model that tool results are data, never instructions.
- **`POST /tasks/{id}/run` is synchronous for now.** The caller waits, which is
  honest for a single-process deployment. Because progress is recorded on the
  task rather than in the response, moving execution to a queue and returning
  202 changes the entrypoint, not the domain logic.
- **One log record per run** (`agent.run.completed`) with steps, tool calls,
  token totals, latency, and the originating request ID. The request ID is
  passed into `AgentRunner.run` explicitly rather than read from a context
  variable, so a run handed to a background worker still correlates with the
  request that created the task.

## Technology stack

- Python 3.12+
- FastAPI + Uvicorn
- Anthropic SDK (Messages API, first-party or Bedrock)
- Pydantic v2 / pydantic-settings
- pytest + pytest-asyncio
- Ruff (lint + format), mypy (strict)
- Docker, GitHub Actions

## Setup

Requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

The default model backend is `fake`: offline, and every call raises. Point it at
a real model with environment variables (never commit a key):

```bash
# Anthropic API — omit the key to use the SDK's own credential resolution
EAP_LLM_PROVIDER=anthropic EAP_ANTHROPIC_API_KEY=sk-ant-... EAP_LLM_MODEL=claude-opus-5

# Bedrock — credentials come from the standard AWS chain
EAP_LLM_PROVIDER=bedrock EAP_AWS_REGION=eu-west-1 EAP_LLM_MODEL=anthropic.claude-opus-5
```

## Local development

Run the service:

```bash
uvicorn enterprise_agent_platform.main:app --reload
```

Then:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","service":"enterprise-agent-platform","environment":"development","version":"0.1.0"}
```

Create, inspect, and cancel a task:

```bash
curl -s -X POST http://127.0.0.1:8000/tasks \
  -H 'Content-Type: application/json' \
  -d '{"goal": "Reconcile supplier payments for March", "requested_by": "analyst-1"}'
# {"id":"<uuid>","status":"pending","is_terminal":false,"version":1,...,"history":[]}

curl -s 'http://127.0.0.1:8000/tasks?status=pending&limit=20'
curl -s http://127.0.0.1:8000/tasks/<uuid>

curl -s -X POST http://127.0.0.1:8000/tasks/<uuid>/cancel \
  -H 'Content-Type: application/json' -d '{"reason": "duplicate request"}'
# 200 with status "cancelled"; cancelling again returns 409 Conflict
```

Run a task with the agent loop:

```bash
curl -s -X POST http://127.0.0.1:8000/tasks/<uuid>/run
# {"task":{"status":"completed","version":3,"history":[...]},
#  "output":"...", "detail":"agent run completed",
#  "steps":2, "tool_calls":1, "input_tokens":812, "output_tokens":96,
#  "pending_tool_calls":[]}
```

The run drives the task: `pending -> running -> completed | failed`, or
`awaiting_approval` when the model asks for a tool riskier than
`EAP_AGENT_AUTO_APPROVE_UP_TO` (then `pending_tool_calls` names what it wants).
Running the same task twice returns 409. The platform ships with no tools of its
own; a deployment registers its own through `create_app(tool_registry=...)`.

| Method & path              | Result                                                  |
| -------------------------- | ------------------------------------------------------- |
| `POST /tasks`              | 201 pending task; 422 on invalid or unknown fields      |
| `GET /tasks`               | Newest first; optional `status` filter, `limit` 1–200   |
| `GET /tasks/{id}`          | 200, or 404 if unknown                                  |
| `POST /tasks/{id}/run`     | 200 with the run outcome; 404 unknown; 409 not pending  |
| `POST /tasks/{id}/cancel`  | 200; 404 unknown; 409 terminal task or concurrent write |

Every response carries an `X-Request-ID` header (the caller's, if well-formed,
otherwise a generated UUID), and every log line written during that request
includes it:

```bash
curl -s -i http://127.0.0.1:8000/tasks -H 'X-Request-ID: req-42' | grep -i x-request-id
# x-request-id: req-42
# log: {"message": "request.completed", "method": "GET", "path": "/tasks",
#       "status_code": 200, "duration_ms": 0.4, "request_id": "req-42", ...}
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

## Quality checks

```bash
ruff check .          # lint
ruff format --check . # formatting
mypy                  # static type checking (strict)
pytest -v             # tests
```

All four run in CI on Python 3.12 and 3.13.

## Docker

```bash
docker build -t enterprise-agent-platform .
docker run --rm -p 8000:8000 enterprise-agent-platform
```

## Roadmap

1. **Foundation** — service skeleton, config, logging, tests, CI ✅
2. **Core domain** — task lifecycle, repository port, task API, request
   correlation IDs ✅
3. **AI capability** — LLM provider abstraction, agent orchestration, tool
   calling, MCP integration *(in progress: provider port, client, the Anthropic
   / Bedrock adapter, the tool registry, and the agent run loop done; MCP next)*
4. **Human-in-the-loop** — approval workflows, structured state
5. **Evaluation** — agent evaluation harness and metrics
6. **Observability** — tracing, latency/cost accounting (OpenTelemetry)
7. **Reliability & security** — retries, rate limits, prompt-injection defense
8. **Deployment** — cloud-oriented deployment and infrastructure

## Limitations

Agent runs execute inside the API process while the caller waits, so a long run
holds an HTTP connection and a restart loses it; moving execution behind a queue
needs durable tasks first. A run that pauses for approval stops there: the
conversation is not persisted, so resuming an approved task is part of the
human-in-the-loop milestone, together with the approve/reject endpoints. No
tools ship with the platform — a deployment registers its own — so an
out-of-the-box run is a single model call. The adapter is tested against a mock
HTTP transport rather than the live API, and does not cover streaming or prompt
caching. Tasks are stored in process memory, so they are lost on restart and are
not shared across multiple workers or replicas. There is no authentication yet,
so `requested_by` is caller-supplied and not verified.

## License

MIT
