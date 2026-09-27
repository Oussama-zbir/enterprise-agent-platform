# Architecture and design decisions

The reasoning behind the code: what this system is, how it is put together, and
why each part works the way it does. The [README](../README.md) is the short
version; this is the long one, and it is where the engineering argument lives.

Every decision below is implemented. Where something is deliberately *not*
built, it says so and says why.

## Contents

- [Problem statement](#problem-statement)
- [Architecture](#architecture)
- [Task lifecycle](#task-lifecycle)
- [Design decisions](#design-decisions)
  - [Foundation, domain, agent, persistence](#foundation-domain-agent-persistence)
  - [Authentication: the approval gate has a lock on it](#authentication-the-approval-gate-has-a-lock-on-it)
  - [MCP: remote tools under local risk policy](#mcp-remote-tools-under-local-risk-policy)
- [Technology stack](#technology-stack)


## Problem statement

Most agentic AI work stops at a prototype: a prompt, a model call, a response.
That gap — between a notebook demo and a system an organization can actually
run — is where the hard engineering lives. This project exists to close it: to
build an agent platform with the same rigor as any other production service —
configuration, structured logging and correlation, testing, reliability,
security, durability and deployment — rather than clever prompting with a web
framework around it.

## Architecture

A FastAPI service with the domain and the agent layered on top of an
operational base (configuration, structured logging, correlation IDs), and the
two things that cross a boundary — the model and the task store — behind ports
with more than one adapter each.

```
                +--------------------------------------------------+
                |                 FastAPI service                  |
                |                                                  |
  HTTP  ----->  |  auth (bearer token -> Principal, scopes)        |
                |      |                                           |
                |      v                                           |
                |  /tasks router                                   |
                |      |                                           |
                |      v                                           |
                |  task service  -->  TaskRepository (port)        |
                |      |                  +--> in-memory adapter   |
                |      |                  +--> PostgreSQL adapter  |
                |      v                                           |
                |  AgentTask state machine                         |
                |      ^                                           |
                |      +-- AgentRunner (model <-> tool loop)       |
                |            +--> LLMClient --> provider port      |
                |            +--> ToolRegistry (typed tools)       |
                |            +--> MCP client --> transport port    |
                |            +--> approval + checkpoint/resume     |
                |                                                  |
                |  cross-cutting: config · JSON logging ·          |
                |  X-Request-ID + principal correlation ·          |
                |  tracing (later)                                 |
                +--------------------------------------------------+
```

Modules:

| Module                                 | Responsibility                                  |
| -------------------------------------- | ----------------------------------------------- |
| `enterprise_agent_platform.config`     | Environment-based settings (Pydantic Settings)  |
| `enterprise_agent_platform.logging`    | Structured JSON logging to stdout               |
| `...request_context`                   | `X-Request-ID` middleware, request access logs  |
| `enterprise_agent_platform.main`       | App factory, lifespan, `/health`, router wiring |
| `...auth.models`                       | `Principal`, `APIClient`, scopes, approval policy |
| `...auth.tokens`                       | Bearer tokens resolved by digest to a principal |
| `...auth.dependencies`                 | Route dependencies: 401 vs 403, scope checks    |
| `...auth.context`                      | The request's principal, for logs               |
| `...tasks.models`                      | `AgentTask`, lifecycle state machine, checkpoint |
| `...tasks.repository`                  | `TaskRepository` port + in-memory adapter       |
| `...tasks.postgres`                    | asyncpg adapter, table constraints, idempotent DDL |
| `...tasks.factory`                     | Store selection from settings                   |
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
| `...agent.runner`                      | The run loop: model calls, tools, budgets, approval |
| `...mcp.protocol`                      | MCP wire types and failure taxonomy             |
| `...mcp.transport`                     | `MCPTransport` port + stdio subprocess adapter  |
| `...mcp.client`                        | Handshake, tool discovery, `tools/call`         |
| `...mcp.policy`                        | Server config and locally owned risk mapping    |
| `...mcp.tools`                         | Remote tools adapted into the tool registry     |
| `...mcp.factory`                       | Connect configured servers, register their tools |
| `...demo.data` / `.tools`              | Synthetic accounts-payable ledger and its four tools |
| `...demo.provider`                     | Scripted stand-in for a model, for the offline demo |
| `...demo.credentials`                  | The two credentials the demo deployment issues  |
| `...demo.walkthrough`                  | The end-to-end demo, which asserts its own claims |

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

### Foundation, domain, agent, persistence

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
- **Explicit state machine over free-form status updates.** Agent tasks are
  driven by several actors: the agent runner, human approvers, and API clients. A
  single transition table rejects impossible states such as completing a task
  that never ran, and each change is appended to an audit history.
- **Immutable tasks with versioning.** Transitions return a new `AgentTask`
  with `version + 1` rather than mutating in place, which keeps history
  append-only and makes stale writes detectable.
- **Optimistic concurrency in the repository.** `update()` takes the version the
  caller read and fails with `ConcurrentUpdateError` (HTTP 409) if the task has
  moved on. This avoids lost updates when, say, an approval and a cancellation
  race, without holding locks across slow LLM or human steps. The PostgreSQL
  adapter maps this to `UPDATE ... WHERE id = $1 AND version = $7`, so the check
  and the write are one statement evaluated under the row lock.
- **Two repository adapters, one contract test suite.** The in-memory and
  PostgreSQL adapters run the same tests (`tests/test_task_repository.py`);
  anything an adapter may differ on is deliberately not in them. The PostgreSQL
  parameters skip unless `EAP_TEST_DATABASE_URL` is set, so the default suite
  stays offline, and CI runs them against a real server — including a test where
  two writes on the same version are issued concurrently, which is the only way
  to catch a version check implemented as two statements.
- **`history` and `checkpoint` are `jsonb` columns, not child tables.** Both are
  only ever read with their task and never queried across tasks, so inlining
  them keeps a transition to a single statement — the same one that performs the
  version check. Child tables would buy queries nobody makes in exchange for a
  multi-statement write whose atomicity then depends on the transaction.
- **The domain's checkpoint invariant is restated as a table constraint.** Only
  an `awaiting_approval` task may carry run state; a `CHECK` enforces it in the
  database, so a future writer that bypasses the model still cannot leave a
  replayable conversation on a finished task. The status `CHECK` list is
  generated from the `TaskStatus` enum so the two cannot drift.
- **Rows are validated back through the domain model on read.** A task written
  by an older build, or edited by hand, has to satisfy the state machine's
  invariants before anything acts on it.
- **Schema creation is not application startup.** `apply_schema` is idempotent
  DDL for local development and tests (`python -m
  enterprise_agent_platform.tasks.postgres`); the service itself never holds DDL
  privileges, and a real deployment substitutes a migration tool.
- **The store is deployment configuration, not application logic.**
  `EAP_TASK_STORE` selects the adapter, `memory` is rejected in production
  (a restart would strand an approval that has already been made), and
  `postgres` without a URL fails at startup rather than on the first write. The
  driver import is deferred, so asyncpg stays an optional dependency.
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
- **The vendor SDK stops at the adapter.** Two modules import `anthropic`:
  `anthropic_provider`, which translates its types and errors, and `llm.factory`,
  which constructs the SDK client for the configured backend — choosing a backend
  is deployment configuration, so it does not belong inside the adapter that
  speaks to one. Nothing above the port sees an SDK type. The adapter maps HTTP
  status codes to the taxonomy
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
- **A pause checkpoints the run, so approval continues it instead of restarting
  it.** The conversation so far, the steps spent, the tool calls made, and the
  tokens burned are stored on the task as a `RunCheckpoint`. Restarting on
  approval would re-ask the model what it has already answered, re-run tools
  that already ran, and quietly reset the step budget — so a task could be
  paused and approved its way past `EAP_AGENT_MAX_STEPS` indefinitely.
- **The checkpoint lives on the task, not in a second store.** Approving,
  rejecting, and cancelling all decide the same thing — what this task does next
  — and the version check exists to make exactly one of them win. Splitting run
  state into its own store would allow a task in `awaiting_approval` with no
  conversation behind it, a state nothing can act on. Every transition rewrites
  the checkpoint and clears it by default, so a resumed or finished task cannot
  carry a stale conversation that a later approval replays.
- **Resuming re-checks the version the approver read.** `/approve` reads the
  checkpoint, then transitions; `transition_task` takes the version from that
  read, so a second approver or a cancellation landing in between produces a 409
  rather than a second agent replaying the same critical calls.
- **Rejection ends the task; it is not fed back to the model.** Returning a
  denial as a recoverable tool error would invite the agent to route around a
  decision a human just made, which is exactly what an approval gate exists to
  prevent. The rejection is recorded with who made it, and the checkpoint is
  dropped so the denied calls cannot be replayed.
- **Approval is a view, not just a verb.** `GET /tasks/{id}/approval` shows the
  whole held turn with each call's arguments and risk, and marks which ones
  tripped the gate — approving a tool by name alone is theatre, and approving
  one action while its siblings run unexamined is not an informed decision. The
  view reads the same threshold the pause did, so it cannot disagree with what
  the run will do. The checkpoint itself is never exposed on `GET /tasks/{id}`:
  it carries model output and tool results, and this is the only slice of it
  anyone needs to act on.
- **Agent failures are outcomes, not exceptions.** A provider error, a refusal,
  a truncated answer, an exhausted budget — each ends with the task in `failed`
  and a reason in its history; only genuine lifecycle errors (unknown task, task
  not startable) propagate as 404/409. The reason returned to the caller is the
  same string recorded in the audit trail, so the API and the history cannot
  disagree. Provider error text is not copied into it, since it can echo the
  prompt.
- **The prompt carries the goal, not the requester.** `requested_by` is
  authenticated now, and it still stays out of the prompt: identity in a
  conversation is an authorization claim the model is in no position to check,
  and a model that reads who asked can be talked into deferring to it. Authority
  is decided by scopes and by the approval policy, outside the conversation. The
  system prompt tells the model that tool results are data, never instructions.
- **`POST /tasks/{id}/run` is synchronous for now.** The caller waits, which is
  honest for a single-process deployment. Because progress is recorded on the
  task rather than in the response, moving execution to a queue and returning
  202 changes the entrypoint, not the domain logic.
- **One log record per run** (`agent.run.completed`) with steps, tool calls,
  token totals, latency, and the originating request ID. The request ID is
  passed into `AgentRunner.run` explicitly rather than read from a context
  variable, so a run handed to a background worker still correlates with the
  request that created the task.

### Authentication: the approval gate has a lock on it

An approval gate that any caller can walk through is not a gate, and an
`approved_by` field a caller fills in is an audit trail that records whatever it
was told. Both routes that release held tool calls, and every route that starts
or reads work, now require a credential.

- **Identity comes from the credential, never from the body.** `requested_by`,
  `approved_by` and `rejected_by` are no longer request fields; they are the
  authenticated subject, written into the task's immutable history. The request
  schemas keep `extra="forbid"`, so a body that tries to name the approver is a
  422 rather than a silently ignored claim. What is left in a body is the part
  only a human can supply: the note or the reason. Cancellation records its
  subject the same way — a history that names who asked and who approved but not
  who cancelled leaves the one transition anyone disputes unattributed.
- **Three types, kept apart.** `APIClient` is configuration — a credential a
  deployment issues, holding a secret that never leaves the process.
  `Principal` is identity — subject plus scopes, carrying no secret, which is
  what routes, logs and the audit trail see. `ApprovalPolicy` is authority over
  a specific decision, which a scope cannot express: holding `tasks:approve`
  says you may approve tasks, not that you may approve *this* one.
- **`tasks:write` and `tasks:approve` are different scopes.** The whole point of
  the gate is that the authority to make an agent act is not the authority to
  release what it wants to do. A deployment may issue both to one credential —
  that is a choice, and the history still records which subject acted.
- **Separation of duties, by default.** A task cannot be approved by the subject
  that requested it. The requester already decided they wanted the action, so
  their approval carries no new information: it is a delay, not a review.
  `EAP_APPROVAL_REQUIRES_SECOND_PERSON=false` turns it off, because a
  single-operator deployment that switches it off has made a decision, while one
  that never had the check has an approval gate in name only. Rejection has no
  such check — withholding a capability needs no second opinion, and requiring
  one would leave someone who spotted their own mistake unable to stop the agent
  acting on it.
- **Only digests are stored, and lookup is by digest.** Tokens are held as
  SHA-256 digests, so a heap dump or a careless `repr` yields nothing
  replayable, and a presented token is hashed and looked up rather than compared
  against a list — the work done is the same whether a token is wrong in its
  first character or its last. Scanning with `==` would leak a prefix oracle.
  SHA-256 rather than a password hash on purpose: these are high-entropy machine
  credentials, so there is no dictionary to slow down, and a deliberately slow
  KDF on every request is a denial-of-service lever.
- **401 and 403 are not interchangeable.** 401 with `WWW-Authenticate: Bearer`
  means *we do not know who you are*, and presenting a credential would change
  the answer. 403 means *we know who you are and it is not enough*, and no
  challenge is sent, because inviting a client to re-authenticate against a scope
  problem produces a retry loop rather than a fix. Authentication runs before
  request validation, so an anonymous caller learns nothing about the shape of
  the API from a 422.
- **Closed by default, and refused in production if closed.** No configured
  credentials means nothing can authenticate, so an unconfigured deployment
  authenticates nobody rather than everybody. In production that state is
  refused at startup: a service that can authenticate nobody is misconfigured
  rather than safe, and it is cheaper to learn that at boot than from a pager.
- **The subject lands on every log record of the request**, next to the request
  ID — which call was this, and who made it, the two halves of the question an
  incident opens with. Scopes and tokens are never logged: a token in a log file
  is a credential in a log file, and rejections are logged with a reason instead.
- **`/health` stays open** so a load balancer does not need a credential, and
  the read routes are closed with the rest: a task's goal, the model's answer and
  the arguments of the tools it wanted to run are not less sensitive than
  starting one.
- **Static tokens, deliberately.** This is the layer an OIDC or JWT verifier
  would replace, and replacing it touches one module, because routes depend on
  `Principal` rather than on how one was obtained. Users, groups, per-task
  ownership and token rotation are out of scope on purpose — the point is that
  the gate has a lock, not that this is an IAM system.

### MCP: remote tools under local risk policy

A deployment can source tools from external MCP servers, declared in
`EAP_MCP_SERVERS`:

```json
[
  {
    "name": "finance",
    "command": "python",
    "args": ["-m", "finance_mcp"],
    "default_risk": "read",
    "tool_risk": { "pay_invoice": "critical" }
  }
]
```

Each server is started at application startup, handshaken, and asked for its
tools; every tool becomes an ordinary `Tool` in the ordinary `ToolRegistry`,
registered as `finance__pay_invoice`. From there the agent runner, the risk
gate, the approval workflow, the per-call timeout, and result truncation apply
to it unchanged — no part of the agent imports anything from the `mcp` package,
and swapping a local tool for a remote one changes no agent code.

The design decisions worth naming:

- **Risk is decided locally, never read from the server.** Tool discovery
  happens at runtime against code outside this repository, so a server could
  otherwise add `transfer_funds` to a running agent, or advertise
  `readOnlyHint: true` on a payment tool. Risk resolves from a per-tool
  override, then the server's configured default, then `CRITICAL` — so an
  unmapped tool is gated rather than waved through. The failure mode of this
  design is an unnecessary approval prompt; it is never an ungated action.
- **Server metadata cannot reach the decision.** `MCPToolDeclaration` parses
  with `extra="ignore"`, so annotations, hints and titles are dropped at the
  parse boundary. The platform has nowhere to put a server's claim about its own
  safety, which is what makes the policy unbypassable rather than merely
  unbypassed. This is the posture the runner already takes toward tool *results*
  (data, not instructions), extended to tool *declarations*.
- **The model sees the server's schema; the platform validates with its own.**
  The server's JSON Schema carries per-property descriptions and constraints
  that make tool calls land, so it is what is offered to the model. A Pydantic
  model derived from it does the validating, so a remote handler never receives
  an unvalidated dictionary. The derived model is deliberately a coarsening —
  required fields and broad types, nothing else — so it can never reject
  arguments the schema the model was shown would allow.
- **Server-side tool failures are data; protocol failures are not.** `isError`
  on a `tools/call` result becomes `ToolResult(is_error=True)` and the model can
  correct itself. A dead subprocess or a malformed frame raises instead: the
  model cannot fix it, and the text can name hosts and commands.
- **The client bounds what a server can impose.** Tools per server, pages of
  `tools/list`, characters per result, and bytes per JSON-RPC frame are all
  capped, because each is otherwise a remote party choosing this platform's
  context window, per-call cost, or memory ceiling.
- **The child process inherits no environment.** An MCP server is third-party
  code with a shell on the host; handing it this process's API keys and database
  URL because it was started from here is an avoidable credential leak. A
  deployment passes exactly what the server needs.
- **The transport is a port.** `MCPTransport` is request/notify/close;
  `StdioTransport` runs a child process and multiplexes replies by JSON-RPC id
  so parallel tool calls share one pipe. The whole client is therefore testable
  against an in-process stub, and the suite also runs a real subprocess server
  — offline, with no MCP server installed.

## Technology stack

- Python 3.12+
- FastAPI + Uvicorn
- Anthropic SDK (Messages API, first-party or Bedrock)
- Pydantic v2 / pydantic-settings
- PostgreSQL (asyncpg), optional — in-memory store by default
- pytest + pytest-asyncio
- Ruff (lint + format), mypy (strict)
- Docker, GitHub Actions
