# Enterprise Agent Platform

[![CI](https://github.com/Oussama-zbir/enterprise-agent-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Oussama-zbir/enterprise-agent-platform/actions/workflows/ci.yml)
[![M8ven Verified](https://m8ven.ai/badge/mcp/oussama-zbir/enterprise-agent-platform?variant=verified)](https://m8ven.ai/mcp/oussama-zbir/enterprise-agent-platform)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![mypy strict](https://img.shields.io/badge/mypy-strict-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An agent platform built like a production service, not a prototype. An agent
drives a task through model and tool calls under a step budget; when it reaches a
tool riskier than the deployment allows unattended, the run **pauses,
checkpoints itself, and waits for a human** — and the approval that releases it
is authenticated, recorded, and refused if it comes from the person who asked for
the work.

Tools are typed and risk-classified locally, including tools sourced from
external **MCP** servers. Tasks, their audit history and the checkpoint a paused
run resumes from are durable in **PostgreSQL** under optimistic concurrency, so
two racing approvals cannot both release the same critical call.

```bash
docker compose up                             # API + PostgreSQL, one command
python -m enterprise_agent_platform.demo      # the whole flow, offline, no API key
```

---

## What this demonstrates

| | |
| --- | --- |
| **Agent orchestration** | A hand-written run loop — model call → tool execution → tool results → model call — bounded by a step budget, with cooperative cancellation and one-runner-per-task enforced by the domain's version check rather than a lock. No agent framework. |
| **Human-in-the-loop** | Approval **pauses a run and resumes it**: the paused conversation and the counters already spent are checkpointed on the task, so an approved run continues where it stopped instead of re-asking the model and re-running tools. A task cannot be paused and approved past its step budget. |
| **Authenticated approvals** | Bearer credentials resolved to a secret-free `Principal`; `tasks:write` and `tasks:approve` are separate scopes, and by default a task cannot be approved by the subject that requested it. The actor in the audit trail is the credential's subject — identity is not a request field. |
| **Local tool-risk governance** | Every tool carries a risk level, and one deployment-wide threshold decides what runs unattended. Risk is assigned **locally**, never read from the tool's own metadata, so a server that self-declares a payment as safe is still gated. |
| **MCP integration** | Tools published by external MCP servers are adapted into the platform's own typed tool layer, so the run loop, risk gate, approval workflow, timeouts and result truncation apply to them unchanged. The transport is a port; stdio is one adapter. |
| **Optimistic concurrency** | Immutable versioned tasks with an append-only transition history. In PostgreSQL the check and the write are one statement (`UPDATE … WHERE id = $1 AND version = $7 RETURNING version`), so they happen under the same row lock — the race is decided by the database, not by the application. |
| **PostgreSQL durability** | An asyncpg adapter behind the repository port: domain invariants restated as table `CHECK` constraints, JSONB for data only ever read with its aggregate, pool lifecycle tied to the app lifespan, and schema management kept out of startup. |
| **Provider abstraction** | A provider-neutral LLM port with a real Anthropic / Bedrock adapter behind it: the vendor SDK reaches two modules — the adapter that translates its types and the factory that builds its client — and nothing above the port sees an SDK type. Failure taxonomy with a `retryable` flag, hard timeouts, schema-validated structured output, and token/latency logs that never carry prompt content. |
| **Testing and CI** | 291 tests, no API key and no network. One contract suite both storage adapters must pass, run against a real `postgres:17` container in CI. Ruff, `mypy --strict`, and a CI job that starts the compose stack and drives the approval flow through it. |

## Architecture

```mermaid
flowchart TB
    analyst(["analyst<br/>tasks:write"])
    manager(["manager<br/>tasks:approve"])

    subgraph service["FastAPI service"]
        auth["auth<br/>bearer token to Principal<br/>subject + scopes"]
        api["/tasks API<br/>create · run · approval<br/>approve · reject · cancel"]
        runner["AgentRunner<br/>model and tool loop<br/>step budget · cancellation"]
        gate{"risk gate<br/>above the unattended<br/>threshold?"}
        registry["ToolRegistry<br/>typed args · risk level<br/>per-call timeout"]
        llm["LLMClient<br/>provider-neutral port"]
        repo["TaskRepository port<br/>version-checked writes"]
    end

    anthropic["Anthropic / Bedrock<br/>adapter"]
    mcp["MCP servers<br/>stdio transport<br/>risk assigned locally"]
    tools["deployment tools<br/>demo: accounts payable"]
    pg[("PostgreSQL<br/>task · audit history<br/>run checkpoint")]
    paused["awaiting_approval<br/>checkpoint: held calls,<br/>conversation, spent budget"]

    analyst --> auth
    manager --> auth
    auth --> api
    api --> runner
    runner --> llm
    llm --> anthropic
    runner --> registry
    tools --> registry
    mcp --> registry
    registry --> gate
    gate -->|"no: run it"| runner
    gate -->|"yes: hold the turn"| paused
    paused -->|"GET /approval<br/>arguments + risk"| manager
    manager -->|"approve: resume<br/>from the checkpoint"| runner
    runner --> repo
    api --> repo
    repo --> pg
```

Everything in that diagram is implemented. The detailed module map, the task
state machine, and the reasoning behind each decision are in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Demo — the whole flow, offline

No API key, no network, no database. A scripted provider stands in for the model;
everything else is the real system:

```bash
pip install -e ".[dev]"
python -m enterprise_agent_platform.demo
```

![Terminal transcript: an agent run pauses on a critical pay_invoice call, the requester's own approval is refused, a second person approves, and the run resumes from its checkpoint](docs/images/demo.svg)

<sub>Abbreviated for width: `$API` is `http://127.0.0.1:8000`, `<id>` is the task
UUID, and `…` marks elided JSON fields. Steps 1 and 6 — create, and the final
audit history — are printed in full by the command.</sub>

The demo drives the real application over HTTP through an in-process ASGI
transport (same routes, middleware, authentication, runner and risk gate) and
**asserts its own story**: the run must pause on the critical call, the
requester's approval must be refused, the approved run must continue the paused
run's budget rather than reset it, and the ledger must end with exactly one
payment. It exits non-zero otherwise, and the same code path runs as a test on
every CI build — so this section cannot quietly stop matching the system.

Two things in that transcript are the demo's, not the platform's. The scripted
provider is a stand-in for a model: it matches the goal to one of two planned
scenarios and reads each turn's arguments out of the previous turn's *actual*
tool results, so the run is still driven by what the tools returned, but it is not
reasoning. And token counts are estimated from character length rather than
tokenized. Point `EAP_LLM_PROVIDER` at `anthropic` and the same tools run against
a real model with nothing else changed.

## Quick start — the durable path

```bash
docker compose up
```

PostgreSQL, then the schema as a one-shot job, then the API waiting on both, with
the demo tools loaded and two credentials issued:

```bash
export API=http://localhost:8000
export ANALYST=demo-analyst-token-please-change-me
export MANAGER=demo-manager-token-please-change-me

id=$(curl -s -X POST $API/tasks -H "Authorization: Bearer $ANALYST" \
  -H 'Content-Type: application/json' \
  -d '{"goal": "Settle invoice INV-1043 with the supplier, in full."}' | jq -r .id)

curl -s -X POST "$API/tasks/$id/run"     -H "Authorization: Bearer $ANALYST" | jq .task.status
# "awaiting_approval"    <- the agent asked to pay; the gate held the whole turn

curl -s -X POST "$API/tasks/$id/approve" -H "Authorization: Bearer $ANALYST" | jq .detail
# "This token holds none of: tasks:approve."      <- 403

curl -s -X POST "$API/tasks/$id/approve" -H "Authorization: Bearer $MANAGER" | jq .task.status
# "completed"            <- the held call ran, on the same run's budget

docker compose restart api && docker compose up -d --wait api
curl -s "$API/tasks/$id" -H "Authorization: Bearer $ANALYST" | jq .status
# "completed"            <- task and audit history outlived the process that ran it
```

Three deliberate choices in that compose file. The schema is a one-shot job
rather than application startup, so the service never holds DDL privileges and a
migration tool can replace the job without the API changing. The API waits on the
database's healthcheck *and* on that job exiting 0, because an API that starts
before its table exists fails its first request instead of its boot. And the
database port is published, so the storage contract suite can be pointed at it.

The compose deployment is a demo, not a production template: its tokens are
published here, its model backend is scripted, and its tools move fictional
money. `EAP_ENVIRONMENT=production` refuses all of it, and additionally refuses
to boot with no credentials configured.

## API

| Method & path | Scope | Result |
| --- | --- | --- |
| `GET /health` | none | 200; open, so probes need no credential |
| `POST /tasks` | `tasks:write` | 201 pending task; the requester is the token's subject |
| `GET /tasks` | `tasks:read` | Newest first; `status` filter, `limit` 1–200 |
| `GET /tasks/{id}` | `tasks:read` | 200, or 404 |
| `POST /tasks/{id}/run` | `tasks:write` | 200 with the run outcome; 409 if not pending |
| `GET /tasks/{id}/approval` | `read`/`approve` | The held calls with their arguments and risk; 409 if not paused |
| `POST /tasks/{id}/approve` | `tasks:approve` | 200 with the resumed run's outcome; **403 if the requester**; 409 if not paused or already decided |
| `POST /tasks/{id}/reject` | `tasks:approve` | 200 cancelled task; the held calls never run |
| `POST /tasks/{id}/cancel` | `tasks:write` | 200; 409 on a terminal task or a concurrent write |

Closed routes answer 401 with `WWW-Authenticate: Bearer` when no credential is
presented, and 403 with no challenge when the credential holds none of the scopes
the route accepts. Every response carries `X-Request-ID`, and every log record
written during that request carries both the ID and the authenticated subject —
which call, and who made it.

Full reference, curl walkthrough and every configuration variable:
**[docs/API.md](docs/API.md)**.

## Testing and quality

```bash
ruff check . && ruff format --check .   # lint and formatting
mypy                                   # strict, over source and tests
pytest                                 # 291 tests, all offline
```

CI runs those on Python 3.12 and 3.13, then the storage contract suite against a
`postgres:17` service container, then a job that starts the compose stack and
drives create → run → refused approval → approval → restart → still `completed`.

The PostgreSQL parameters of the contract suite skip locally unless a database is
named:

```bash
EAP_TEST_DATABASE_URL=postgresql://eap:eap@localhost:5432/eap \
  pytest -v tests/test_task_repository.py
```

## Documentation

| | |
| --- | --- |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Problem statement, module map, task state machine, and the full design-decision record — including the authentication model and the MCP risk policy. The engineering argument behind every choice made here. |
| **[docs/API.md](docs/API.md)** | Endpoint reference with curl examples, correlation IDs in logs, and the complete configuration table. |
| **[.env.example](.env.example)** | Every setting, documented in place. |

## Limitations

Stated plainly, because a system's known edges are part of its design.

- **Runs are synchronous and in-process.** The caller waits, and a restart
  mid-run leaves a `running` task with nothing driving it — the task and its
  checkpoint survive, the in-flight execution does not. Moving execution behind a
  queue with a lease and a sweeper is the clearest next step, and it is a change
  to the entrypoint rather than to the domain, because progress is already
  recorded on the task instead of in the response.
- **Authentication is static bearer tokens issued by configuration.** No user
  store, no rotation or revocation beyond editing `EAP_API_CLIENTS` and
  restarting, no per-task ownership (any subject with `tasks:read` can read any
  task; any subject with `tasks:approve` can approve any task they did not
  request), and no rate limiting. It is the layer an OIDC or JWT verifier
  replaces — one module, because routes depend on `Principal` — not an IAM
  system.
- **No evaluation harness.** Nothing here measures the quality of agent output:
  no scored task suite, no LLM-judge, no regression baseline. Deliberately
  deferred rather than half-built — evaluation deserves its own repository rather
  than a ninth milestone in this one.
- **Observability is structured logs, not traces.** JSON to stdout with
  correlation IDs carried into work that outlives the request, plus per-run steps,
  tool calls, token totals and latency. There is no OpenTelemetry, no distributed
  tracing, and no cost accounting beyond those raw token counts.
- **No tools ship for real use.** A deployment registers its own or points
  `EAP_MCP_SERVERS` at a server. `EAP_DEMO_TOOLS=true` turns on the synthetic
  accounts-payable set, whose data is fictional and whose payments move nothing.
- **The model adapter has met a mock transport, not the live API**, and does not
  implement streaming, prompt caching or thinking blocks.
- **The MCP client speaks stdio only.** HTTP and SSE would be a second adapter
  behind the same transport port, not a change to the client. Tools are discovered
  once at startup, so a server that gains or loses a tool mid-process is not
  noticed until a restart.
- **No migration tool.** The schema is idempotent DDL behind
  `python -m enterprise_agent_platform.tasks.postgres` — correct for one table,
  and exactly where Alembic would go.

## License

MIT — see [LICENSE](LICENSE).
