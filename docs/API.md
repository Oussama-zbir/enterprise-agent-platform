# API and configuration reference

Every route, every failure code, and the environment variables that select a
backend. The [README](../README.md) covers getting it running; this is the
detail. Interactive docs are served at `/docs` when the service is up.

## Running it locally

Run the service:

```bash
uvicorn enterprise_agent_platform.main:app --reload
```

Then:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","service":"enterprise-agent-platform","environment":"development","version":"0.1.0"}
```

`/health` is the only open route. Everything below presents a token from
`EAP_API_CLIENTS` (see [Configuration](#configuration)); without one the answer is
`401` with `WWW-Authenticate: Bearer`, and with a token that lacks the scope the
route needs it is `403`:

```bash
export WRITER=<a token whose scopes include tasks:write>
export APPROVER=<a token whose scopes include tasks:approve>
```

Create, inspect, and cancel a task. The requester is the token's subject, so
there is no `requested_by` field to send:

```bash
curl -s -X POST http://127.0.0.1:8000/tasks \
  -H "Authorization: Bearer $WRITER" -H 'Content-Type: application/json' \
  -d '{"goal": "Reconcile supplier payments for March"}'
# {"id":"<uuid>","requested_by":"analyst-1","status":"pending","version":1,...,"history":[]}

curl -s 'http://127.0.0.1:8000/tasks?status=pending&limit=20' -H "Authorization: Bearer $WRITER"
curl -s http://127.0.0.1:8000/tasks/<uuid> -H "Authorization: Bearer $WRITER"

curl -s -X POST http://127.0.0.1:8000/tasks/<uuid>/cancel \
  -H "Authorization: Bearer $WRITER" -H 'Content-Type: application/json' \
  -d '{"reason": "duplicate request"}'
# 200 with status "cancelled" and history "cancelled by analyst-1: duplicate
# request"; cancelling again returns 409 Conflict
```

Run a task with the agent loop:

```bash
curl -s -X POST http://127.0.0.1:8000/tasks/<uuid>/run -H "Authorization: Bearer $WRITER"
# {"task":{"status":"completed","version":3,"history":[...]},
#  "output":"...", "detail":"agent run completed",
#  "steps":3, "tool_calls":2, "input_tokens":909, "output_tokens":109,
#  "pending_tool_calls":[]}
```

The run drives the task: `pending -> running -> completed | failed`, or
`awaiting_approval` when the model asks for a tool riskier than
`EAP_AGENT_AUTO_APPROVE_UP_TO` (then `pending_tool_calls` names what it wants).
Running the same task twice returns 409. The platform ships with no tools of its
own: a deployment registers them through `create_app(tool_registry=...)`, points
`EAP_MCP_SERVERS` at an MCP server, or sets `EAP_DEMO_TOOLS=true` for the
synthetic set above.

Approve or reject what a paused run wants to do:

```bash
curl -s http://127.0.0.1:8000/tasks/<uuid>/approval -H "Authorization: Bearer $APPROVER"
# {"task_id":"<uuid>","goal":"Settle invoice INV-1","status":"awaiting_approval",
#  "paused_at":"...","detail":"approval required for: pay_invoice",
#  "pending_tool_calls":[{"id":"call_1","name":"pay_invoice",
#    "arguments":{"invoice_id":"INV-1"},"risk":"critical","needs_approval":true}]}

curl -s -X POST http://127.0.0.1:8000/tasks/<uuid>/approve \
  -H "Authorization: Bearer $APPROVER" -H 'Content-Type: application/json' \
  -d '{"note": "supplier verified"}'
# the held calls run, the run continues from its checkpoint, and the response is
# a run outcome whose steps/tokens cover the whole run, pause included.
# The approver written into the history is this token's subject; an
# `approved_by` field in the body is a 422, and the subject that requested the
# task is a 403 unless EAP_APPROVAL_REQUIRES_SECOND_PERSON is off.

curl -s -X POST http://127.0.0.1:8000/tasks/<uuid>/reject \
  -H "Authorization: Bearer $APPROVER" -H 'Content-Type: application/json' \
  -d '{"reason": "supplier not verified"}'
# 200 with status "cancelled"; the held calls never run
```

| Method & path               | Scope           | Result                                  |
| --------------------------- | --------------- | --------------------------------------- |
| `GET /health`               | none            | 200; open, so probes need no credential |
| `POST /tasks`               | `tasks:write`   | 201 pending task; 422 on invalid or unknown fields |
| `GET /tasks`                | `tasks:read`    | Newest first; optional `status` filter, `limit` 1–200 |
| `GET /tasks/{id}`           | `tasks:read`    | 200, or 404 if unknown                  |
| `POST /tasks/{id}/run`      | `tasks:write`   | 200 with the run outcome; 404 unknown; 409 not pending |
| `GET /tasks/{id}/approval`  | `read`/`approve`| 200 with the held calls; 404 unknown; 409 if not paused |
| `POST /tasks/{id}/approve`  | `tasks:approve` | 200 with the resumed run's outcome; 403 if the requester; 404; 409 if not paused |
| `POST /tasks/{id}/reject`   | `tasks:approve` | 200 cancelled task; 404 unknown; 409 if not paused |
| `POST /tasks/{id}/cancel`   | `tasks:write`   | 200; 404 unknown; 409 terminal task or concurrent write |

Every closed route answers 401 with `WWW-Authenticate: Bearer` when no
credential is presented, and 403 with no challenge when the credential holds
none of the scopes the route accepts.

Every response carries an `X-Request-ID` header (the caller's, if well-formed,
otherwise a generated UUID), and every log line written during that request
includes it:

```bash
curl -s -i http://127.0.0.1:8000/tasks -H 'X-Request-ID: req-42' | grep -i x-request-id
# x-request-id: req-42
# log: {"message": "request.completed", "method": "GET", "path": "/tasks",
#       "status_code": 200, "duration_ms": 0.4, "request_id": "req-42", ...}
```

Records emitted while handling an authenticated request also carry the subject
that made it — which call, and who made it:

```bash
# log: {"message": "task.created", "task_id": "<uuid>", "request_id": "req-42",
#       "principal": "analyst-1"}
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

## Configuration

Every setting is an `EAP_`-prefixed environment variable or an `.env` entry,
validated at load time; `.env.example` documents all of them. The ones that
decide how the service behaves:

| Variable | Default | What it selects |
| --- | --- | --- |
| `EAP_ENVIRONMENT` | `development` | `production` refuses the fake/demo backends, the in-memory store, and an empty credential list |
| `EAP_LLM_PROVIDER` | `fake` | `fake` (offline, every call raises), `demo` (scripted), `anthropic`, `bedrock` |
| `EAP_TASK_STORE` | `memory` | `memory` or `postgres` |
| `EAP_API_CLIENTS` | `()` | The credentials this deployment issues; empty means every `/tasks` call is 401 |
| `EAP_APPROVAL_REQUIRES_SECOND_PERSON` | `true` | Whether the requester may approve their own task |
| `EAP_AGENT_AUTO_APPROVE_UP_TO` | `read` | Highest tool risk an agent may run unattended |
| `EAP_AGENT_MAX_STEPS` | `8` | Model calls allowed in one run |
| `EAP_MCP_SERVERS` | `()` | External MCP servers, and the risk this deployment assigns their tools |
| `EAP_DEMO_TOOLS` | `false` | Register the synthetic accounts-payable tools |

The default model backend is `fake`: offline, and every call raises. Point it at
a real model with environment variables (never commit a key):

```bash
# Anthropic API — omit the key to use the SDK's own credential resolution
EAP_LLM_PROVIDER=anthropic EAP_ANTHROPIC_API_KEY=sk-ant-... EAP_LLM_MODEL=claude-opus-5

# Bedrock — credentials come from the standard AWS chain
EAP_LLM_PROVIDER=bedrock EAP_AWS_REGION=eu-west-1 EAP_LLM_MODEL=anthropic.claude-opus-5
```

Every `/tasks` route needs a bearer token, and the deployment issues them. With
none configured, nothing can authenticate and every call is a 401 — closed by
default rather than open by default:

```bash
export EAP_API_CLIENTS='[
  {"subject":"analyst-1","token":"'"$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"'",
   "scopes":["tasks:read","tasks:write"]},
  {"subject":"finance-manager-2","token":"'"$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"'",
   "scopes":["tasks:read","tasks:approve"]}
]'
```

The subject is what lands in logs and in the task's audit history. `tasks:write`
starts work, `tasks:approve` releases what a paused run wants to do, and by
default a task cannot be approved by the subject that requested it. Tokens are
at least 32 characters, stored only as digests, and never logged.

Tasks are held in memory by default, which is lost on restart. For a durable
store, install the extra and point the service at PostgreSQL:

```bash
pip install -e ".[dev,postgres]"

export EAP_TASK_STORE=postgres
export EAP_DATABASE_URL=postgresql://eap:eap@localhost:5432/eap

# Create the table and indexes once (stands in for a migration tool)
python -m enterprise_agent_platform.tasks.postgres
```
