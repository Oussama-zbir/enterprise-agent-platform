# Enterprise Agent Platform

A production-oriented platform for building and operating enterprise AI agents.

> **Status: foundation.** This repository is being built incrementally. Today it
> establishes the engineering foundation — service skeleton, configuration,
> structured logging, testing, typing, linting, containerization, and CI.
> Agent orchestration, tool calling, MCP integration, human-in-the-loop
> approval, and evaluation are planned milestones (see [Roadmap](#roadmap)).

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
  HTTP  ----->  |  routes  -->  agent orchestration (later)        |
                |                 |                                |
                |                 +--> tool calling / MCP (later)  |
                |                 +--> human-in-the-loop (later)   |
                |                                                  |
                |  cross-cutting: config · structured logging ·   |
                |  evaluation & observability (later)              |
                +--------------------------------------------------+
```

Current modules:

| Module                                 | Responsibility                                  |
| -------------------------------------- | ----------------------------------------------- |
| `enterprise_agent_platform.config`     | Environment-based settings (Pydantic Settings)  |
| `enterprise_agent_platform.logging`    | Structured JSON logging to stdout               |
| `enterprise_agent_platform.main`       | App factory, lifespan, `/health` endpoint       |

## Design decisions

- **Application factory (`create_app`)** rather than a single global app, so
  tests and future deployments can build isolated instances with overridden
  configuration.
- **Environment-based config with an `EAP_` prefix** via Pydantic Settings —
  twelve-factor style, validated at load time, no config scattered in code.
- **Structured JSON logging, dependency-free** — parseable in containers and
  cloud log aggregators today; a full OpenTelemetry tracing stack is deferred
  to the observability milestone rather than added prematurely.
- **`src/` layout** to keep the importable package separate from tooling and
  tests, and to catch packaging mistakes early.
- **Strict typing and linting from day one** so quality is enforced by CI
  before the codebase grows.

## Technology stack

- Python 3.12+
- FastAPI + Uvicorn
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

1. **Foundation** — service skeleton, config, logging, tests, CI ✅ *(current)*
2. **Core domain** — request/task modeling, persistence
3. **AI capability** — agent orchestration, tool calling, MCP integration
4. **Human-in-the-loop** — approval workflows, structured state
5. **Evaluation** — agent evaluation harness and metrics
6. **Observability** — tracing, latency/cost accounting (OpenTelemetry)
7. **Reliability & security** — retries, rate limits, prompt-injection defense
8. **Deployment** — cloud-oriented deployment and infrastructure

## Limitations

This is an early foundation. It does **not** yet perform any agent work — there
is no LLM integration, tool calling, or persistence. The health endpoint and
configuration exist to anchor the operational foundation the rest of the
platform will build on.

## License

MIT
