"""The credentials the demo deployment issues.

Two subjects, because one is not enough to show the control that matters: the
analyst may start work but not release it, the manager may release it but not
start it, and neither can approve a task they asked for themselves.

These tokens are published in this file, in ``docker-compose.yml`` and in the
README, which is exactly why the demo deployment is refused in production. A
real deployment issues its own through ``EAP_API_CLIENTS`` and keeps them out of
its source tree.
"""

from __future__ import annotations

from pydantic import SecretStr

from enterprise_agent_platform.auth.models import APIClient, Scope

ANALYST_TOKEN = "demo-analyst-token-please-change-me"
MANAGER_TOKEN = "demo-manager-token-please-change-me"

DEMO_ANALYST = APIClient(
    subject="analyst-1",
    token=SecretStr(ANALYST_TOKEN),
    scopes=frozenset({Scope.TASKS_READ, Scope.TASKS_WRITE}),
)
DEMO_MANAGER = APIClient(
    subject="finance-manager-2",
    token=SecretStr(MANAGER_TOKEN),
    scopes=frozenset({Scope.TASKS_READ, Scope.TASKS_APPROVE}),
)
DEMO_API_CLIENTS = (DEMO_ANALYST, DEMO_MANAGER)
