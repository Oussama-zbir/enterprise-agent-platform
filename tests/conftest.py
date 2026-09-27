"""Test-wide isolation from the developer's environment.

The suite is offline and deterministic, and several tests assert what the
configuration *refuses*: that `EAP_TASK_STORE=postgres` without a URL fails, that
`production` rejects the fake backend, that no tools are registered unless a
deployment asks. Those assertions are about the absence of a setting, so an
`EAP_` variable exported in the shell — while running the demo, or pointing at
the compose database — would otherwise decide the outcome of the test.

Two things are therefore reset before every test: the ambient `EAP_` variables,
and the `get_settings` cache, which importing the application populates on the
way to building its module-level ASGI `app`.

`EAP_TEST_DATABASE_URL` is kept. It is the suite's own switch for the PostgreSQL
contract parameters, not application configuration.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from enterprise_agent_platform.config import get_settings

_KEPT = frozenset({"EAP_TEST_DATABASE_URL"})


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test the configuration it declares, and nothing else.

    Runs before the test body, so a test that sets a variable deliberately still
    sees its own value; the cache is cleared again afterwards so the value it set
    does not survive into the next one.
    """
    for name in [n for n in os.environ if n.startswith("EAP_") and n not in _KEPT]:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()
