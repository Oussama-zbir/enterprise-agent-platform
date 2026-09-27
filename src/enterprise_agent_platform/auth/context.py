"""The authenticated principal of the request currently being handled.

The same reasoning as ``request_context``: an agent run fans out into model
calls, tool invocations and an approval pause, and "which request was this"
is only half of the question an incident asks. The other half is "who asked
for it", and threading a principal through every call signature to answer it
would put identity into layers that have no business knowing about HTTP.

Only the subject is ever emitted. Scopes are an authorisation input, not an
audit fact, and the credential itself never enters this module.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from enterprise_agent_platform.auth.models import Principal

_principal: ContextVar[Principal | None] = ContextVar("principal", default=None)


def get_principal() -> Principal | None:
    """Return the caller authenticated for this request, if any."""
    return _principal.get()


@contextmanager
def bind_principal(principal: Principal) -> Iterator[None]:
    """Attach ``principal`` to logs emitted inside the block."""
    token = _principal.set(principal)
    try:
        yield
    finally:
        _principal.reset(token)
