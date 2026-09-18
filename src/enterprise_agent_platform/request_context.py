"""Request-scoped correlation IDs.

Every HTTP request gets a request ID that is echoed in the ``X-Request-ID``
response header and attached to every log record emitted while the request is
handled. Agent runs will fan out into LLM calls, tool invocations, and approval
steps; a single ID that ties those log lines back to the originating request is
the minimum needed to debug them before full tracing (OpenTelemetry) lands.

The ID lives in a ``ContextVar`` so it follows the request across ``await``
points and into threadpool-executed sync code without being passed explicitly.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"

# Inbound IDs are logged and echoed back, so only accept short, header- and
# log-safe tokens (UUIDs, W3C trace IDs, most gateway formats). Anything else is
# replaced rather than trusted, which also blocks log injection via the header.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

logger = logging.getLogger(__name__)


def get_request_id() -> str | None:
    """Return the ID of the request currently being handled, if any."""
    return _request_id.get()


@contextmanager
def bind_request_id(request_id: str | None) -> Iterator[None]:
    """Attach ``request_id`` to logs emitted inside the block.

    The middleware covers work done while an HTTP request is open. Work that
    outlives it — an agent run handed to a background worker — has no such
    context, so the originating ID is passed explicitly and bound here. ``None``
    leaves whatever context is already in place untouched.
    """
    if request_id is None:
        yield
        return
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)


def resolve_request_id(candidate: str | None) -> str:
    """Reuse a well-formed caller-supplied ID, otherwise generate a new one."""
    if candidate is not None and _VALID_REQUEST_ID.fullmatch(candidate):
        return candidate
    return str(uuid4())


class RequestContextMiddleware:
    """Assign a request ID, expose it in the response, and log request outcomes.

    Implemented as pure ASGI rather than ``BaseHTTPMiddleware`` so the context
    variable is set in the same task that runs the endpoint and streaming
    responses are not buffered.

    Unhandled exceptions are logged with the request ID and turned into a JSON
    500 that still carries ``X-Request-ID``, so a client-reported ID leads
    straight to the traceback. If the response has already started, the
    exception is re-raised because nothing more can safely be sent.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        token = _request_id.set(request_id)
        started = time.perf_counter()
        status_code: int | None = None

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            logger.exception("request.failed")
            if status_code is not None:
                raise
            response = JSONResponse({"detail": "Internal Server Error"}, status_code=500)
            await response(scope, receive, send_with_request_id)
        finally:
            logger.info(
                "request.completed",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            _request_id.reset(token)
