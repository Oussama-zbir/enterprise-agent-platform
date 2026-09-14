"""Structured JSON logging.

Emits one JSON object per log record to stdout so logs are machine-parseable in
container and cloud environments. Kept dependency-free on purpose; a richer
tracing/observability stack (e.g. OpenTelemetry) is a later milestone.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from enterprise_agent_platform.request_context import get_request_id

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__)


class RequestContextFilter(logging.Filter):
    """Attach the current request ID to records emitted during a request.

    Runs in the emitting thread/task, where the request context is visible, so
    the ID is captured even if formatting is later moved to a queue listener.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = get_request_id()
        if request_id is not None:
            record.request_id = request_id
        return True


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Promote any non-reserved attributes attached via ``logger.info(..., extra=...)``.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger.

    Idempotent: existing handlers are replaced so repeated calls (e.g. app
    startup in tests) do not duplicate log output.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
