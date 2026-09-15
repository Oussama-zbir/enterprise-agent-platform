"""LLM failure taxonomy.

Adapters translate vendor exceptions into these types so callers can make one
decision — retry, fail the task, or ask a human — without knowing which SDK
raised. ``retryable`` marks transient failures where the same request may
succeed later; it does not decide *whether* to retry, which is caller policy.
"""

from __future__ import annotations


class LLMError(Exception):
    retryable: bool = False


class LLMTimeoutError(LLMError):
    retryable = True


class LLMRateLimitError(LLMError):
    retryable = True

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMUnavailableError(LLMError):
    """Connection failures and provider-side 5xx errors."""

    retryable = True


class LLMRequestError(LLMError):
    """The provider rejected the request itself (e.g. 400, auth, unknown model)."""


class LLMRefusalError(LLMError):
    """The model declined to answer; resending the same request will not help."""


class StructuredOutputError(LLMError):
    """The response could not be parsed into the requested schema."""

    def __init__(self, message: str, *, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text
