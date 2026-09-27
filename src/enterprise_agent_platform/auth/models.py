"""Identity and authority: who is calling, and what they are allowed to do.

Three types, kept apart on purpose:

* ``APIClient`` is *configuration* — a credential a deployment issues. It holds
  a secret, so it never leaves this process and never reaches a response.
* ``Principal`` is *identity* — the authenticated subject and the scopes it
  holds. It is what routes, logs, and the task audit trail see, and it carries
  no secret, so it is safe to put in both.
* ``ApprovalPolicy`` is *authority over a specific decision*, which scopes
  cannot express: holding ``tasks:approve`` says you may approve tasks, not
  that you may approve *this* one.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

SUBJECT_PATTERN = r"^[A-Za-z0-9._@:-]{1,128}$"
"""Subjects are written into logs and into the immutable task history, so they
are restricted to an alphabet that cannot forge a log line or a JSON field."""

MIN_TOKEN_LENGTH = 32
"""A shared secret short enough to be guessed offline is not a credential.
32 characters is the length of ``secrets.token_urlsafe(24)``."""


class Scope(StrEnum):
    """What a credential is allowed to do.

    Deliberately coarse. The split that matters is ``tasks:write`` versus
    ``tasks:approve``: the whole point of the approval gate is that the
    authority to make an agent act is not the authority to release what it
    wants to do. A deployment that issues both to one credential has chosen to
    collapse that, and the audit trail still records which one it was.
    """

    TASKS_READ = "tasks:read"
    TASKS_WRITE = "tasks:write"
    TASKS_APPROVE = "tasks:approve"


class Principal(BaseModel):
    """An authenticated caller. Contains no secret, by construction."""

    model_config = ConfigDict(frozen=True)

    subject: str = Field(pattern=SUBJECT_PATTERN)
    scopes: frozenset[Scope] = Field(min_length=1)

    def holds_any(self, scopes: frozenset[Scope]) -> bool:
        return bool(self.scopes & scopes)


class APIClient(BaseModel):
    """One credential this deployment has issued."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: str = Field(
        pattern=SUBJECT_PATTERN,
        description="Recorded as the actor in logs and task history. Name a person or a service.",
    )
    token: SecretStr = Field(description="Bearer token. Compared by digest and never logged.")
    scopes: frozenset[Scope] = Field(min_length=1)

    @field_validator("token")
    @classmethod
    def _long_enough_to_be_a_secret(cls, token: SecretStr) -> SecretStr:
        if len(token.get_secret_value()) < MIN_TOKEN_LENGTH:
            raise ValueError(f"API tokens must be at least {MIN_TOKEN_LENGTH} characters")
        return token

    @property
    def principal(self) -> Principal:
        """The secret-free identity this credential authenticates to."""
        return Principal(subject=self.subject, scopes=self.scopes)


class ApprovalPolicy(BaseModel):
    """Who may release the tool calls a paused run is holding.

    Separation of duties is the control the approval gate exists for. An agent
    asked to do something dangerous by one person, and released by that same
    person, has been through a delay rather than through a review — the
    requester already decided they wanted this, so their approval carries no
    new information. Requiring a second subject makes the gate a second
    judgement.

    It is a deployment decision rather than a constant because a single-operator
    deployment that turns it off has made a choice; one that never had it has
    an approval gate in name only.
    """

    model_config = ConfigDict(frozen=True)

    requires_second_person: bool = True

    def may_approve(self, principal: Principal, *, requested_by: str) -> bool:
        """Scope has already been checked; this is authority over *this* task."""
        return not (self.requires_second_person and principal.subject == requested_by)
