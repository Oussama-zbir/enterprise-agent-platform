"""Authentication and authorisation for the HTTP surface."""

from enterprise_agent_platform.auth.context import bind_principal, get_principal
from enterprise_agent_platform.auth.models import (
    APIClient,
    ApprovalPolicy,
    Principal,
    Scope,
)
from enterprise_agent_platform.auth.tokens import TokenAuthenticator

__all__ = [
    "APIClient",
    "ApprovalPolicy",
    "Principal",
    "Scope",
    "TokenAuthenticator",
    "bind_principal",
    "get_principal",
]
