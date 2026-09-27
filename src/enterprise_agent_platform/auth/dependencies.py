"""FastAPI wiring for authentication and scope checks.

Routes depend on ``Principal``, never on a header or a token, so replacing
static tokens with OIDC means replacing this module alone.

The status codes are not interchangeable. 401 with ``WWW-Authenticate: Bearer``
means *we do not know who you are* — presenting a credential would change the
answer. 403 means *we know who you are and it is not enough* — retrying with
the same credential never will be, and no challenge header is sent, because
inviting a client to re-authenticate against a scope problem produces a retry
loop rather than a fix.

Rejections are logged with a reason and never with the presented token: a token
in a log file is a credential in a log file.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from http import HTTPStatus
from typing import Annotated, NoReturn

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from enterprise_agent_platform.auth.context import bind_principal
from enterprise_agent_platform.auth.models import ApprovalPolicy, Principal, Scope
from enterprise_agent_platform.auth.tokens import TokenAuthenticator

logger = logging.getLogger(__name__)

# `auto_error=False` so this module owns the failure shape (reason logging and
# the challenge header) instead of Starlette's bare 403. The scheme is declared
# here rather than read by hand so it appears in the OpenAPI document and the
# docs page grows an Authorize button. Credentials are accepted in this header
# only: query strings end up in access logs, proxy logs and browser history.
_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="BearerToken",
    description="An API token issued by this deployment (EAP_API_CLIENTS).",
)


def get_authenticator(request: Request) -> TokenAuthenticator:
    authenticator: TokenAuthenticator = request.app.state.authenticator
    return authenticator


def get_approval_policy(request: Request) -> ApprovalPolicy:
    policy: ApprovalPolicy = request.app.state.approval_policy
    return policy


CredentialsDep = Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]
AuthenticatorDep = Annotated[TokenAuthenticator, Depends(get_authenticator)]
ApprovalPolicyDep = Annotated[ApprovalPolicy, Depends(get_approval_policy)]


def _unauthenticated(reason: str) -> NoReturn:
    logger.warning("auth.rejected", extra={"reason": reason})
    raise HTTPException(
        HTTPStatus.UNAUTHORIZED,
        detail="Bearer token required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def current_principal(
    credentials: CredentialsDep, authenticator: AuthenticatorDep
) -> AsyncIterator[Principal]:
    """Authenticate the caller and bind it to the logs of this request.

    A generator dependency so the binding is unwound when the request ends,
    rather than left on a context that a later task could inherit.
    """
    if credentials is None:
        # Covers both a missing header and a scheme this API does not accept;
        # which one it was is the client's business, not an error message's.
        _unauthenticated("missing_bearer_credentials")
    principal = authenticator.authenticate(credentials.credentials)
    if principal is None:
        _unauthenticated("unknown_token")
    with bind_principal(principal):
        yield principal


PrincipalDep = Annotated[Principal, Depends(current_principal)]


def requires(*scopes: Scope) -> Callable[[Principal], Principal]:
    """Dependency factory: the caller must hold at least one of ``scopes``.

    Any-of rather than all-of because the scopes here are alternative reasons
    to be allowed through a route, not a set of permissions to accumulate: an
    approver may read the decision in front of them without also being issued
    the blanket read scope.
    """
    accepted = frozenset(scopes)

    def check_scope(principal: PrincipalDep) -> Principal:
        if not principal.holds_any(accepted):
            logger.warning(
                "auth.rejected",
                extra={
                    "reason": "insufficient_scope",
                    "principal": principal.subject,
                    "required_scopes": sorted(accepted),
                },
            )
            raise HTTPException(
                HTTPStatus.FORBIDDEN,
                detail=f"This token holds none of: {', '.join(sorted(accepted))}.",
            )
        return principal

    return check_scope
