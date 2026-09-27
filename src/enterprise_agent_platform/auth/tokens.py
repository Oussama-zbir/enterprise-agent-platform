"""Bearer-token authentication.

Static tokens issued by configuration: enough for a service-to-service API, and
the layer an OIDC/JWT verifier would later replace without any route changing,
because routes depend on ``Principal`` rather than on how one was obtained.

Two properties this implementation is built for:

* **The stored form is not a usable credential.** Only SHA-256 digests are
  held, so a heap dump, a traceback, or a careless ``repr`` yields nothing that
  can be replayed against the API.
* **Verification time does not depend on the secret.** Presented tokens are
  hashed and looked up by digest, so the comparison walks a digest rather than
  the secret, and the work done is the same whether a token is wrong in its
  first character or its last. Scanning a list of tokens with ``==`` would leak
  a prefix oracle instead.

SHA-256 rather than a password hash on purpose: these are high-entropy machine
credentials, not human passwords, so there is no dictionary to slow down — and
a deliberately slow KDF on every request would be a denial-of-service lever.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from enterprise_agent_platform.auth.models import APIClient, Principal


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TokenAuthenticator:
    """Resolves a presented bearer token to the principal that issued it."""

    def __init__(self, clients: Iterable[APIClient] = ()) -> None:
        index: dict[str, Principal] = {}
        for client in clients:
            digest = _digest(client.token.get_secret_value())
            if digest in index:
                # Two subjects behind one token means the audit trail cannot say
                # who acted, which defeats the reason approvals are recorded.
                raise ValueError(
                    f"API clients '{index[digest].subject}' and '{client.subject}' share a token"
                )
            index[digest] = client.principal
        self._by_digest = index

    @property
    def is_configured(self) -> bool:
        """False when no credentials exist, which means nothing can authenticate.

        Checked at startup so an operator is told why every call is a 401. The
        API stays closed either way: an unconfigured deployment authenticates
        nobody rather than everybody.
        """
        return bool(self._by_digest)

    @property
    def subjects(self) -> frozenset[str]:
        """The issued subjects. For startup logging; carries no secret."""
        return frozenset(principal.subject for principal in self._by_digest.values())

    def authenticate(self, presented_token: str) -> Principal | None:
        """Return the principal behind ``presented_token``, or None."""
        return self._by_digest.get(_digest(presented_token))
