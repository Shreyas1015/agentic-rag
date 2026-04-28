"""Logto access-token verification for FastAPI.

We follow Logto's "Organization-level API resources" permission model
(see https://docs.logto.io/authorization). Concretely the access tokens we
accept must carry:

  iss              = {LOGTO_ENDPOINT}/oidc
  aud              = LOGTO_RESOURCE                  (the API resource indicator)
  organization_id  = the tenant's Logto org id       (required by require_tenant)
  scope            = space-separated permissions     (checked by require_scopes)

Signature verification uses PyJWT against Logto's JWKS endpoint
({LOGTO_ENDPOINT}/oidc/jwks). Logto OSS signs with ES384 by default;
ES256 / RS256 are kept in the algorithms list as forward compatibility.

`require_auth`     -> verifies + parses the token, returns AuthInfo
`require_tenant`   -> AuthInfo whose organization_id is non-empty (->tenant_id)
`require_scopes`   -> dependency factory; checks AuthInfo.scopes contains all
                      of the listed scopes (Phase 2 RBAC plumbing).
"""

from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from pydantic import BaseModel

from app.core.config import settings

_security = HTTPBearer(auto_error=True)


class AuthInfo(BaseModel):
    sub: str
    client_id: str | None = None
    organization_id: str | None = None
    scopes: list[str] = []
    audience: list[str] = []


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient:
    if not settings.LOGTO_ENDPOINT:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth misconfigured: LOGTO_ENDPOINT not set.",
        )
    return PyJWKClient(settings.logto_jwks_uri, cache_keys=True, lifespan=300)


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> AuthInfo:
    if not settings.LOGTO_RESOURCE:
        # Refuse to validate at all if the audience is unset — silently skipping
        # the aud check would let tokens minted for a different API into ours.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth misconfigured: LOGTO_RESOURCE not set.",
        )
    token = credentials.credentials
    try:
        signing_key = _jwks_client().get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            # Logto OSS signs with ES384 by default; ES256/RS256 kept for
            # tenants that have rotated/changed signing keys.
            algorithms=["ES384", "ES256", "RS256"],
            audience=settings.LOGTO_RESOURCE,
            issuer=settings.logto_issuer,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        ) from exc

    audience = payload.get("aud") or []
    if isinstance(audience, str):
        audience = [audience]
    scope_claim = payload.get("scope") or ""
    return AuthInfo(
        sub=payload["sub"],
        client_id=payload.get("client_id"),
        organization_id=payload.get("organization_id"),
        scopes=scope_claim.split() if scope_claim else [],
        audience=audience,
    )


async def require_tenant(auth: AuthInfo = Depends(require_auth)) -> str:
    """Extracts tenant_id from the Logto `organization_id` claim.

    Tokens minted without an organization context are rejected — every
    protected endpoint in this system operates on a specific tenant.
    """
    if not auth.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token has no organization context (organization_id claim missing).",
        )
    return auth.organization_id


def require_scopes(*needed: str):
    """Dependency factory: returns a FastAPI dep that 403s unless the token
    carries every scope in `needed`.

    Usage:
        @router.post("/ingest", dependencies=[Depends(require_scopes("ingest:write"))])

    We don't enforce scopes anywhere yet (Phase 1 only checks
    authentication + tenant). This stub locks the API contract for Phase 2 RBAC.
    Roles -> scopes mapping is configured in the Logto admin console.
    """

    async def _checker(auth: AuthInfo = Depends(require_auth)) -> AuthInfo:
        missing = [s for s in needed if s not in auth.scopes]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope(s): {', '.join(missing)}",
            )
        return auth

    return _checker
