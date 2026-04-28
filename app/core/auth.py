"""Logto access-token verification for FastAPI.

Verifies tokens against Logto's JWKS endpoint using PyJWT. The tenant_id
for the request is the `organization_id` claim Logto adds when a token
is minted for a specific organization.
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
    token = credentials.credentials
    try:
        signing_key = _jwks_client().get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=settings.LOGTO_RESOURCE or None,
            issuer=settings.logto_issuer,
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
