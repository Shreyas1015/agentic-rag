"""GET /me — profile + tenant block + first_login flag.

Calls Logto's `/oidc/me` userinfo endpoint with the inbound bearer to fetch
profile claims (Logto strips most claims from access tokens by default), then
reads the tenant row + does a quick ownership check to compute `first_login`
(no docs AND no sessions yet → tell the UI to send the user to onboarding).
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthInfo, require_auth
from app.core.config import settings
from app.db.models import ChatSession, Document, Tenant
from app.db.session import get_session

log = logging.getLogger(__name__)
router = APIRouter(tags=["me"])
_security = HTTPBearer(auto_error=True)


async def _userinfo(token: str) -> dict:
    base = settings.LOGTO_INTERNAL_ENDPOINT or settings.LOGTO_ENDPOINT
    if not base:
        return {}
    url = f"{base.rstrip('/')}/oidc/me"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url, headers={"Authorization": f"Bearer {token}"})
            if r.status_code == 200:
                return r.json()
    except Exception:
        log.exception("logto userinfo fetch failed (non-fatal)")
    return {}


async def _ensure_tenant(
    session: AsyncSession, *, tenant_id: str, fallback_name: str | None
) -> Tenant:
    row = (
        await session.execute(select(Tenant).where(Tenant.tenant_id == tenant_id))
    ).scalar_one_or_none()
    if row is not None:
        return row
    row = Tenant(
        tenant_id=tenant_id,
        name=fallback_name,
        qdrant_collection=f"tenant_{tenant_id}",
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


@router.get("/me", summary="Current user, organisation, and onboarding flag")
async def get_me(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
    auth: AuthInfo = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> dict:
    if not auth.organization_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Token missing organization_id (no active org).",
        )

    profile = await _userinfo(credentials.credentials)
    tenant = await _ensure_tenant(
        db,
        tenant_id=auth.organization_id,
        fallback_name=profile.get("organization_name"),
    )

    doc_count = (
        await db.execute(
            select(func.count(Document.id)).where(
                Document.tenant_id == auth.organization_id
            )
        )
    ).scalar_one()
    sess_count = (
        await db.execute(
            select(func.count(ChatSession.id)).where(
                ChatSession.tenant_id == auth.organization_id
            )
        )
    ).scalar_one()

    return {
        "user": {
            "sub": auth.sub,
            "email": profile.get("email"),
            "name": profile.get("name") or profile.get("username"),
            "picture": profile.get("picture") or profile.get("avatar"),
        },
        "org": {
            "id": tenant.tenant_id,
            "name": tenant.name,
            "qdrant_collection": tenant.qdrant_collection,
        },
        "scopes": auth.scopes,
        "first_login": (doc_count == 0) and (sess_count == 0),
    }
