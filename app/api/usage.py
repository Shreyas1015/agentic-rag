"""GET /usage — aggregate metrics for the dashboard tiles."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthInfo, require_auth
from app.db.models import ChatSession, Document, EvalLog, ParentChunk
from app.db.session import get_session

router = APIRouter(tags=["usage"])


@router.get("/usage", summary="Aggregate metrics for the active tenant")
async def get_usage(
    auth: AuthInfo = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> dict:
    if not auth.organization_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No active org")
    tenant_id = auth.organization_id

    docs = list(
        (
            await db.execute(
                select(Document).where(Document.tenant_id == tenant_id)
            )
        )
        .scalars()
        .all()
    )
    total_docs = len(docs)
    active_docs = sum(1 for d in docs if d.is_active)
    ingesting = sum(
        1 for d in docs if d.is_active and (not d.chunk_count or d.chunk_count == 0)
    )
    storage_bytes = 0
    for d in docs:
        if d.file_path:
            try:
                storage_bytes += os.path.getsize(d.file_path)
            except OSError:
                pass

    chunk_total = (
        await db.execute(
            select(func.count(ParentChunk.id)).where(
                ParentChunk.tenant_id == tenant_id,
                ParentChunk.is_active.is_(True),
            )
        )
    ).scalar_one()

    now = datetime.now(timezone.utc)
    cutoff_30d = now - timedelta(days=30)
    cutoff_today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    queries_30d = (
        await db.execute(
            select(func.count(EvalLog.id)).where(
                EvalLog.tenant_id == tenant_id,
                EvalLog.created_at >= cutoff_30d,
            )
        )
    ).scalar_one()
    queries_today = (
        await db.execute(
            select(func.count(EvalLog.id)).where(
                EvalLog.tenant_id == tenant_id,
                EvalLog.created_at >= cutoff_today,
            )
        )
    ).scalar_one()

    sessions_total = (
        await db.execute(
            select(func.count(ChatSession.id)).where(
                ChatSession.tenant_id == tenant_id
            )
        )
    ).scalar_one()

    return {
        "documents": {
            "total": int(total_docs),
            "active": int(active_docs),
            "ingesting": int(ingesting),
        },
        "chunks": {"total": int(chunk_total)},
        "queries": {
            "last_30d": int(queries_30d),
            "today": int(queries_today),
        },
        "sessions": {"total": int(sessions_total)},
        "storage": {"bytes": int(storage_bytes)},
    }
