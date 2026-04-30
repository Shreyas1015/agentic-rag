"""Chat session + message persistence.

These endpoints back the chat history sidebar. The actual answer streaming
still happens in /chat/stream — the BFF posts the user turn here, opens the
SSE stream, and posts the assistant turn back here when it finishes.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthInfo, require_auth
from app.db.models import ChatMessage, ChatSession
from app.db.session import get_session

router = APIRouter(prefix="/sessions", tags=["sessions"])


# ── Schemas ──────────────────────────────────────────────────────────


class Citation(BaseModel):
    document_id: str | None = None
    filename: str | None = None
    page_num: int | None = None
    text: str | None = None
    score: float | None = None


class MessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] | None = None
    trace_id: str | None = None
    from_cache: bool = False
    context_score: float | None = None
    created_at: datetime


class SessionSummary(BaseModel):
    id: str
    title: str
    last_message_preview: str | None = None
    message_count: int
    updated_at: datetime


class SessionDetail(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]


class SessionListOut(BaseModel):
    items: list[SessionSummary]
    next_cursor: str | None = None


class CreateSessionIn(BaseModel):
    title: str | None = Field(default=None, max_length=512)


class UpdateSessionIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)


class CreateMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=200_000)
    citations: list[Citation] | None = None
    trace_id: str | None = Field(default=None, max_length=128)
    from_cache: bool = False
    context_score: float | None = None


# ── Helpers ──────────────────────────────────────────────────────────


def _require_org(auth: AuthInfo) -> str:
    if not auth.organization_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No active org")
    return auth.organization_id


async def _load_owned(
    db: AsyncSession, *, session_id: uuid.UUID, tenant_id: str
) -> ChatSession:
    row = (
        await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.tenant_id == tenant_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return row


def _to_summary(s: ChatSession, *, preview: str | None, count: int) -> SessionSummary:
    return SessionSummary(
        id=str(s.id),
        title=s.title,
        last_message_preview=preview,
        message_count=count,
        updated_at=s.updated_at,
    )


# ── List ─────────────────────────────────────────────────────────────


@router.get("", response_model=SessionListOut, summary="List chat sessions")
async def list_sessions(
    limit: int = Query(default=30, ge=1, le=100),
    cursor: str | None = Query(default=None),
    auth: AuthInfo = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> SessionListOut:
    tenant_id = _require_org(auth)

    stmt = (
        select(ChatSession)
        .where(ChatSession.tenant_id == tenant_id)
        .order_by(desc(ChatSession.updated_at))
        .limit(limit + 1)
    )
    if cursor:
        try:
            cursor_dt = datetime.fromisoformat(cursor)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Invalid cursor"
            ) from exc
        stmt = stmt.where(ChatSession.updated_at < cursor_dt)

    rows = list((await db.execute(stmt)).scalars().all())

    next_cursor = None
    if len(rows) > limit:
        next_cursor = rows[limit - 1].updated_at.isoformat()
        rows = rows[:limit]

    if not rows:
        return SessionListOut(items=[], next_cursor=None)

    ids = [r.id for r in rows]
    counts = dict(
        (
            await db.execute(
                select(ChatMessage.session_id, func.count(ChatMessage.id))
                .where(ChatMessage.session_id.in_(ids))
                .group_by(ChatMessage.session_id)
            )
        ).all()
    )

    last_msg_subq = (
        select(
            ChatMessage.session_id,
            func.max(ChatMessage.created_at).label("max_created"),
        )
        .where(ChatMessage.session_id.in_(ids))
        .group_by(ChatMessage.session_id)
        .subquery()
    )
    last_msgs_q = select(ChatMessage).join(
        last_msg_subq,
        (ChatMessage.session_id == last_msg_subq.c.session_id)
        & (ChatMessage.created_at == last_msg_subq.c.max_created),
    )
    previews: dict[uuid.UUID, str] = {
        m.session_id: (m.content[:140] + ("…" if len(m.content) > 140 else ""))
        for m in (await db.execute(last_msgs_q)).scalars().all()
    }

    return SessionListOut(
        items=[
            _to_summary(s, preview=previews.get(s.id), count=int(counts.get(s.id, 0)))
            for s in rows
        ],
        next_cursor=next_cursor,
    )


# ── Get one ──────────────────────────────────────────────────────────


@router.get(
    "/{session_id}", response_model=SessionDetail, summary="Get one session + messages"
)
async def get_one_session(
    session_id: uuid.UUID,
    auth: AuthInfo = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> SessionDetail:
    tenant_id = _require_org(auth)
    s = await _load_owned(db, session_id=session_id, tenant_id=tenant_id)

    msgs = list(
        (
            await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == s.id)
                .order_by(ChatMessage.created_at)
            )
        )
        .scalars()
        .all()
    )
    return SessionDetail(
        id=str(s.id),
        title=s.title,
        created_at=s.created_at,
        updated_at=s.updated_at,
        messages=[
            MessageOut(
                id=str(m.id),
                role=m.role,  # type: ignore[arg-type]
                content=m.content,
                citations=[Citation(**c) for c in (m.citations or [])] or None,
                trace_id=m.trace_id,
                from_cache=m.from_cache,
                context_score=m.context_score,
                created_at=m.created_at,
            )
            for m in msgs
        ],
    )


# ── Create ───────────────────────────────────────────────────────────


@router.post("", status_code=status.HTTP_201_CREATED, response_model=SessionSummary)
async def create_session(
    body: CreateSessionIn,
    auth: AuthInfo = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> SessionSummary:
    tenant_id = _require_org(auth)
    s = ChatSession(
        tenant_id=tenant_id,
        user_sub=auth.sub,
        title=(body.title or "New chat").strip()[:512],
    )
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return _to_summary(s, preview=None, count=0)


# ── Rename ───────────────────────────────────────────────────────────


@router.patch("/{session_id}", response_model=SessionSummary)
async def rename_session(
    session_id: uuid.UUID,
    body: UpdateSessionIn,
    auth: AuthInfo = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> SessionSummary:
    tenant_id = _require_org(auth)
    s = await _load_owned(db, session_id=session_id, tenant_id=tenant_id)
    s.title = body.title.strip()[:512]
    await db.commit()
    await db.refresh(s)
    count = (
        await db.execute(
            select(func.count(ChatMessage.id)).where(ChatMessage.session_id == s.id)
        )
    ).scalar_one()
    return _to_summary(s, preview=None, count=int(count))


# ── Delete ───────────────────────────────────────────────────────────


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID,
    auth: AuthInfo = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> None:
    tenant_id = _require_org(auth)
    s = await _load_owned(db, session_id=session_id, tenant_id=tenant_id)
    await db.delete(s)
    await db.commit()


# ── Append message ───────────────────────────────────────────────────


@router.post(
    "/{session_id}/messages",
    status_code=status.HTTP_201_CREATED,
    response_model=MessageOut,
)
async def append_message(
    session_id: uuid.UUID,
    body: CreateMessageIn,
    auth: AuthInfo = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> MessageOut:
    tenant_id = _require_org(auth)
    s = await _load_owned(db, session_id=session_id, tenant_id=tenant_id)

    m = ChatMessage(
        session_id=s.id,
        role=body.role,
        content=body.content,
        citations=[c.model_dump() for c in body.citations] if body.citations else None,
        trace_id=body.trace_id,
        from_cache=body.from_cache,
        context_score=body.context_score,
    )
    db.add(m)

    # Auto-title once: when the first user message lands and the title is the
    # default, use the first 60 chars of the message as the title.
    if body.role == "user" and s.title in {"", "New chat"}:
        s.title = body.content.strip().split("\n", 1)[0][:60] or "New chat"

    s.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(m)

    return MessageOut(
        id=str(m.id),
        role=m.role,  # type: ignore[arg-type]
        content=m.content,
        citations=[Citation(**c) for c in (m.citations or [])] or None,
        trace_id=m.trace_id,
        from_cache=m.from_cache,
        context_score=m.context_score,
        created_at=m.created_at,
    )
