"""Feedback API — POST /feedback.

Closes the loop on /chat/stream: the user's UI captures the `trace_id`
from the SSE `done` event, lets the user 👍 / 👎 the answer, and posts
{trace_id, rating, comment?} back here. We:

  1. Update `eval_logs.user_feedback` for that trace (tenant-scoped so
     no one can write feedback to another tenant's row).
  2. `langfuse.create_score(name="user_feedback", value=±1)` so the
     score is visible alongside the trace tree in Langfuse.

Race: a fresh /chat/stream fires score_and_log async, so the eval_logs
row may not exist yet when /feedback arrives. We tolerate that — store
the rating in a stub row keyed by trace_id, and the RAGAS task's later
INSERT will see the existing row and merge in (we use ON CONFLICT
semantics via SQLAlchemy upsert). For Phase 2 we keep it simpler:
return 404 if the row isn't there yet, the UI retries.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import require_tenant
from app.db.models import EvalLog
from app.db.session import get_session
from app.observability.langfuse_client import langfuse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=1, max_length=128)
    rating: Literal[-1, 1] = Field(
        ..., description="1 = thumbs up, -1 = thumbs down"
    )
    comment: str | None = Field(default=None, max_length=2000)


@router.post(
    "",
    summary="Submit thumbs up/down for an answered query",
)
async def post_feedback(
    body: FeedbackRequest,
    tenant_id: str = Depends(require_tenant),
    session: AsyncSession = Depends(get_session),
) -> dict:
    # Tenant-scoped lookup — a token for tenant A can never write feedback
    # against tenant B's eval_logs row, even with a guessed trace_id.
    stmt = select(EvalLog).where(
        EvalLog.trace_id == body.trace_id,
        EvalLog.tenant_id == tenant_id,
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No eval_logs row for this trace_id (yet). RAGAS scoring is "
                "async; retry after a few seconds."
            ),
        )

    await session.execute(
        update(EvalLog)
        .where(EvalLog.id == row.id)
        .values(user_feedback=body.rating)
    )
    await session.commit()

    # Push to Langfuse — tolerate failures (the DB is the source of truth).
    try:
        langfuse.create_score(
            name="user_feedback",
            value=int(body.rating),
            trace_id=body.trace_id,
            comment=body.comment,
        )
        langfuse.flush()
    except Exception:
        log.exception("langfuse score push failed (non-fatal)")

    return {
        "status": "recorded",
        "trace_id": body.trace_id,
        "rating": body.rating,
    }
