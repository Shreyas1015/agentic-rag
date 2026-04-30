"""Documents list + delete.

The /ingest endpoint already handles uploads and dispatches Celery work; here
we expose the tenant's document inventory plus the soft-delete that flips both
the Postgres rows and the Qdrant payloads' is_active flag.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from qdrant_client.http import models as qmodels
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AuthInfo, require_auth
from app.core.qdrant_client import collection_name_for, get_async_qdrant_client
from app.db import crud
from app.db.models import Document
from app.db.session import get_session

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentOut(BaseModel):
    id: str
    filename: str
    doc_type: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    is_active: bool
    status: str  # queued|processing|ready|failed
    size_bytes: int | None = None
    created_at: datetime


class DocumentListOut(BaseModel):
    items: list[DocumentOut]


def _derive_status(doc: Document) -> str:
    """Status without the Celery roundtrip — we infer from DB state.

    `chunk_count` is set by the ingestion task on success; absence of it on an
    active row is taken as 'still processing'. Inactive rows are 'failed' only
    if they never got chunks; otherwise they were soft-deleted by the user.
    """
    if not doc.is_active:
        return "deleted"
    if doc.chunk_count and doc.chunk_count > 0:
        return "ready"
    return "processing"


def _size_of(doc: Document) -> int | None:
    if not doc.file_path:
        return None
    try:
        return os.path.getsize(doc.file_path)
    except OSError:
        return None


@router.get("", response_model=DocumentListOut, summary="List tenant documents")
async def list_documents(
    auth: AuthInfo = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> DocumentListOut:
    if not auth.organization_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No active org")

    rows = list(
        (
            await db.execute(
                select(Document)
                .where(Document.tenant_id == auth.organization_id)
                .order_by(desc(Document.created_at))
            )
        )
        .scalars()
        .all()
    )

    return DocumentListOut(
        items=[
            DocumentOut(
                id=str(d.id),
                filename=d.filename,
                doc_type=d.doc_type,
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                is_active=d.is_active,
                status=_derive_status(d),
                size_bytes=_size_of(d),
                created_at=d.created_at,
            )
            for d in rows
            if d.is_active  # hide soft-deleted rows from the UI
        ]
    )


@router.delete("/{document_id}", summary="Soft-delete a document")
async def delete_document(
    document_id: uuid.UUID,
    auth: AuthInfo = Depends(require_auth),
    db: AsyncSession = Depends(get_session),
) -> dict:
    if not auth.organization_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No active org")

    doc = (
        await db.execute(
            select(Document).where(
                Document.id == document_id,
                Document.tenant_id == auth.organization_id,
            )
        )
    ).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    await crud.mark_inactive(db, document_id=doc.id)
    await db.commit()

    # Flip Qdrant payloads to is_active=false for this document so retrieval
    # ignores them. We tolerate Qdrant errors (Postgres is the source of
    # truth for active state via the parent_chunks join).
    try:
        client = get_async_qdrant_client()
        collection = collection_name_for(auth.organization_id)
        await client.set_payload(
            collection_name=collection,
            payload={"is_active": False},
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=str(doc.id)),
                        )
                    ]
                )
            ),
        )
    except Exception:
        log.exception("Qdrant payload flip failed for %s (non-fatal)", doc.id)

    # And remove the file on disk; ignore failure (cleanup, not correctness).
    if doc.file_path:
        try:
            os.remove(doc.file_path)
        except OSError:
            pass

    return {"status": "deleted", "id": str(doc.id)}
