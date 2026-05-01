"""Ingestion API.

POST /ingest                  multipart upload -> Celery task
GET  /ingest/status/{task_id} poll Celery AsyncResult for state + result
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

import logging

from app.core.auth import require_tenant
from app.core.celery_app import celery_app
from app.db import crud
from app.db.session import get_session
from app.ingestion.tasks import ingest_document
from app.storage.s3_client import upload_document

log = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

UPLOAD_DIR = Path("data/uploads")
ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB hard cap; tune later if needed


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF and dispatch async ingestion",
)
async def post_ingest(
    file: Annotated[UploadFile, File(...)],
    doc_type: Annotated[str | None, Form()] = None,
    tenant_id: str = Depends(require_tenant),
    session: AsyncSession = Depends(get_session),
) -> dict:
    # ── Validation ────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content_type {file.content_type!r}; expected application/pdf",
        )
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB cap",
        )

    # ── Dedup ─────────────────────────────────────────────
    content_hash = crud.compute_sha256(file_bytes)
    existing = await crud.dedup_check(session, content_hash)
    if existing and existing.is_active:
        # Already-ingested. Don't re-dispatch; return what we have so the
        # caller can move on. (Re-ingest of inactive documents is a separate
        # flow we'll add in Phase 2.)
        return {
            "status": "already_ingested",
            "document_id": str(existing.id),
            "filename": existing.filename,
            "tenant_id": existing.tenant_id,
        }

    # ── Persist Document row ──────────────────────────────
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Tentatively insert with placeholder file_path so we can use the assigned
    # UUID in the on-disk filename, then update.
    doc = await crud.insert_document(
        session,
        tenant_id=tenant_id,
        filename=file.filename or "upload.pdf",
        content_hash=content_hash,
        file_path="",  # set immediately below
        doc_type=doc_type,
    )
    file_path = UPLOAD_DIR / f"{doc.id}.pdf"
    file_path.write_bytes(file_bytes)
    doc.file_path = str(file_path)
    await session.commit()
    await session.refresh(doc)

    # ── Mirror upload to S3/MinIO so the browser can preview/download ──
    # via presigned URLs. Worker still reads from disk for parsing — we
    # keep both copies for now to avoid changing the worker contract.
    # If S3 is misconfigured we log + continue: the doc is still ingestible,
    # just not previewable. Re-upload from /documents/{id}/repair would
    # heal it (out of scope here).
    try:
        await upload_document(tenant_id, str(doc.id), file_bytes)
    except Exception:
        log.exception("S3 upload failed for %s — preview link will 404", doc.id)

    # ── Dispatch Celery task ──────────────────────────────
    async_result = ingest_document.apply_async(
        args=[str(file_path), tenant_id, str(doc.id), doc_type]
    )

    return {
        "status": "queued",
        "task_id": async_result.id,
        "document_id": str(doc.id),
        "filename": doc.filename,
    }


@router.get(
    "/status/{task_id}",
    summary="Poll a Celery ingestion task by id",
)
async def get_ingest_status(
    task_id: str,
    _tenant_id: str = Depends(require_tenant),
) -> dict:
    """Returns the Celery task state + result/error for the given task_id.

    NOTE Phase-2 hardening: this endpoint currently doesn't bind a task_id to
    a tenant, so any authenticated tenant can poll any task_id (low risk —
    task_ids are unguessable UUIDs). To tighten, store task_id on the
    Document row and verify ownership before exposing state.
    """
    result = AsyncResult(task_id, app=celery_app)
    state = (result.state or "PENDING").lower()
    response: dict = {"task_id": task_id, "state": state}
    if result.successful():
        response["result"] = result.result
    elif result.failed():
        response["error"] = str(result.result)
    return response
