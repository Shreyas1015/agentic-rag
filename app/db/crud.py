"""Async CRUD helpers used by ingestion + (later) the API layer.

All postgres access goes through SQLAlchemy AsyncSession (project rule). The
ingestion path uses these in this order:

    sha = compute_sha256(file_bytes)
    if existing := await dedup_check(session, sha): ...
    doc = await insert_document(session, ...)              # commit -> get doc.id
    # ... celery task does the parse / chunk / embed / upsert ...
    await bulk_insert_parent_chunks(session, parents, ...)
    await update_document_counts(session, doc.id, page_count=N, chunk_count=M)
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, ParentChunk
from app.ingestion.chunker import ParentNode


def compute_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


async def dedup_check(
    session: AsyncSession, content_hash: str
) -> Document | None:
    """Return the existing Document row for this hash, or None."""
    stmt = select(Document).where(Document.content_hash == content_hash)
    return (await session.execute(stmt)).scalar_one_or_none()


async def insert_document(
    session: AsyncSession,
    *,
    tenant_id: str,
    filename: str,
    content_hash: str,
    file_path: str,
    doc_type: str | None = None,
    page_count: int | None = None,
    chunk_count: int | None = None,
    source_date: date | None = None,
) -> Document:
    """Insert a fresh Document row, returns it with .id populated (after flush)."""
    doc = Document(
        tenant_id=tenant_id,
        filename=filename,
        content_hash=content_hash,
        file_path=file_path,
        doc_type=doc_type,
        page_count=page_count,
        chunk_count=chunk_count,
        source_date=source_date,
        version=1,
        is_active=True,
    )
    session.add(doc)
    await session.flush()  # populates doc.id from the DB
    return doc


async def update_document_counts(
    session: AsyncSession,
    document_id: uuid.UUID,
    *,
    page_count: int | None = None,
    chunk_count: int | None = None,
) -> None:
    values: dict = {}
    if page_count is not None:
        values["page_count"] = page_count
    if chunk_count is not None:
        values["chunk_count"] = chunk_count
    if not values:
        return
    await session.execute(
        update(Document).where(Document.id == document_id).values(**values)
    )


async def bulk_insert_parent_chunks(
    session: AsyncSession,
    parents: list[ParentNode],
    *,
    document_id: uuid.UUID,
    tenant_id: str,
) -> int:
    """Persist a batch of 1024-token parent chunks. Returns row count."""
    rows = [
        ParentChunk(
            parent_id=p.parent_id,
            document_id=document_id,
            tenant_id=tenant_id,
            text=p.text,
            page_num=p.page_num,
            chunk_index=p.chunk_index,
            is_active=True,
        )
        for p in parents
    ]
    if not rows:
        return 0
    session.add_all(rows)
    await session.flush()
    return len(rows)


async def fetch_parent_chunks(
    session: AsyncSession,
    *,
    tenant_id: str,
    parent_ids: list[str],
) -> list[ParentChunk]:
    """Bulk fetch active parent chunks by id (used by the agent's parent_fetch
    node after Qdrant returns hits)."""
    if not parent_ids:
        return []
    stmt = select(ParentChunk).where(
        ParentChunk.tenant_id == tenant_id,
        ParentChunk.is_active.is_(True),
        ParentChunk.parent_id.in_(parent_ids),
    )
    return list((await session.execute(stmt)).scalars().all())


async def estimate_tenant_token_count(
    session: AsyncSession, *, tenant_id: str
) -> int:
    """Rough token estimate (~4 chars/token, English-ish) for ALL active
    parent chunks belonging to a tenant. Used by the long-context-bypass
    decision in the agent — if the whole corpus fits in `gemini-2.5-pro`'s
    1M-token window we skip RAG and stuff everything into the prompt.
    """
    stmt = select(func.coalesce(func.sum(func.length(ParentChunk.text)), 0)).where(
        ParentChunk.tenant_id == tenant_id,
        ParentChunk.is_active.is_(True),
    )
    total_chars = (await session.execute(stmt)).scalar_one() or 0
    return int(int(total_chars) / 4)


async def fetch_all_active_parent_chunks(
    session: AsyncSession, *, tenant_id: str
) -> list[ParentChunk]:
    """Return every active parent chunk for a tenant in deterministic order.
    Used by the long-context-bypass node — it stuffs the lot into the
    long-context model's prompt."""
    stmt = (
        select(ParentChunk)
        .where(
            ParentChunk.tenant_id == tenant_id,
            ParentChunk.is_active.is_(True),
        )
        .order_by(ParentChunk.document_id, ParentChunk.chunk_index)
    )
    return list((await session.execute(stmt)).scalars().all())


async def fetch_document_filenames(
    session: AsyncSession,
    *,
    tenant_id: str,
    document_ids: list[str],
) -> dict[str, str]:
    """Bulk lookup `document_id (UUID-as-str) -> filename` for citation rendering."""
    if not document_ids:
        return {}
    uuids = [uuid.UUID(d) for d in document_ids]
    stmt = select(Document.id, Document.filename).where(
        Document.tenant_id == tenant_id,
        Document.id.in_(uuids),
    )
    rows = (await session.execute(stmt)).all()
    return {str(rid): name for rid, name in rows}


async def mark_inactive(
    session: AsyncSession, document_id: uuid.UUID
) -> None:
    """Soft-delete a document and all its parent chunks. Qdrant's points are
    flipped to is_active=false separately by the caller."""
    await session.execute(
        update(Document).where(Document.id == document_id).values(is_active=False)
    )
    await session.execute(
        update(ParentChunk)
        .where(ParentChunk.document_id == document_id)
        .values(is_active=False)
    )
