"""Celery tasks for the ingestion pipeline.

`ingest_document` is dispatched by the /ingest endpoint after it:
  - validates the upload,
  - computes SHA256 + dedup-checks,
  - writes the Document row (so we have document.id),
  - saves raw bytes to data/uploads/<document_id>.pdf,
  - dispatches this task with (file_path, tenant_id, document_id, doc_type).

The task itself owns: parse -> chunk -> embed -> write parent_chunks -> upsert
to Qdrant -> validate counts. Failures retry up to 3 times with backoff.

Celery worker is sync; we call our async helpers via asyncio.run(...). On
prefork (Linux/Docker) each child process gets its own loop; on Windows we
recommend --pool=solo for dev (see app/core/celery_app.py module docstring).
"""

from __future__ import annotations

import asyncio
import uuid

from app.core.celery_app import celery_app
from app.core.qdrant_client import collection_name_for
from app.db import crud
from app.db.session import async_session_maker
from app.ingestion.chunker import hierarchical_chunk
from app.ingestion.embedder import embed_chunks
from app.ingestion.parser import parse_pdf
from app.ingestion.upserter import (
    build_point,
    count_active_for_document,
    upsert_to_qdrant,
)


@celery_app.task(name="ingestion.ping")
def ping(message: str = "pong") -> dict:
    """Diagnostic — kept around for worker liveness checks."""
    return {"ok": True, "echo": message}


@celery_app.task(
    name="ingestion.ingest_document",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
)
def ingest_document(
    self,
    file_path: str,
    tenant_id: str,
    document_id: str,
    doc_type: str | None = None,
) -> dict:
    """Run the full ingestion pipeline for a single Document row.

    All heavy lifting happens inside `_ingest_async`; this wrapper is sync so
    Celery's retry / acks-late machinery sees a normal function.
    """
    return asyncio.run(
        _ingest_async(file_path, tenant_id, document_id, doc_type)
    )


async def _ingest_async(
    file_path: str,
    tenant_id: str,
    document_id: str,
    doc_type: str | None,
) -> dict:
    # 1. Parse PDF (Docling -> per-page text + tables-as-markdown).
    pages = parse_pdf(file_path)

    # 2. Chunk hierarchically (1024-tok parents, 256-tok children).
    parents, children = hierarchical_chunk(pages)
    if not children:
        # Nothing to embed/upsert; still record counts so the doc isn't stuck.
        async with async_session_maker() as session:
            await crud.update_document_counts(
                session, uuid.UUID(document_id), page_count=len(pages), chunk_count=0
            )
            await session.commit()
        return {"pages": len(pages), "parents": 0, "children": 0, "qdrant": 0}

    # 3. Embed children (dense via OpenRouter, sparse via FastEmbed BM25).
    embedded = await embed_chunks(children)

    # 4. Write parent_chunks + counts to PostgreSQL.
    async with async_session_maker() as session:
        await crud.bulk_insert_parent_chunks(
            session, parents, document_id=uuid.UUID(document_id), tenant_id=tenant_id
        )
        await crud.update_document_counts(
            session,
            uuid.UUID(document_id),
            page_count=len(pages),
            chunk_count=len(children),
        )
        await session.commit()

    # 5. Upsert children to Qdrant.
    collection = collection_name_for(tenant_id)
    points = [
        build_point(e, tenant_id=tenant_id, document_id=document_id, doc_type=doc_type)
        for e in embedded
    ]
    upserted = upsert_to_qdrant(collection, points)

    # 6. Validate: Qdrant active-point count for this doc must equal child count.
    actual = count_active_for_document(collection, document_id)
    if actual != len(children):
        # Surface to Celery as an error so the task retries.
        raise RuntimeError(
            f"Post-upsert count mismatch for {document_id}: expected={len(children)} got={actual}"
        )

    return {
        "pages": len(pages),
        "parents": len(parents),
        "children": len(children),
        "qdrant": actual,
        "upserted": upserted,
    }
