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
    name="observability.score_and_log",
    bind=True,
    max_retries=2,
    default_retry_delay=15,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    ignore_result=True,
)
def score_and_log(
    self,
    *,
    trace_id: str | None,
    tenant_id: str,
    query: str,
    answer: str,
    contexts: list[str],
    context_score: float | None = None,
    ground_truth: str | None = None,
) -> dict:
    """Run RAGAS on a completed (query, answer, contexts) triple, write a
    row into eval_logs, and push the scores back onto the Langfuse trace.

    Fires fire-and-forget from /chat/stream after the `done` event lands.
    Failures retry twice but never block the user — if eval is broken,
    the user already has their answer.
    """
    return asyncio.run(
        _score_and_log_async(
            trace_id=trace_id,
            tenant_id=tenant_id,
            query=query,
            answer=answer,
            contexts=contexts,
            context_score=context_score,
            ground_truth=ground_truth,
        )
    )


async def _score_and_log_async(
    *,
    trace_id: str | None,
    tenant_id: str,
    query: str,
    answer: str,
    contexts: list[str],
    context_score: float | None,
    ground_truth: str | None,
) -> dict:
    # Lazy imports to avoid pulling RAGAS into the api hot path.
    from app.db.models import EvalLog
    from app.observability.langfuse_client import langfuse
    from app.observability.ragas_eval import score as ragas_score

    scores = await ragas_score(query, answer, contexts, ground_truth=ground_truth)

    # Persist to eval_logs.
    async with async_session_maker() as session:
        row = EvalLog(
            trace_id=trace_id,
            tenant_id=tenant_id,
            query=query,
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("answer_relevancy"),
            context_precision=scores.get("context_precision"),
            context_recall=scores.get("context_recall"),
            context_score=context_score,
        )
        session.add(row)
        await session.commit()

    # Push each non-null score back to the Langfuse trace.
    if trace_id:
        try:
            for name, val in scores.items():
                if val is None:
                    continue
                langfuse.create_score(name=f"ragas.{name}", value=val, trace_id=trace_id)
            langfuse.flush()
        except Exception:
            # Never let observability failures take down the task.
            pass

    return scores


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
