"""Celery application bootstrap.

Run the worker with:
    uv run celery -A app.core.celery_app worker --loglevel=info

On Windows, Celery's default prefork pool doesn't work — pass --pool=solo
(single-threaded, fine for dev) or --pool=threads for local iteration:
    uv run celery -A app.core.celery_app worker --loglevel=info --pool=solo

In Docker (Linux), prefork is the default and what we want in production.

Broker:  Redis DB 0  (CELERY_BROKER_URL)
Backend: Redis DB 1  (CELERY_RESULT_BACKEND) — semantic cache lives in DB 2
Include: app.ingestion.tasks  (auto-discovered on worker start)
"""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "agentic_rag",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.ingestion.tasks"],
)

celery_app.conf.update(
    # ── Serialization ──────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # ── Reliability ────────────────────────────────
    # ack tasks AFTER they finish so a worker crash mid-task re-queues the work.
    task_acks_late=True,
    # one task at a time per worker child — ingestion tasks load big models /
    # PDFs; fair scheduling beats throughput here.
    worker_prefetch_multiplier=1,
    # don't lose results, but expire them so Redis doesn't grow forever.
    result_expires=60 * 60 * 24,  # 24h
    # if the worker dies after `max_retries`, surface the error rather than
    # silently swallowing it.
    task_reject_on_worker_lost=True,
    # ── Visibility ─────────────────────────────────
    task_track_started=True,
)
