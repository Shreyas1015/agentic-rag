"""Celery tasks for the ingestion pipeline.

Right now this only contains a `ping` no-op task used to smoke-test the worker
+ broker + result backend. Step B replaces it with `ingest_document` (Docling
parse → chunk → embed → upsert).
"""

from app.core.celery_app import celery_app


@celery_app.task(name="ingestion.ping")
def ping(message: str = "pong") -> dict:
    """Diagnostic — round-trips a payload through the broker and backend."""
    return {"ok": True, "echo": message}
