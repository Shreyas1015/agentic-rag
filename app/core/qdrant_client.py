"""Qdrant client singletons (sync + async).

The sync client is used by ingestion (Celery task is sync), the async one
by the FastAPI request path. Both share the same host/port from settings.
"""

from functools import lru_cache

from qdrant_client import AsyncQdrantClient, QdrantClient

from app.core.config import settings


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


@lru_cache(maxsize=1)
def get_async_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


def collection_name_for(tenant_id: str) -> str:
    """One Qdrant collection per tenant. Naming: tenant_<tenant_id>."""
    return f"tenant_{tenant_id}"
