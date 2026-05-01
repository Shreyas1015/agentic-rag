"""Qdrant client singletons (sync + async).

The sync client is used by ingestion (Celery task is sync), the async one
by the FastAPI request path. Both share the same host/port from settings.
"""

import logging
from functools import lru_cache

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


@lru_cache(maxsize=1)
def get_async_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


def collection_name_for(tenant_id: str) -> str:
    """One Qdrant collection per tenant. Naming: tenant_<tenant_id>."""
    return f"tenant_{tenant_id}"


def ensure_tenant_collection(tenant_id: str) -> str:
    """Create the tenant's Qdrant collection if it doesn't exist. Idempotent.

    Used at the top of the ingest task so a missing-collection error never
    causes the Celery task to retry-then-die with orphaned Postgres rows.
    Mirrors the layout produced by scripts/create_collections.py.
    """
    client = get_qdrant_client()
    name = collection_name_for(tenant_id)
    if client.collection_exists(name):
        return name

    log.info("Auto-creating Qdrant collection %s on first ingest", name)
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=settings.EMBEDDING_DIMS, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(
                modifier=Modifier.IDF,
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    for field, schema in [
        ("tenant_id", PayloadSchemaType.KEYWORD),
        ("is_active", PayloadSchemaType.BOOL),
        ("document_id", PayloadSchemaType.KEYWORD),
        ("parent_id", PayloadSchemaType.KEYWORD),
        ("doc_type", PayloadSchemaType.KEYWORD),
        ("page_num", PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(name, field_name=field, field_schema=schema)
    return name
