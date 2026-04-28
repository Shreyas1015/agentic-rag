"""Build Qdrant points from EmbeddedChunks and upsert them, batched.

One point per child chunk. Named vectors carry both the dense and the BM25
sparse representation so we can hybrid-search with `Prefetch + Fusion.RRF`
(see app/retrieval/hybrid_search.py — Step D).

Payload schema (mirrored across all tenants):
  tenant_id    : str   - REQUIRED filter on every query
  document_id  : str   - UUID-as-string of the Document row
  parent_id    : str   - join key back to PostgreSQL parent_chunks.parent_id
  doc_type     : str?  - optional, free-form (e.g. "credit_policy")
  page_num     : int   - 1-indexed source page (for citations)
  chunk_index  : int   - ordinal within the document
  is_active    : bool  - flipped to false on soft-delete / re-ingest
  text_preview : str   - first 200 chars, useful in Qdrant dashboard / debug
"""

from __future__ import annotations

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
)

from app.core.qdrant_client import get_qdrant_client
from app.ingestion.embedder import EmbeddedChunk

UPSERT_BATCH_SIZE = 256


def build_point(
    embedded: EmbeddedChunk,
    *,
    tenant_id: str,
    document_id: str,
    doc_type: str | None,
) -> PointStruct:
    return PointStruct(
        id=embedded.child.chunk_id,
        vector={
            "dense": embedded.dense,
            "bm25": SparseVector(
                indices=embedded.sparse_indices,
                values=embedded.sparse_values,
            ),
        },
        payload={
            "tenant_id": tenant_id,
            "document_id": document_id,
            "parent_id": embedded.child.parent_id,
            "doc_type": doc_type,
            "page_num": embedded.child.page_num,
            "chunk_index": embedded.child.chunk_index,
            "is_active": True,
            "text_preview": embedded.child.text[:200],
        },
    )


def upsert_to_qdrant(
    collection_name: str,
    points: list[PointStruct],
    *,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> int:
    """Upsert points in fixed-size batches. Returns total upserted."""
    if not points:
        return 0
    client = get_qdrant_client()
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + batch_size],
            wait=True,
        )
    return len(points)


def count_active_for_document(collection_name: str, document_id: str) -> int:
    """Exact count of active points for a document — used to validate that
    the upsert wrote what we expected."""
    client = get_qdrant_client()
    result = client.count(
        collection_name=collection_name,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="document_id", match=MatchValue(value=document_id)
                ),
                FieldCondition(key="is_active", match=MatchValue(value=True)),
            ]
        ),
        exact=True,
    )
    return result.count


def deactivate_document_points(collection_name: str, document_id: str) -> int:
    """Flip all points for a document to is_active=false (soft-delete).

    Used by re-ingest (so old chunks don't surface alongside new ones) and by
    the /documents DELETE endpoint."""
    client = get_qdrant_client()
    client.set_payload(
        collection_name=collection_name,
        payload={"is_active": False},
        points=Filter(
            must=[
                FieldCondition(
                    key="document_id", match=MatchValue(value=document_id)
                )
            ]
        ),
        wait=True,
    )
    # Return the new count post-deactivation for caller verification.
    return count_active_for_document(collection_name, document_id)
