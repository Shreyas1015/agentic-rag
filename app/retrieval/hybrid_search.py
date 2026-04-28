"""Qdrant hybrid search — dense + BM25 fused via Reciprocal Rank Fusion.

Both vectors are computed from the user's query at request time (dense via
OpenRouter, sparse via local FastEmbed). Qdrant runs the two prefetches in
parallel and fuses their rankings with RRF; we filter by `tenant_id` AND
`is_active=true` on every prefetch so soft-deleted / cross-tenant chunks
can never leak.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from app.core.qdrant_client import collection_name_for, get_async_qdrant_client
from app.ingestion.embedder import embed_query_dense, embed_query_sparse

DEFAULT_TOP_K = 30


@dataclass
class RetrievedChunk:
    chunk_id: str  # Qdrant point id (= LlamaIndex node id of the child)
    score: float  # RRF-fused score
    parent_id: str
    document_id: str
    page_num: int
    chunk_index: int
    doc_type: str | None
    text_preview: str


def _tenant_filter(tenant_id: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
            FieldCondition(key="is_active", match=MatchValue(value=True)),
        ]
    )


async def hybrid_search(
    query_text: str,
    *,
    tenant_id: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """Return the top-K children for `query_text` within `tenant_id`."""
    if not query_text.strip():
        return []

    # 1) Embed query — dense via OpenRouter, sparse via local BM25.
    dense_vec = await embed_query_dense(query_text)
    sparse_indices, sparse_values = embed_query_sparse(query_text)
    sparse_vec = SparseVector(indices=sparse_indices, values=sparse_values)

    # 2) Two filtered prefetches (each pulls top_k); Qdrant fuses with RRF.
    flt = _tenant_filter(tenant_id)
    collection = collection_name_for(tenant_id)
    qdrant = get_async_qdrant_client()
    response = await qdrant.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", filter=flt, limit=top_k),
            Prefetch(query=sparse_vec, using="bm25", filter=flt, limit=top_k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    # 3) Adapt Qdrant ScoredPoint → our typed dataclass.
    out: list[RetrievedChunk] = []
    for p in response.points:
        payload = p.payload or {}
        out.append(
            RetrievedChunk(
                chunk_id=str(p.id),
                score=float(p.score),
                parent_id=payload.get("parent_id", ""),
                document_id=payload.get("document_id", ""),
                page_num=int(payload.get("page_num", 0)),
                chunk_index=int(payload.get("chunk_index", 0)),
                doc_type=payload.get("doc_type"),
                text_preview=payload.get("text_preview", ""),
            )
        )
    return out
