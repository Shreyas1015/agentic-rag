"""Tests for the retrieval layer: hybrid search + semantic cache."""

from __future__ import annotations

import uuid

import pytest

from app.core.config import settings
from app.ingestion.embedder import embed_query_dense
from app.retrieval.cache import (
    check_cache,
    invalidate_tenant_cache,
    write_cache,
)
from app.retrieval.hybrid_search import RetrievedChunk, hybrid_search

# ── Hybrid search ────────────────────────────────────────────────


@pytest.mark.integration
async def test_hybrid_search_returns_results_for_known_query(ingested_doc):
    """Search for content known to be in the fixture PDF — must return hits."""
    hits = await hybrid_search(
        "AMD EPYC CPU inference timing",
        tenant_id=ingested_doc["tenant_id"],
        top_k=10,
    )
    assert hits, "expected hits for a query that mentions content in the PDF"
    assert all(isinstance(h, RetrievedChunk) for h in hits)
    # Hits must come from our test doc.
    doc_ids = {h.document_id for h in hits}
    assert ingested_doc["document_id"] in doc_ids
    # Scores monotonically descending.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
@pytest.mark.usefixtures("ingested_doc")
async def test_hybrid_search_filters_by_tenant():
    """A tenant_id we never seeded must return zero hits — no cross-tenant
    leakage. The `ingested_doc` fixture (applied via usefixtures) is the
    precondition: data EXISTS for the test tenant, so a zero-hit result on
    a different tenant proves the filter is doing the work (not just an
    empty index)."""
    bogus_tenant = f"never-existed-{uuid.uuid4().hex[:8]}"
    hits = await hybrid_search("anything", tenant_id=bogus_tenant, top_k=10)
    assert hits == [], (
        f"expected no hits for unknown tenant; got {len(hits)} — multi-tenant filter broken"
    )


@pytest.mark.integration
async def test_hybrid_search_respects_top_k(ingested_doc):
    hits = await hybrid_search(
        "TED score table structure",
        tenant_id=ingested_doc["tenant_id"],
        top_k=2,
    )
    assert len(hits) <= 2


# ── Semantic cache ────────────────────────────────────────────────


@pytest.mark.integration
async def test_cache_miss_then_hit_then_invalidate(ingested_doc):
    tenant = ingested_doc["tenant_id"]
    query = f"unique-test-query-{uuid.uuid4().hex[:8]} how does it work?"

    # Pre-condition: a freshly-generated query is not in cache.
    embedding = await embed_query_dense(query)
    miss = await check_cache(query, tenant_id=tenant, embedding=embedding)
    assert miss is None, "expected fresh query to miss"

    # Write, then re-check — exact-string lookup must hit at similarity ~1.0.
    response = {"answer": "test cache value", "citations": []}
    await write_cache(query, tenant_id=tenant, embedding=embedding, response=response)

    hit = await check_cache(query, tenant_id=tenant, embedding=embedding)
    assert hit is not None
    assert hit["answer"] == "test cache value"
    sim = hit.get("_cache_similarity")
    assert sim is not None and sim >= settings.CACHE_SIMILARITY_THRESHOLD

    # Unrelated query for the same tenant must miss.
    other_query = f"completely different topic {uuid.uuid4().hex[:8]} kangaroos"
    other_embed = await embed_query_dense(other_query)
    other_miss = await check_cache(
        other_query, tenant_id=tenant, embedding=other_embed
    )
    assert other_miss is None

    # Cleanup — leave no test entries behind.
    deleted = await invalidate_tenant_cache(tenant)
    assert deleted >= 1
