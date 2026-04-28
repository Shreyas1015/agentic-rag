"""Redis-backed semantic cache for query responses.

Lives on Redis DB 2 (Celery uses DB 0/1). Keyed by tenant + a stable
SHA256 of the exact query string; the value is JSON containing the dense
embedding of the query and the cached response. Lookup re-embeds the new
query and compares against every cached embedding for the tenant — if any
exceeds CACHE_SIMILARITY_THRESHOLD (cosine), we return the response.

Phase 1 limitations (acceptable for single-tenant low volume):
  - O(N) scan over the tenant's cache on each lookup. For high volume,
    swap to RediSearch / a Qdrant cache collection.
  - Stores the raw embedding in Redis (~6 KB / entry for 1536-d) — fine
    given the 24 h TTL.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import redis.asyncio as aioredis

from app.core.config import settings
from app.ingestion.embedder import embed_query_dense

_redis: aioredis.Redis | None = None


def _get_cache_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.SEMANTIC_CACHE_REDIS_URL, decode_responses=True
        )
    return _redis


def _key(tenant_id: str, query: str) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"cache:{tenant_id}:{digest}"


def _cosine(a: list[float], b: list[float]) -> float:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / denom)


async def check_cache(query: str, *, tenant_id: str) -> dict[str, Any] | None:
    """Return the cached response if any tenant entry's embedding is similar
    enough to `query`'s embedding (cosine ≥ threshold), else None.

    Computes the dense query embedding once per call. Misses are O(N) over
    the tenant's cache — see module docstring."""
    if not query.strip():
        return None
    r = _get_cache_redis()

    # Iterate just this tenant's entries (cheap on Phase-1 traffic).
    matches: list[tuple[float, dict]] = []
    new_vec = await embed_query_dense(query)
    async for key in r.scan_iter(match=f"cache:{tenant_id}:*", count=100):
        raw = await r.get(key)
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        sim = _cosine(new_vec, entry.get("embedding") or [])
        if sim >= settings.CACHE_SIMILARITY_THRESHOLD:
            matches.append((sim, entry))

    if not matches:
        return None
    # Best match wins.
    matches.sort(key=lambda t: t[0], reverse=True)
    best_sim, best = matches[0]
    response = best.get("response")
    if isinstance(response, dict):
        response = {**response, "_cache_similarity": best_sim}
    return response


async def write_cache(
    query: str,
    *,
    tenant_id: str,
    embedding: list[float],
    response: dict[str, Any],
) -> str:
    """Persist (query embedding, response) under the tenant's namespace.
    Reuses the embedding the caller already computed at retrieval time so we
    don't pay for it twice."""
    r = _get_cache_redis()
    key = _key(tenant_id, query)
    payload = json.dumps(
        {"query": query, "embedding": embedding, "response": response}
    )
    await r.setex(key, settings.CACHE_TTL_SECONDS, payload)
    return key


async def invalidate_tenant_cache(tenant_id: str) -> int:
    """Drop every cached query for a tenant. Useful after re-ingest /
    soft-delete so stale answers don't keep getting returned."""
    r = _get_cache_redis()
    deleted = 0
    async for key in r.scan_iter(match=f"cache:{tenant_id}:*", count=200):
        deleted += await r.delete(key)
    return deleted
