"""Run hybrid search and put the results into state.retrieved_chunks.

For multi_part queries we fan out one search per sub_question in parallel
(asyncio.gather) and merge the results, deduping by chunk_id (keeping the
best score per chunk). For other query types we just search once.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from langfuse import observe

from app.agent.state import AgentState
from app.observability.langfuse_client import langfuse
from app.retrieval.hybrid_search import RetrievedChunk, hybrid_search

TOP_K = 30


def _dedupe_keep_best(buckets: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Merge per-sub-question hits, keeping the highest score per chunk_id."""
    best: dict[str, RetrievedChunk] = {}
    for hits in buckets:
        for h in hits:
            prior = best.get(h.chunk_id)
            if prior is None or h.score > prior.score:
                best[h.chunk_id] = h
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


@observe(name="retrieve")
async def retrieve(state: AgentState) -> dict:
    tenant_id = state["tenant_id"]
    sub_questions = state.get("sub_questions") or []

    if sub_questions and len(sub_questions) > 1:
        # multi_part: parallel hybrid search per sub-question, merge + dedupe.
        buckets = await asyncio.gather(
            *(
                hybrid_search(q, tenant_id=tenant_id, top_k=TOP_K)
                for q in sub_questions
            )
        )
        merged = _dedupe_keep_best(buckets)[:TOP_K]
        search_mode = "multi_part"
    else:
        merged = await hybrid_search(state["query"], tenant_id=tenant_id, top_k=TOP_K)
        search_mode = "single"

    langfuse.update_current_span(
        input={"query": state["query"], "sub_questions": sub_questions},
        output={"hits": len(merged)},
        metadata={
            "mode": search_mode,
            "top_k": TOP_K,
            "top_5_scores": [round(c.score, 4) for c in merged[:5]],
        },
    )
    return {"retrieved_chunks": [asdict(c) for c in merged]}
