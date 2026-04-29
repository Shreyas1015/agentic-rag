"""LangGraph node: rerank retrieved children with the local BGE cross-encoder.

Inserted between `retrieve` and `parent_fetch` in the graph. Takes the up-to-30
RRF-fused candidates from hybrid search and returns the top-8 by BGE score.

The torch forward pass is CPU-bound (~20 ms for 30 pairs), so we hop off the
event loop with `asyncio.to_thread` — keeps concurrent SSE streams responsive.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from langfuse import observe

from app.agent.state import AgentState
from app.observability.langfuse_client import langfuse
from app.retrieval.bge_reranker import rerank_chunks
from app.retrieval.hybrid_search import RetrievedChunk

TOP_K = 8


@observe(name="rerank")
async def rerank(state: AgentState) -> dict:
    raw = state.get("retrieved_chunks") or []
    if not raw:
        langfuse.update_current_span(
            output={"after": 0}, metadata={"reason": "no retrieved_chunks"}
        )
        return {"retrieved_chunks": []}

    # State carries dicts (asdict() of RetrievedChunk in the retrieve node);
    # reconstruct the dataclass for the reranker, then dump back to dicts so
    # downstream nodes (parent_fetch) keep the existing contract.
    chunks = [RetrievedChunk(**c) for c in raw]
    pre_top3 = [round(c.score, 4) for c in chunks[:3]]

    reranked = await asyncio.to_thread(
        rerank_chunks, state["query"], chunks, top_k=TOP_K
    )

    post_top3 = [round(c.score, 4) for c in reranked[:3]]
    langfuse.update_current_span(
        input={"before": len(chunks), "query": state["query"]},
        output={"after": len(reranked)},
        metadata={
            "model": "BAAI/bge-reranker-v2-m3",
            "top_k": TOP_K,
            "rrf_top3": pre_top3,
            "bge_top3": post_top3,
            "top1_changed": chunks[0].chunk_id != reranked[0].chunk_id if reranked else False,
        },
    )
    return {"retrieved_chunks": [asdict(c) for c in reranked]}
