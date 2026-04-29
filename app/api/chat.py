"""Chat API — POST /chat/stream (Server-Sent Events).

Event types emitted to the client:
  status     : a graph node is starting   {node, message}
  cache_hit  : answered from semantic cache directly
  token      : a streamed token from the generate node {text}
  done       : final payload {answer, citations, context_score, iteration, from_cache}
  error      : something blew up — for visibility, not for retry semantics

Flow:
  1. Auth via Logto (require_tenant) + body validation
  2. Embed the query once
  3. Semantic cache lookup; on hit, replay the cached answer + done, return
  4. On miss, drive the LangGraph agent via astream_events("v2"), translating
     LangChain events into SSE events; only the generate node's tokens are
     forwarded as `token` events (a flag tracks whether we're inside it).
  5. After completion, write the response back to the cache for next time.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import build_graph
from app.core.auth import require_tenant
from app.core.qdrant_client import collection_name_for
from app.ingestion.embedder import embed_query_dense
from app.observability.langfuse_client import current_trace_id, langfuse
from app.retrieval.cache import check_cache, write_cache

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Friendly status messages emitted as the graph progresses.
_NODE_MESSAGES: dict[str, str] = {
    "classify": "Classifying query...",
    "decompose": "Decomposing into sub-questions...",
    "retrieve": "Searching documents...",
    "parent_fetch": "Fetching context...",
    "assess": "Assessing context quality...",
    "reformulate": "Reformulating query...",
    "generate": "Generating answer...",
}


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


def _sse(event: str, data: dict[str, Any]) -> str:
    """One SSE message — `event:` line + JSON `data:` line + blank-line terminator."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_cache_hit(cached: dict[str, Any]) -> AsyncIterator[str]:
    similarity = cached.pop("_cache_similarity", None)
    yield _sse(
        "cache_hit",
        {"message": "Found a cached answer", "similarity": similarity},
    )
    answer = cached.get("answer") or ""
    if answer:
        yield _sse("token", {"text": answer})
    yield _sse("done", {**cached, "from_cache": True})


async def _stream_agent(
    query: str, tenant_id: str, embedding: list[float]
) -> AsyncIterator[str]:
    graph = build_graph()
    initial = {
        "query": query,
        "original_query": query,
        "tenant_id": tenant_id,
        "collection": collection_name_for(tenant_id),
        "iteration": 0,
        "sub_questions": [],
    }

    final_state: dict[str, Any] = {}
    in_generate = False  # token events only fire while the generate node is active

    # Root span — every @observe inside the graph becomes a child, sharing
    # one trace_id we surface in the `done` event so /feedback can target it.
    with langfuse.start_as_current_observation(
        name="chat.stream",
        as_type="agent",
        input={"query": query, "tenant_id": tenant_id},
    ):
        trace_id = current_trace_id()

        try:
            async for ev in graph.astream_events(initial, version="v2"):
                kind = ev.get("event", "")
                name = ev.get("name", "")

                if kind == "on_chain_start" and name in _NODE_MESSAGES:
                    if name == "generate":
                        in_generate = True
                    yield _sse(
                        "status", {"node": name, "message": _NODE_MESSAGES[name]}
                    )

                elif kind == "on_chain_end" and name in _NODE_MESSAGES:
                    if name == "generate":
                        in_generate = False
                    # Each node returns a partial state update — accumulate.
                    update = (ev.get("data") or {}).get("output")
                    if isinstance(update, dict):
                        final_state.update(update)

                elif kind == "on_chat_model_stream" and in_generate:
                    chunk = (ev.get("data") or {}).get("chunk")
                    text = getattr(chunk, "content", "") if chunk is not None else ""
                    if text:
                        yield _sse("token", {"text": text})
        except Exception as exc:
            log.exception("chat stream agent error")
            yield _sse("error", {"message": str(exc), "trace_id": trace_id})
            return

        payload = {
            "answer": final_state.get("final_answer", ""),
            "citations": final_state.get("citations", []),
            "context_score": final_state.get("context_score"),
            "iteration": final_state.get("iteration", 0),
            "from_cache": False,
            "trace_id": trace_id,
        }
        langfuse.update_current_span(
            output={
                "answer_length": len(payload["answer"]),
                "citations": len(payload["citations"]),
                "context_score": payload["context_score"],
                "iteration": payload["iteration"],
            },
        )

    # Yield + cache write happen *outside* the span so the trace closes promptly.
    yield _sse("done", payload)

    if payload["answer"]:
        try:
            await write_cache(
                query, tenant_id=tenant_id, embedding=embedding, response=payload
            )
        except Exception:
            log.exception("cache write failed (non-fatal)")


@router.post("/stream", summary="Submit a query, receive an SSE stream")
async def chat_stream(
    body: ChatRequest, tenant_id: str = Depends(require_tenant)
) -> StreamingResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty query")

    # Embed once, share with check + write so we don't pay twice.
    try:
        embedding = await embed_query_dense(query)
    except Exception as exc:
        log.exception("embedding failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"Embedding service error: {exc}"
        ) from exc

    cached = await check_cache(query, tenant_id=tenant_id, embedding=embedding)

    async def _gen() -> AsyncIterator[str]:
        if cached is not None:
            async for chunk in _stream_cache_hit(cached):
                yield chunk
        else:
            async for chunk in _stream_agent(query, tenant_id, embedding):
                yield chunk

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
        },
    )
