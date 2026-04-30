"""LangGraph node: long-context bypass.

When the full active corpus for a tenant fits inside Gemini 2.5 Pro's 1M-token
window, retrieval is the bottleneck — there's nothing to "retrieve" because
*everything is relevant*. We skip retrieve / rerank / parent_fetch / assess
and stuff the whole corpus directly into the long-context model alongside
the question.

The decision is made by `_route_after_classify` in `app/agent/graph.py`:
it calls `crud.estimate_tenant_token_count` once per request and routes
to this node if the count is under `LONG_CONTEXT_BYPASS_TOKEN_BUDGET`.

This node populates `parent_chunks` and `final_answer` on state so the
downstream `faithfulness` node still runs as a defensive check on the
generated answer.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from app.agent.state import AgentState
from app.core.config import settings
from app.core.llm import get_chat_model
from app.db import crud
from app.db.session import async_session_maker
from app.observability.langfuse_client import langfuse, usage_from_response

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant for the user's document corpus, operating in long-context mode. The full corpus for this tenant is supplied below.

If the user asks something about the documents:
1. Answer using ONLY the corpus.
2. If the corpus doesn't contain the answer, say: "The provided documents don't contain enough information to answer that."
3. Every factual claim must be grounded in the corpus.
4. Cite each claim inline using `[Source: <filename>, Page X]` (filename + page appear in each chunk header).
5. Quote numbers / names verbatim — do not paraphrase quantitative values.

If the user is greeting you, thanking you, or making small-talk (e.g. "hello", "hi", "thanks", "how are you"):
- Respond naturally in one short line.
- Do NOT recite "the documents don't contain enough information" — that rule is only for genuine document questions.
- Briefly invite them to ask something about the documents.

Always be concise. Short paragraphs or short bullets, no preamble."""


def _format_corpus(parent_chunks: list[dict], filenames: dict[str, str]) -> str:
    """Render every parent chunk with filename + page header so citations
    work the same as in the RAG path."""
    blocks = []
    for i, pc in enumerate(parent_chunks, 1):
        fn = filenames.get(pc.get("document_id", ""), "unknown.pdf")
        page = pc.get("page_num", "?")
        blocks.append(f"[chunk {i} | {fn} | page {page}]\n{pc.get('text', '')}")
    return "\n\n---\n\n".join(blocks)


@observe(name="bypass_generate", as_type="generation")
async def bypass_generate(state: AgentState) -> dict:
    """Skip-RAG generation: load the entire active corpus, pass to a
    long-context model, build citations from the chunks shown."""
    tenant_id = state["tenant_id"]
    question = state.get("original_query") or state["query"]

    # Pull every active parent chunk for the tenant + their filenames.
    async with async_session_maker() as session:
        rows = await crud.fetch_all_active_parent_chunks(session, tenant_id=tenant_id)
        document_ids = list({str(r.document_id) for r in rows})
        filenames = await crud.fetch_document_filenames(
            session, tenant_id=tenant_id, document_ids=document_ids
        )

    parent_chunks: list[dict] = [
        {
            "parent_id": r.parent_id,
            "document_id": str(r.document_id),
            "tenant_id": r.tenant_id,
            "text": r.text,
            "page_num": r.page_num,
            "chunk_index": r.chunk_index,
        }
        for r in rows
    ]

    if not parent_chunks:
        langfuse.update_current_generation(
            output="no parent_chunks", metadata={"reason": "tenant has no active corpus"}
        )
        return {
            "parent_chunks": [],
            "final_answer": "The provided documents don't contain enough information to answer that.",
            "citations": [],
            "context_score": 0.0,
        }

    corpus = _format_corpus(parent_chunks, filenames)
    user_message = f"USER QUESTION:\n{question}\n\nCORPUS:\n{corpus}"

    model = get_chat_model(
        settings.LLM_MODEL_LONG_CONTEXT, temperature=0.0, streaming=True
    )
    response = await model.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )
    answer = (response.content or "").strip()

    # Build citations from the chunks we showed the model — same shape as RAG.
    citations: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for pc in parent_chunks:
        doc_id = pc.get("document_id", "")
        page = int(pc.get("page_num", 0))
        if (doc_id, page) in seen:
            continue
        seen.add((doc_id, page))
        citations.append(
            {
                "document_id": doc_id,
                "filename": filenames.get(doc_id, "unknown.pdf"),
                "page_num": page,
                "parent_id": pc.get("parent_id", ""),
            }
        )

    usage_kwargs = usage_from_response(response)
    usage_kwargs.setdefault("model", settings.LLM_MODEL_LONG_CONTEXT)
    langfuse.update_current_generation(
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        output=answer,
        metadata={
            "parent_chunks": len(parent_chunks),
            "documents": len(filenames),
            "citations": len(citations),
            "bypass": True,
        },
        **usage_kwargs,
    )

    return {
        # Populate parent_chunks + context_score so faithfulness has signal,
        # and the SSE done payload looks identical to a RAG run.
        "parent_chunks": parent_chunks,
        "context_score": 10.0,
        "final_answer": answer,
        "citations": citations,
    }
