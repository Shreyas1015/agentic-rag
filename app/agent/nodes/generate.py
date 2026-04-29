"""Final answer generation with citations.

GPT-4o reads the parent chunks and answers the original user question.
The system prompt is strict about hallucination control: "if the sources
don't say it, don't say it." Inline `[Source: <filename>, Page X]` markers
are emitted by the model; we also build a structured `citations` list
from the parent_chunks actually shown to it (one per unique document).

Streaming: get_chat_model returns a ChatOpenAI with streaming=True so
LangGraph's astream_events surfaces tokens to the API endpoint (Step F).
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

SYSTEM_PROMPT = """You are a precise document QA assistant. Answer the user's question using ONLY the SOURCE CHUNKS supplied below.

Rules — follow strictly:
1. If the sources don't contain the answer, say: "The provided documents don't contain enough information to answer that."
2. Every factual claim must be grounded in a specific source chunk.
3. Cite each claim inline using `[Source: <filename>, Page X]`. Use the filename and page from the chunk header.
4. Quote numbers / names verbatim — do not paraphrase quantitative values.
5. Be concise. Short paragraphs or short bullets, no preamble.
"""


def _format_context(parent_chunks: list[dict], filenames: dict[str, str]) -> str:
    blocks = []
    for i, pc in enumerate(parent_chunks, 1):
        fn = filenames.get(pc.get("document_id", ""), "unknown.pdf")
        page = pc.get("page_num", "?")
        blocks.append(f"[chunk {i} | {fn} | page {page}]\n{pc.get('text', '')}")
    return "\n\n---\n\n".join(blocks)


@observe(name="generate", as_type="generation")
async def generate_answer(state: AgentState) -> dict:
    parent_chunks = state.get("parent_chunks") or []
    tenant_id = state["tenant_id"]
    question = state.get("original_query") or state["query"]

    # Fetch filenames once for citation rendering (one query per generate call).
    document_ids = list({pc.get("document_id", "") for pc in parent_chunks if pc.get("document_id")})
    if document_ids:
        async with async_session_maker() as session:
            filenames = await crud.fetch_document_filenames(
                session, tenant_id=tenant_id, document_ids=document_ids
            )
    else:
        filenames = {}

    if not parent_chunks:
        langfuse.update_current_generation(
            input=[{"role": "user", "content": question}],
            output={"answer": "no parent chunks"},
            metadata={"reason": "empty_context"},
        )
        return {
            "final_answer": "The provided documents don't contain enough information to answer that.",
            "citations": [],
        }

    user_message = (
        f"USER QUESTION:\n{question}\n\nSOURCE CHUNKS:\n"
        f"{_format_context(parent_chunks, filenames)}"
    )

    model = get_chat_model(
        settings.LLM_MODEL_GENERATE, temperature=0.0, streaming=True
    )
    response = await model.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )

    # Build citations from the chunks we actually showed the model.
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

    answer = (response.content or "").strip()
    usage_kwargs = usage_from_response(response)
    # Fall back to the settings-level model id if the response didn't carry one.
    usage_kwargs.setdefault("model", settings.LLM_MODEL_GENERATE)
    langfuse.update_current_generation(
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        output=answer,
        metadata={
            "context_score": state.get("context_score"),
            "iteration": int(state.get("iteration", 0)),
            "citations": len(citations),
            "documents": len(filenames),
        },
        **usage_kwargs,
    )
    return {
        "final_answer": answer,
        "citations": citations,
    }
