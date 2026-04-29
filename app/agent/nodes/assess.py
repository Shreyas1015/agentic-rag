"""Score retrieved context against the query (0-10) and explain why.

Drives the conditional edge in the graph:
  >= CONTEXT_SCORE_THRESHOLD               -> generate
  <  threshold AND iteration < MAX_RETRIES -> reformulate (loop back)
  <  threshold AND iteration >= MAX_RETRIES -> generate anyway (best effort)
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from app.agent.state import AgentState
from app.core.config import settings
from app.core.llm import extract_json, get_chat_model
from app.observability.langfuse_client import langfuse

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a context grader. Score how well the SOURCE CHUNKS support an answer to the USER QUESTION.

Important rules:
- Score based ONLY on what is literally written in the chunks. Do NOT use outside knowledge of what a term "usually" means; if the chunks define a term in a particular way, that is the term's meaning here.
- A high score means the answer can be assembled from the chunks; a low score means it cannot.

Score from 0 to 10:
  10: chunks contain the full answer with specifics
   7-9: most of the answer is present, minor gaps
   4-6: partial — useful but missing key parts
   0-3: chunks are off-topic or insufficient

Respond ONLY with valid JSON: {"score": <int 0-10>, "reason": "<one short sentence about what the chunks actually say>"}"""


def _format_chunks(parent_chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[chunk {i + 1}, page {pc.get('page_num', '?')}]\n{pc.get('text', '')}"
        for i, pc in enumerate(parent_chunks)
    )


@observe(name="assess")
async def assess_context(state: AgentState) -> dict:
    parent_chunks = state.get("parent_chunks") or []
    if not parent_chunks:
        langfuse.update_current_span(
            output={"score": 0.0}, metadata={"reason": "no parent chunks"}
        )
        return {"context_score": 0.0, "context_reason": "no parent chunks were retrieved"}

    user_message = (
        f"USER QUESTION:\n{state['query']}\n\nSOURCE CHUNKS:\n"
        f"{_format_chunks(parent_chunks)}"
    )

    model = get_chat_model(
        settings.LLM_MODEL_ASSESS, temperature=0.0, json_mode=True
    )
    response = await model.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )

    score: float = 0.0
    reason: str = ""
    try:
        parsed = extract_json(response.content)
        score = float(parsed.get("score", 0))
        reason = str(parsed.get("reason", "")).strip()
    except (ValueError, AttributeError, TypeError) as exc:
        log.warning("assess: parse failed (%s); defaulting score=0", exc)

    score = max(0.0, min(10.0, score))
    langfuse.update_current_span(
        input={"query": state["query"], "chunks": len(parent_chunks)},
        output={"score": score, "reason": reason},
        metadata={
            "threshold": settings.CONTEXT_SCORE_THRESHOLD,
            "passes": score >= settings.CONTEXT_SCORE_THRESHOLD,
            "iteration": int(state.get("iteration", 0)),
            "model": settings.LLM_MODEL_ASSESS,
        },
    )
    return {"context_score": score, "context_reason": reason}
