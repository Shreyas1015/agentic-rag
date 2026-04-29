"""LangGraph node: post-generation faithfulness verification.

Runs after `generate`. Asks Gemini Flash whether every factual claim in
the answer is supported by the source chunks; if `score < THRESHOLD`,
regenerates ONCE with the unsupported claims explicitly excluded from
the system prompt.

The original `generate` node uses `streaming=True` so the API endpoint
can stream tokens to the client. The regeneration here is non-streaming
(we already streamed the first attempt) — the client gets the corrected
answer in the `done` event payload, replacing the streamed-but-flawed
text. That's a small UX wart but the right safety trade-off; alternative
designs (don't stream until faithfulness passes) defeat the streaming
UX entirely.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from app.agent.state import AgentState
from app.core.config import settings
from app.core.llm import extract_json, get_chat_model
from app.observability.langfuse_client import langfuse

log = logging.getLogger(__name__)

THRESHOLD = 0.85

VERIFY_PROMPT = """You are a faithfulness verifier. Given an ANSWER and the SOURCE CHUNKS it was supposed to be grounded in, identify any factual claim in the answer that is NOT supported by the chunks.

Rules:
- Score 1.0 if every factual claim is verbatim or paraphrased from the chunks.
- Lower the score for each unsupported / fabricated claim.
- "Unsupported" means: the chunks don't say it (silence is unsupported, even if the claim is plausible).
- DO NOT use outside knowledge — judge only against the chunks.

Respond ONLY with valid JSON:
{"faithfulness_score": <float 0.0-1.0>, "unsupported_claims": ["...", "..."]}"""


def _format_sources(parent_chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[chunk {i + 1}, page {pc.get('page_num', '?')}]\n{pc.get('text', '')}"
        for i, pc in enumerate(parent_chunks)
    )


async def _verify(answer: str, parent_chunks: list[dict]) -> tuple[float, list[str]]:
    """Single Gemini Flash JSON call. Returns (score, unsupported_claims)."""
    user_message = (
        f"ANSWER:\n{answer}\n\n"
        f"SOURCE CHUNKS:\n{_format_sources(parent_chunks)}"
    )
    model = get_chat_model(
        settings.LLM_MODEL_ASSESS, temperature=0.0, json_mode=True
    )
    response = await model.ainvoke(
        [SystemMessage(content=VERIFY_PROMPT), HumanMessage(content=user_message)]
    )
    try:
        parsed = extract_json(response.content)
        score = float(parsed.get("faithfulness_score", 0.0))
        claims = parsed.get("unsupported_claims") or []
        if not isinstance(claims, list):
            claims = []
        claims = [str(c).strip() for c in claims if str(c).strip()]
    except (ValueError, AttributeError, TypeError) as exc:
        log.warning("faithfulness: parse failed (%s); defaulting score=0", exc)
        return 0.0, []
    return max(0.0, min(1.0, score)), claims


async def _regenerate_with_constraints(
    state: AgentState, unsupported: list[str], filenames_hint: str
) -> str:
    """Re-run a non-streaming generation excluding the listed claims."""
    from app.agent.nodes.generate import SYSTEM_PROMPT  # avoid circular import

    constraints = (
        "\n\nADDITIONAL CONSTRAINT — the previous attempt made these "
        "unsupported claims; do NOT repeat them: "
        + "; ".join(unsupported)
    )
    user_message = (
        f"USER QUESTION:\n{state.get('original_query') or state['query']}\n\n"
        f"SOURCE CHUNKS:\n{filenames_hint}"
    )
    model = get_chat_model(
        settings.LLM_MODEL_GENERATE, temperature=0.0, streaming=False
    )
    response = await model.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT + constraints),
            HumanMessage(content=user_message),
        ]
    )
    return (response.content or "").strip()


@observe(name="faithfulness")
async def faithfulness_check(state: AgentState) -> dict:
    answer = state.get("final_answer") or ""
    parent_chunks = state.get("parent_chunks") or []
    if not answer or not parent_chunks:
        langfuse.update_current_span(
            output={"score": None}, metadata={"reason": "no answer or no chunks"}
        )
        return {"faithfulness_score": 0.0}

    score, unsupported = await _verify(answer, parent_chunks)
    update: dict[str, Any] = {"faithfulness_score": score}

    if score < THRESHOLD and unsupported:
        log.info(
            "faithfulness: %.2f below threshold %.2f; regenerating with %d constraints",
            score, THRESHOLD, len(unsupported),
        )
        # Reuse generate.py's source-formatting (filenames + page numbers).
        from app.agent.nodes.generate import _format_context
        from app.db import crud
        from app.db.session import async_session_maker

        document_ids = list(
            {pc.get("document_id", "") for pc in parent_chunks if pc.get("document_id")}
        )
        async with async_session_maker() as session:
            filenames = await crud.fetch_document_filenames(
                session, tenant_id=state["tenant_id"], document_ids=document_ids
            )
        sources_block = _format_context(parent_chunks, filenames)
        new_answer = await _regenerate_with_constraints(
            state, unsupported, sources_block
        )
        # Re-verify the regenerated answer (one more pass — helps observability,
        # we don't loop forever).
        new_score, _ = await _verify(new_answer, parent_chunks)
        update["final_answer"] = new_answer
        update["faithfulness_score"] = new_score
        langfuse.update_current_span(
            input={"answer_len": len(answer), "unsupported": len(unsupported)},
            output={"score_pre": score, "score_post": new_score, "regenerated": True},
            metadata={"threshold": THRESHOLD, "claims_dropped": unsupported[:5]},
        )
    else:
        langfuse.update_current_span(
            input={"answer_len": len(answer)},
            output={"score": score, "regenerated": False},
            metadata={"threshold": THRESHOLD, "passed": score >= THRESHOLD},
        )

    return update
