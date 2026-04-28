"""Rewrite the query using the assessment reason, then bump iteration.

Used when context_score < threshold AND iteration < MAX_RETRIEVAL_ITERATIONS.
The reformulated query feeds back into retrieve through the graph.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.core.config import settings
from app.core.llm import extract_json, get_chat_model

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a query rewriter for a RAG system. The previous retrieval was inadequate; rewrite the user's question so a vector + keyword search has a better chance of pulling the missing context.

Strategies:
- expand abbreviations / acronyms
- include likely synonyms
- be more specific about what's missing (per the reason)
- keep the rewrite concise — one sentence

Reply ONLY with valid JSON: {"query": "<rewritten question>"}"""


async def reformulate_query(state: AgentState) -> dict:
    original = state.get("original_query") or state["query"]
    reason = state.get("context_reason") or "previous chunks were insufficient"
    iteration = int(state.get("iteration", 0)) + 1

    user_message = (
        f"ORIGINAL QUESTION:\n{original}\n\nLAST QUERY USED:\n{state['query']}\n\n"
        f"WHY THE RETRIEVAL FAILED:\n{reason}\n\n"
        f"Rewrite the question to recover the missing context."
    )

    model = get_chat_model(
        settings.LLM_MODEL_REFORMULATE, temperature=0.0, json_mode=True
    )
    response = await model.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )

    new_query = state["query"]
    try:
        parsed = extract_json(response.content)
        candidate = str(parsed.get("query", "")).strip() if isinstance(parsed, dict) else ""
        if candidate:
            new_query = candidate
    except (ValueError, AttributeError) as exc:
        log.warning("reformulate: parse failed (%s); keeping prior query", exc)

    # multi_part: clear sub_questions so retrieve treats this as a single query
    return {
        "query": new_query,
        "iteration": iteration,
        "sub_questions": [],
    }
