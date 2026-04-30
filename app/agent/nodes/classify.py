"""Classify the user's query into one of three routes for the graph.

Returns a partial state update setting `query_type`:
  simple_factual : single-fact lookup; goes straight to retrieve
  multi_part     : compound / comparison; goes through decompose first
  procedural     : how-to / step-by-step; goes straight to retrieve
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from app.agent.state import AgentState, QueryType
from app.core.config import settings
from app.core.llm import extract_json, get_chat_model
from app.db import crud
from app.db.session import async_session_maker
from app.observability.langfuse_client import langfuse, usage_from_response

log = logging.getLogger(__name__)

_VALID: set[QueryType] = {"conversational", "simple_factual", "multi_part", "procedural"}

SYSTEM_PROMPT = """You are a query router for a document-QA assistant. Classify the user's input into exactly one of four categories.

Categories:
- "conversational": greetings, thanks, small-talk, or meta-questions about the assistant itself. The user is NOT asking about the corpus. Cheap to answer with no retrieval.
- "simple_factual": a single-fact lookup against the corpus.
- "multi_part": a compound / comparison / multi-entity question that benefits from being split into sub-questions before retrieval.
- "procedural": a how-to or step-by-step question against the corpus.

Examples:
Input: "hello"
Output: {"query_type": "conversational"}

Input: "thanks!"
Output: {"query_type": "conversational"}

Input: "what can you do?"
Output: {"query_type": "conversational"}

Input: "What is the TED score?"
Output: {"query_type": "simple_factual"}

Input: "Who wrote Docling?"
Output: {"query_type": "simple_factual"}

Input: "Compare BERT and GPT-4 across latency and quality."
Output: {"query_type": "multi_part"}

Input: "What are the differences between dense, sparse, and hybrid retrieval?"
Output: {"query_type": "multi_part"}

Input: "How do I run inference?"
Output: {"query_type": "procedural"}

Input: "Steps to set up the environment?"
Output: {"query_type": "procedural"}

Tie-breaker rule: if the input could plausibly be either conversational or a real question about the corpus, pick the corpus category (simple_factual / multi_part / procedural). Misclassifying a real question as conversational hides information from the user. Misclassifying small-talk as a real question only wastes a few tokens.

Reply ONLY with valid JSON: {"query_type": "<one-of-the-four>"}"""


@observe(name="classify", as_type="generation")
async def classify_query(state: AgentState) -> dict:
    query = state["query"]
    model = get_chat_model(
        settings.LLM_MODEL_CLASSIFY, temperature=0.0, json_mode=True
    )
    response = await model.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
    )
    try:
        parsed = extract_json(response.content)
        qt = parsed.get("query_type")
    except (ValueError, AttributeError) as exc:
        log.warning("classify: JSON parse failed (%s); defaulting to simple_factual", exc)
        qt = "simple_factual"

    if qt not in _VALID:
        log.warning("classify: invalid query_type %r; defaulting to simple_factual", qt)
        qt = "simple_factual"

    # Populate tenant_token_count so the next route can decide whether to
    # take the long-context bypass path. Cheap (one COALESCE/SUM in PG).
    async with async_session_maker() as session:
        token_count = await crud.estimate_tenant_token_count(
            session, tenant_id=state["tenant_id"]
        )

    langfuse.update_current_generation(
        input=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}],
        output={"query_type": qt},
        metadata={
            "tenant_token_count": token_count,
            "bypass_budget": settings.LONG_CONTEXT_BYPASS_TOKEN_BUDGET,
        },
        **usage_from_response(response),
    )
    return {"query_type": qt, "tenant_token_count": token_count}
