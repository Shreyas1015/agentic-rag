"""Decompose a multi-part query into 2-4 atomic sub-questions.

Only invoked when classify returned `multi_part`. Each sub-question gets
its own retrieval pass (in parallel) downstream.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from app.agent.state import AgentState
from app.core.config import settings
from app.core.llm import extract_json, get_chat_model
from app.observability.langfuse_client import langfuse, usage_from_response

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a query decomposer. Break the user's compound question into 2-4 atomic sub-questions, each independently answerable from a document corpus.

Rules:
- Keep each sub-question short and concrete.
- Don't add commentary or numbering.
- If the question is already simple, return a list with just that question.

Reply ONLY with valid JSON: {"sub_questions": ["...", "..."]}"""


@observe(name="decompose", as_type="generation")
async def decompose_question(state: AgentState) -> dict:
    query = state["query"]
    model = get_chat_model(
        settings.LLM_MODEL_DECOMPOSE, temperature=0.0, json_mode=True
    )
    response = await model.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
    )
    try:
        parsed = extract_json(response.content)
        subs = parsed.get("sub_questions") if isinstance(parsed, dict) else parsed
        if not isinstance(subs, list) or not subs:
            raise ValueError("no sub_questions in response")
        subs = [str(s).strip() for s in subs if str(s).strip()][:4]
    except (ValueError, AttributeError) as exc:
        log.warning("decompose: parse failed (%s); falling back to original query", exc)
        subs = [query]

    langfuse.update_current_generation(
        input=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}],
        output={"sub_questions": subs},
        metadata={"count": len(subs)},
        **usage_from_response(response),
    )
    return {"sub_questions": subs}
