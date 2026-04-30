"""Fast small-talk responder.

When classify routes a query as `conversational` (greetings, thanks,
meta-questions about the assistant), we skip retrieval and the entire
RAG/bypass path. One cheap LLM call, no documents, no faithfulness check.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe

from app.agent.state import AgentState
from app.core.config import settings
from app.core.llm import get_chat_model
from app.observability.langfuse_client import langfuse, usage_from_response

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the conversational voice of a document-QA assistant.

The user has greeted you, thanked you, or asked something about you — they are NOT asking about their documents.

Reply in ONE short, warm line. Then in the same line or a brief follow-up sentence, gently invite them to ask a question about their documents.

Hard rules:
- Never say "the documents don't contain enough information" — that line is reserved for genuine document questions, not small-talk.
- Never invent facts about the documents.
- Do not use citations or `[Source: ...]` markers.
- No preamble, no apologies, no markdown headings."""


@observe(name="chat_smalltalk", as_type="generation")
async def chat_smalltalk(state: AgentState) -> dict:
    query = state.get("original_query") or state["query"]
    model = get_chat_model(
        settings.LLM_MODEL_CLASSIFY, temperature=0.4, streaming=True
    )
    response = await model.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
    )
    answer = (response.content or "").strip()
    usage_kwargs = usage_from_response(response)
    usage_kwargs.setdefault("model", settings.LLM_MODEL_CLASSIFY)
    langfuse.update_current_generation(
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        output=answer,
        **usage_kwargs,
    )
    return {"final_answer": answer, "citations": []}
