"""Async RAGAS scoring of a single (query, answer, context) triple.

Drives `app.ingestion.tasks.score_and_log`, which fires after every fresh
`/chat/stream` response. The four metrics:

    faithfulness        Does every claim in the answer appear in the contexts?
    answer_relevancy    Does the answer actually address the user's question?
    context_precision   Are the retrieved contexts relevant to the question?
    context_recall      Do the contexts cover everything needed for ground truth?
                        (Only computed when a ground_truth is supplied — i.e.
                        the batch eval script. The live hot path returns None
                        for this metric.)

The eval LLM is the same OpenRouter-backed `openai/gpt-4o` we use for
generation; eval embeddings are `openai/text-embedding-3-small`. This is
deliberate — having the eval model be at least as strong as the generation
model is the standard RAGAS recommendation.
"""

from __future__ import annotations

import logging
from typing import Any

from ragas.embeddings.openai_provider import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.core.config import settings
from app.core.llm import get_async_openai_client

log = logging.getLogger(__name__)

_llm = None
_embeddings = None


def _get_eval_llm():
    global _llm
    if _llm is None:
        _llm = llm_factory(
            model=settings.LLM_MODEL_GENERATE,
            provider="openai",
            client=get_async_openai_client(),
        )
    return _llm


def _get_eval_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = RagasOpenAIEmbeddings(
            client=get_async_openai_client(),
            model=settings.EMBEDDING_MODEL,
        )
    return _embeddings


def _to_float(metric_result: Any) -> float | None:
    """RAGAS returns a MetricResult; pull a single float out of it."""
    if metric_result is None:
        return None
    val = getattr(metric_result, "value", metric_result)
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


async def score(
    query: str,
    answer: str,
    contexts: list[str],
    *,
    ground_truth: str | None = None,
) -> dict[str, float | None]:
    """Score a single (query, answer, contexts) triple. Returns a dict with
    keys faithfulness / answer_relevancy / context_precision / context_recall.

    `context_recall` requires a `ground_truth`; without one it's None.
    Each metric is wrapped in try/except — one failed metric never
    poisons the whole eval (we'd rather ship 3/4 scores than 0/4).
    """
    if not query or not answer or not contexts:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

    llm = _get_eval_llm()
    embeddings = _get_eval_embeddings()

    out: dict[str, float | None] = {}

    try:
        f = await Faithfulness(llm=llm).ascore(
            user_input=query, response=answer, retrieved_contexts=contexts
        )
        out["faithfulness"] = _to_float(f)
    except Exception:
        log.exception("ragas faithfulness failed")
        out["faithfulness"] = None

    try:
        r = await AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(
            user_input=query, response=answer
        )
        out["answer_relevancy"] = _to_float(r)
    except Exception:
        log.exception("ragas answer_relevancy failed")
        out["answer_relevancy"] = None

    # context_precision REQUIRES a reference. When no ground_truth is supplied
    # (live hot path), pass the answer itself — RAGAS docs note this gives
    # a "self-precision" approximation that's still useful as a relative
    # signal across queries on the same corpus.
    try:
        ref = ground_truth or answer
        cp = await ContextPrecision(llm=llm).ascore(
            user_input=query, reference=ref, retrieved_contexts=contexts
        )
        out["context_precision"] = _to_float(cp)
    except Exception:
        log.exception("ragas context_precision failed")
        out["context_precision"] = None

    if ground_truth:
        try:
            cr = await ContextRecall(llm=llm).ascore(
                user_input=query, retrieved_contexts=contexts, reference=ground_truth
            )
            out["context_recall"] = _to_float(cr)
        except Exception:
            log.exception("ragas context_recall failed")
            out["context_recall"] = None
    else:
        out["context_recall"] = None

    return out
