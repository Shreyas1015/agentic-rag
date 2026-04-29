"""Local cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Replaces the artifact's deprecated Cohere-API plan. Runs in-process inside
both the api and worker containers (single image, persisted in the
model_cache named volume at /root/.cache/huggingface). CPU is fine for
the ~30-pair forward pass we do per query — ~15-25 ms.

Usage from `app/agent/nodes/rerank.py`:

    reranked = await asyncio.to_thread(
        rerank_chunks, query, retrieved_chunks, top_k=8
    )

The asyncio.to_thread wrapper keeps the event loop responsive — a single
torch forward pass blocks for ~20 ms which is enough to throw off other
concurrent SSE streams if we don't off-thread it.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from threading import Lock

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.retrieval.hybrid_search import RetrievedChunk

log = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512  # BGE-reranker-v2-m3 truncates pairs to this many tokens

_tokenizer = None
_model = None
_load_lock = Lock()


def _load_model() -> tuple:
    """Lazy + thread-safe load. First call ~5s; subsequent calls instant."""
    global _tokenizer, _model
    if _model is None:
        with _load_lock:
            if _model is None:  # double-checked under lock
                log.info("loading BGE reranker model %s ...", MODEL_NAME)
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
                _model.eval()
    return _tokenizer, _model


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    """Cross-encoder rerank: return the top-k chunks by BGE score.

    The new score replaces the prior RRF score on the returned RetrievedChunk
    so downstream code (e.g. the assess prompt, Langfuse metadata) sees the
    reranker's view of the world. Tied / identical scores fall back to input
    order.
    """
    if not chunks:
        return []

    tokenizer, model = _load_model()
    pairs = [(query, c.text_preview) for c in chunks]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        logits = model(**inputs).logits.view(-1)
        scores = logits.tolist()

    rescored = [
        replace(c, score=float(s)) for c, s in zip(chunks, scores, strict=True)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_k]
