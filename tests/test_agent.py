"""Tests for the LangGraph agent: full pipeline, end-to-end."""

from __future__ import annotations

import pytest

from app.agent.graph import build_graph
from app.core.qdrant_client import collection_name_for


@pytest.mark.slow
@pytest.mark.integration
async def test_graph_invoke_grounded_answer(ingested_doc):
    """Drive the full graph against the fixture document: classify ->
    retrieve -> parent_fetch -> assess -> generate. The query asks about
    something the chunks definitely contain (an AMD CPU model number), so
    the assess node should pass on the first iteration and we get a
    grounded answer with citations.

    This is the highest-fidelity Phase-1 test we have — exercises every
    OpenRouter call (classify, assess, generate) plus Qdrant + Postgres."""
    graph = build_graph()
    initial = {
        "query": "What CPU was used for the inference timing experiments?",
        "original_query": "What CPU was used for the inference timing experiments?",
        "tenant_id": ingested_doc["tenant_id"],
        "collection": collection_name_for(ingested_doc["tenant_id"]),
        "iteration": 0,
        "sub_questions": [],
    }

    state = await graph.ainvoke(initial)

    assert state.get("query_type") in {"simple_factual", "multi_part", "procedural"}
    assert state.get("retrieved_chunks"), "retrieve node produced nothing"
    assert state.get("parent_chunks"), "parent_fetch node produced nothing"

    score = float(state.get("context_score") or 0.0)
    assert score > 0.0, f"assess returned zero score; reason={state.get('context_reason')!r}"

    answer = state.get("final_answer") or ""
    assert answer, "generate node produced empty answer"
    # The chunks contain "AMD EPYC 7763" — a high-context-score answer should
    # actually surface that fact rather than the safety fallback.
    if score >= 7:
        assert "EPYC" in answer or "AMD" in answer or "7763" in answer, (
            f"high-context-score answer doesn't reference the source CPU: {answer!r}"
        )

    citations = state.get("citations") or []
    assert citations, "expected at least one citation"
    cite = citations[0]
    assert cite.get("document_id") == ingested_doc["document_id"]
    assert cite.get("filename") == "sample.pdf"
    assert cite.get("page_num", 0) >= 1


@pytest.mark.slow
@pytest.mark.integration
async def test_graph_classifies_simple_query(ingested_doc):
    """A short factual question should classify as simple_factual and skip
    the decompose node. (Side note: classify is deterministic at temp=0.)"""
    graph = build_graph()
    state = await graph.ainvoke(
        {
            "query": "What CPU was used?",
            "original_query": "What CPU was used?",
            "tenant_id": ingested_doc["tenant_id"],
            "collection": collection_name_for(ingested_doc["tenant_id"]),
            "iteration": 0,
            "sub_questions": [],
        }
    )
    # Either simple_factual or procedural is acceptable; multi_part is wrong.
    assert state.get("query_type") in {"simple_factual", "procedural"}
