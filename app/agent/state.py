"""Shared state for the LangGraph agent.

The graph (see app/agent/graph.py) is a state machine over `AgentState`.
Every node receives the current state and returns a partial-update dict
that LangGraph merges into the next state.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

QueryType = Literal["simple_factual", "multi_part", "procedural"]


class AgentState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────
    query: str  # current query (may be reformulated)
    original_query: str  # the user's original phrasing — kept for citations
    tenant_id: str
    collection: str  # Qdrant collection name (derived from tenant_id at entry)

    # ── Classification + decomposition ──────────────
    query_type: QueryType
    sub_questions: list[str]

    # ── Loop control ────────────────────────────────
    iteration: int  # bumped by reformulate, capped at MAX_RETRIEVAL_ITERATIONS

    # ── Retrieval ───────────────────────────────────
    retrieved_chunks: list[dict[str, Any]]  # children from Qdrant (RetrievedChunk dumps)
    parent_chunks: list[dict[str, Any]]  # 1024-token windows from PG
    context_score: float  # 0..10 from assess node
    context_reason: str  # used by reformulate

    # ── Generation ──────────────────────────────────
    final_answer: str
    citations: list[dict[str, Any]]
    faithfulness_score: float  # Phase 2

    # ── LangGraph plumbing (unused by Phase 1 nodes) ─
    messages: Annotated[list[BaseMessage], add_messages]
