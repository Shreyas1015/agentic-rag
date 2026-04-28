"""LangGraph state machine for the agentic-rag flow.

    classify ─┬─ multi_part ──► decompose ─┐
              ├─ simple_factual ───────────┤
              └─ procedural ───────────────┤
                                            ▼
                                          retrieve
                                            │
                                            ▼
                                        parent_fetch
                                            │
                                            ▼
                                          assess
                                          /     \\
                              score >= 7 /       \\ score < 7
                                        ▼         ▼
                                     generate    iter < MAX ?
                                        │       /        \\
                                        │   yes ▼         ▼ no
                                        │  reformulate  generate
                                        │      │            │
                                        │      └► retrieve  │
                                        │                   │
                                        ▼                   ▼
                                       END                 END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agent.nodes.assess import assess_context
from app.agent.nodes.classify import classify_query
from app.agent.nodes.decompose import decompose_question
from app.agent.nodes.generate import generate_answer
from app.agent.nodes.parent_fetch import parent_fetch
from app.agent.nodes.reformulate import reformulate_query
from app.agent.nodes.retrieve import retrieve
from app.agent.state import AgentState
from app.core.config import settings


def _route_after_classify(state: AgentState) -> str:
    return "decompose" if state.get("query_type") == "multi_part" else "retrieve"


def _route_after_assess(state: AgentState) -> str:
    score = float(state.get("context_score", 0.0))
    iteration = int(state.get("iteration", 0))
    if score >= settings.CONTEXT_SCORE_THRESHOLD:
        return "generate"
    if iteration < settings.MAX_RETRIEVAL_ITERATIONS:
        return "reformulate"
    # Best-effort: out of retries — generate from whatever we have.
    return "generate"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify", classify_query)
    g.add_node("decompose", decompose_question)
    g.add_node("retrieve", retrieve)
    g.add_node("parent_fetch", parent_fetch)
    g.add_node("assess", assess_context)
    g.add_node("reformulate", reformulate_query)
    g.add_node("generate", generate_answer)

    g.set_entry_point("classify")

    g.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"decompose": "decompose", "retrieve": "retrieve"},
    )
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "parent_fetch")
    g.add_edge("parent_fetch", "assess")

    g.add_conditional_edges(
        "assess",
        _route_after_assess,
        {"generate": "generate", "reformulate": "reformulate"},
    )
    g.add_edge("reformulate", "retrieve")
    g.add_edge("generate", END)

    return g.compile()
