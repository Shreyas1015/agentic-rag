"""LangGraph state machine for the agentic-rag flow.

    classify
       │ writes tenant_token_count + query_type to state
       │
       ├─── corpus < BYPASS_BUDGET ──► bypass (long-context generate)
       │                                 │
       ├─── multi_part ──► decompose ─┐  │
       ├─── simple_factual ───────────┤  │
       └─── procedural ───────────────┤  │
                                      ▼  │
                                 retrieve (top-30 from Qdrant RRF)
                                      │
                                      ▼
                                   rerank (BGE local, top-30 → top-8)
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
                           faithfulness ◄────────────┘
                                 │
                                 ▼
                                END

bypass: corpus fits in the long-context model's window — skip RAG and
        stuff everything into the prompt. faithfulness still runs.
faithfulness: verifies the answer against parent_chunks; if score < 0.85,
              regenerates ONCE with the unsupported claims excluded.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agent.nodes.assess import assess_context
from app.agent.nodes.bypass import bypass_generate
from app.agent.nodes.classify import classify_query
from app.agent.nodes.decompose import decompose_question
from app.agent.nodes.faithfulness import faithfulness_check
from app.agent.nodes.generate import generate_answer
from app.agent.nodes.parent_fetch import parent_fetch
from app.agent.nodes.reformulate import reformulate_query
from app.agent.nodes.rerank import rerank
from app.agent.nodes.retrieve import retrieve
from app.agent.state import AgentState
from app.core.config import settings


def _route_after_classify(state: AgentState) -> str:
    """Route from classify into one of three lanes:

    - bypass    : tenant's full corpus fits in the long-context budget;
                  skip RAG entirely and stuff everything in the prompt.
    - decompose : compound query; split into sub-questions before retrieve.
    - retrieve  : default RAG path.
    """
    token_count = int(state.get("tenant_token_count") or 0)
    if 0 < token_count < settings.LONG_CONTEXT_BYPASS_TOKEN_BUDGET:
        return "bypass"
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
    g.add_node("bypass", bypass_generate)
    g.add_node("decompose", decompose_question)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank)
    g.add_node("parent_fetch", parent_fetch)
    g.add_node("assess", assess_context)
    g.add_node("reformulate", reformulate_query)
    g.add_node("generate", generate_answer)
    g.add_node("faithfulness", faithfulness_check)

    g.set_entry_point("classify")

    g.add_conditional_edges(
        "classify",
        _route_after_classify,
        {
            "bypass": "bypass",
            "decompose": "decompose",
            "retrieve": "retrieve",
        },
    )
    # Bypass already produced final_answer + citations; faithfulness still
    # runs as a defensive verifier.
    g.add_edge("bypass", "faithfulness")
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "parent_fetch")
    g.add_edge("parent_fetch", "assess")

    g.add_conditional_edges(
        "assess",
        _route_after_assess,
        {"generate": "generate", "reformulate": "reformulate"},
    )
    g.add_edge("reformulate", "retrieve")
    g.add_edge("generate", "faithfulness")
    g.add_edge("faithfulness", END)

    return g.compile()
