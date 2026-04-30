"""Profile a chat request end-to-end: per-node total time, time-in-LLM, gap.

Drives the LangGraph agent the same way the SSE endpoint does
(astream_events v2) and records wall-clock timestamps for every event.
Per-node timing comes from on_chain_start / on_chain_end pairs;
time-in-LLM comes from on_chat_model_start / on_chat_model_end.

The "gap" column is per-node total minus LLM time — that's where
DB / Qdrant / Python work shows up. Parent-chunk fetch (Postgres on RDS,
~250ms RTT each) and embedding calls (also network-bound) are the usual
suspects.

Run inside the api container:

    docker compose exec api uv run python scripts/profile_chat.py
    docker compose exec api uv run python scripts/profile_chat.py --query "summarise" --repeat 3
    docker compose exec api uv run python scripts/profile_chat.py --json > runs.json

What's measurable from astream_events:
- Each node's start/end (so total per-node time, accurate).
- Each LLM call's start/end (so total time-in-LLM per node, accurate).
- The gap = network + DB + Python. CANNOT distinguish OpenRouter network
  latency vs token-generation time without deeper instrumentation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.graph import build_graph  # noqa: E402
from app.core.qdrant_client import collection_name_for  # noqa: E402

# Friendly names mirror chat.py's _NODE_MESSAGES — keep in sync if that grows.
_KNOWN_NODES = {
    "classify",
    "chat_smalltalk",
    "bypass",
    "decompose",
    "retrieve",
    "rerank",
    "parent_fetch",
    "assess",
    "reformulate",
    "generate",
    "faithfulness",
}


async def profile_one(query: str, tenant_id: str) -> dict[str, Any]:
    """One request → returns {total_ms, per_node: {node: {total, llm, gap}}, route}."""
    graph = build_graph()
    initial: dict[str, Any] = {
        "query": query,
        "original_query": query,
        "tenant_id": tenant_id,
        "collection": collection_name_for(tenant_id),
        "iteration": 0,
        "sub_questions": [],
    }

    node_starts: dict[str, float] = {}
    node_totals_ms: dict[str, float] = defaultdict(float)
    node_llm_ms: dict[str, float] = defaultdict(float)
    # Time-in-LLM is attributed to the most-recently-entered node (the LLM call
    # always happens inside a node).
    active_node_stack: list[str] = []
    llm_start: float | None = None
    saw_retrieve = False

    t0 = time.perf_counter()
    async for ev in graph.astream_events(initial, version="v2"):
        kind = ev.get("event", "")
        name = ev.get("name", "")
        now = time.perf_counter()

        if kind == "on_chain_start" and name in _KNOWN_NODES:
            node_starts[name] = now
            active_node_stack.append(name)
            if name in {"retrieve", "rerank", "parent_fetch", "assess", "generate"}:
                saw_retrieve = True
        elif kind == "on_chain_end" and name in _KNOWN_NODES:
            start = node_starts.pop(name, None)
            if start is not None:
                node_totals_ms[name] += (now - start) * 1000
            if active_node_stack and active_node_stack[-1] == name:
                active_node_stack.pop()
        elif kind == "on_chat_model_start":
            llm_start = now
        elif kind == "on_chat_model_end" and llm_start is not None:
            dur_ms = (now - llm_start) * 1000
            attribute_to = active_node_stack[-1] if active_node_stack else "<orphan>"
            node_llm_ms[attribute_to] += dur_ms
            llm_start = None

    total_ms = (time.perf_counter() - t0) * 1000

    per_node = {}
    for node in node_totals_ms:
        total = node_totals_ms[node]
        llm = node_llm_ms.get(node, 0.0)
        per_node[node] = {
            "total_ms": round(total, 1),
            "llm_ms": round(llm, 1),
            "gap_ms": round(max(0.0, total - llm), 1),
        }

    route = (
        "chat_smalltalk" if "chat_smalltalk" in node_totals_ms
        else ("rag" if saw_retrieve else "bypass")
    )
    return {"total_ms": round(total_ms, 1), "per_node": per_node, "route": route}


def print_run(idx: int, run: dict[str, Any]) -> None:
    header = f"\n=== Run {idx}  total={run['total_ms']:>7.0f}ms  route={run['route']} ==="
    print(header)
    print(f"  {'node':<18} {'total':>10} {'llm':>10} {'gap':>10}")
    print(f"  {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 10}")
    # Sort nodes by total time desc — biggest contributors first.
    rows = sorted(run["per_node"].items(), key=lambda kv: -kv[1]["total_ms"])
    for node, t in rows:
        print(
            f"  {node:<18} {t['total_ms']:>9.0f}ms {t['llm_ms']:>9.0f}ms {t['gap_ms']:>9.0f}ms"
        )


def print_aggregate(runs: list[dict[str, Any]]) -> None:
    print("\n=== Aggregate (median of {} runs) ===".format(len(runs)))
    totals = [r["total_ms"] for r in runs]
    print(f"  total_ms        : median={statistics.median(totals):.0f}  "
          f"min={min(totals):.0f}  max={max(totals):.0f}")

    # Per-node medians across runs.
    by_node: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"total": [], "llm": [], "gap": []}
    )
    for r in runs:
        for node, t in r["per_node"].items():
            by_node[node]["total"].append(t["total_ms"])
            by_node[node]["llm"].append(t["llm_ms"])
            by_node[node]["gap"].append(t["gap_ms"])

    print(f"\n  {'node':<18} {'total (med)':>14} {'llm (med)':>14} {'gap (med)':>14}")
    print(f"  {'-' * 18} {'-' * 14} {'-' * 14} {'-' * 14}")
    rows = sorted(by_node.items(), key=lambda kv: -statistics.median(kv[1]["total"]))
    for node, vals in rows:
        print(
            f"  {node:<18} "
            f"{statistics.median(vals['total']):>13.0f}ms "
            f"{statistics.median(vals['llm']):>13.0f}ms "
            f"{statistics.median(vals['gap']):>13.0f}ms"
        )


async def run(query: str, tenant_id: str, repeat: int, as_json: bool) -> int:
    runs: list[dict[str, Any]] = []
    for i in range(1, repeat + 1):
        if not as_json:
            print(f"Profiling run {i}/{repeat}: query={query!r}, tenant={tenant_id}",
                  flush=True)
        runs.append(await profile_one(query, tenant_id))
        if not as_json:
            print_run(i, runs[-1])

    if as_json:
        print(json.dumps({"query": query, "tenant_id": tenant_id, "runs": runs}, indent=2))
    else:
        print_aggregate(runs)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="what is this corpus about?")
    parser.add_argument("--tenant-id", default="s1gpeu5ksyeb")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON instead of a human table.")
    args = parser.parse_args()
    return asyncio.run(run(args.query, args.tenant_id, args.repeat, args.json))


if __name__ == "__main__":
    raise SystemExit(main())
