"""End-to-end retrieval + answer eval for the agentic-rag pipeline.

Drives the LangGraph agent synchronously (no SSE) for each labelled item and
scores three deterministic metrics — no LLM-as-judge, no extra cost:

  citation_recall  : fraction of expected (filename, page) pairs that show up
                     in the answer's citations. 1.0 = all hit.
  answer_contains  : 1 if the canonical phrase is a case-insensitive substring
                     of the final answer, else 0. Skipped when expected_answer
                     is empty (cosmetic items kept only for citation grading).
  refused_correctly: for negative items (expected_answer is null), 1 if the
                     answer matches the canonical refusal string.

Run inside the api container so the OpenRouter / Postgres / Qdrant clients
resolve:

    docker compose exec api uv run python scripts/eval_retrieval.py
    docker compose exec api uv run python scripts/eval_retrieval.py --verbose
    docker compose exec api uv run python scripts/eval_retrieval.py --recall-threshold 0.6

Exits non-zero when mean citation_recall < --recall-threshold OR refusal
accuracy < --refusal-threshold so this doubles as a CI gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Make `app.*` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.graph import build_graph  # noqa: E402
from app.core.qdrant_client import collection_name_for  # noqa: E402

REFUSAL_PHRASE = "documents don't contain enough information"
DEFAULT_EVAL = Path(__file__).parent / "data" / "retrieval_eval.json"
DEFAULT_RECALL_THRESHOLD = 0.7
DEFAULT_REFUSAL_THRESHOLD = 0.9


async def run_one(graph, query: str, tenant_id: str) -> tuple[dict[str, Any], float]:
    """Invoke the agent synchronously, return (final_state, latency_ms)."""
    initial: dict[str, Any] = {
        "query": query,
        "original_query": query,
        "tenant_id": tenant_id,
        "collection": collection_name_for(tenant_id),
        "iteration": 0,
        "sub_questions": [],
    }
    t0 = time.perf_counter()
    final = await graph.ainvoke(initial)
    latency_ms = (time.perf_counter() - t0) * 1000
    return final, latency_ms


def score_citation_recall(
    citations: list[dict], expected_filenames: list[str], expected_pages: list[int]
) -> float:
    """Fraction of expected (filename, page) pairs that appear in the citations.

    Pairing rule: if expected_pages has N entries and expected_filenames has
    1 entry, every page is checked against that single filename. If both have
    N entries, they are zipped. This matches how the eval JSON is shaped.
    """
    if not expected_filenames:
        return 1.0  # nothing required, perfect score

    cited_pairs = {
        ((c.get("filename") or "").strip(), int(c.get("page_num") or 0))
        for c in citations
    }
    pages = expected_pages or [None]  # type: ignore[list-item]
    if len(expected_filenames) == 1:
        expected_pairs = {(expected_filenames[0], p) for p in pages if p is not None}
    elif len(pages) == 1:
        expected_pairs = {(fn, pages[0]) for fn in expected_filenames if pages[0] is not None}
    else:
        expected_pairs = {(fn, p) for fn, p in zip(expected_filenames, pages) if p is not None}

    if not expected_pairs:
        # Filenames asserted but no specific pages — credit any cite of those filenames.
        cited_filenames = {fn for fn, _ in cited_pairs}
        hits = sum(1 for fn in expected_filenames if fn in cited_filenames)
        return hits / len(expected_filenames)

    hits = len(expected_pairs & cited_pairs)
    return hits / len(expected_pairs)


def detect_route(state: dict[str, Any]) -> str:
    """Best-effort route label from the final state. We don't get the path
    explicitly, but the presence of retrieval-only fields is a tell."""
    qt = state.get("query_type", "")
    if qt == "conversational":
        return "chat_smalltalk"
    if state.get("retrieved_chunks"):
        return "rag"
    return "bypass"


async def run(
    eval_path: Path,
    tenant_id: str,
    recall_threshold: float,
    refusal_threshold: float,
    verbose: bool,
) -> int:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    items = payload["items"]

    print(f"Evaluating {len(items)} items against tenant {tenant_id}...\n")
    graph = build_graph()

    recalls: list[float] = []
    contains_results: list[bool] = []  # only for items with non-empty expected_answer
    refusals: list[bool] = []  # only for negative items
    latencies: list[float] = []
    routes = Counter[str]()
    failures: list[tuple[str, dict[str, Any]]] = []

    for i, item in enumerate(items, 1):
        qid = item.get("id") or f"item-{i}"
        query = item["query"]
        expected_answer = item.get("expected_answer")
        expected_filenames: list[str] = item.get("expected_filenames", [])
        expected_pages: list[int] = item.get("expected_pages", [])
        is_negative = expected_answer is None

        try:
            final, latency_ms = await run_one(graph, query, tenant_id)
        except Exception as exc:
            print(f"[{i:>2}] ERROR  id={qid:<28}  {exc}")
            failures.append((qid, {"error": str(exc)}))
            continue

        latencies.append(latency_ms)
        answer = (final.get("final_answer") or "").strip()
        citations = list(final.get("citations") or [])
        route = detect_route(final)
        routes[route] += 1

        recall = score_citation_recall(citations, expected_filenames, expected_pages)
        recalls.append(recall)

        contains = None
        if not is_negative and expected_answer:
            contains = expected_answer.lower() in answer.lower()
            contains_results.append(contains)

        refused = None
        if is_negative:
            refused = REFUSAL_PHRASE.lower() in answer.lower()
            refusals.append(refused)

        item_passed = (
            (is_negative and refused)
            or (not is_negative and recall >= recall_threshold and (contains is None or contains))
        )
        marker = "PASS" if item_passed else "FAIL"
        contains_str = "n/a" if contains is None else ("y" if contains else "n")
        refused_str = "n/a" if refused is None else ("y" if refused else "n")
        print(
            f"[{i:>2}] {marker}   id={qid:<28}  "
            f"recall={recall:.2f}  contains={contains_str:<3} refused={refused_str:<3} "
            f"route={route:<14}  latency={int(latency_ms):>5}ms  "
            f"q={query!r}"
        )
        if not item_passed:
            failures.append((qid, {"answer": answer[:200], "citations": citations[:5]}))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  items evaluated      : {len(latencies)}")
    if recalls:
        mean_recall = sum(recalls) / len(recalls)
        print(f"  mean citation_recall : {mean_recall:.3f}  (threshold {recall_threshold})")
    else:
        mean_recall = 0.0
        print("  mean citation_recall : n/a")
    if contains_results:
        contains_acc = sum(contains_results) / len(contains_results)
        print(f"  answer_contains acc  : {contains_acc:.3f}  ({sum(contains_results)}/{len(contains_results)})")
    if refusals:
        refusal_acc = sum(refusals) / len(refusals)
        print(f"  refusal accuracy     : {refusal_acc:.3f}  ({sum(refusals)}/{len(refusals)})  (threshold {refusal_threshold})")
    else:
        refusal_acc = 1.0
        print("  refusal accuracy     : n/a (no negative items)")
    if latencies:
        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[max(0, int(len(sorted_lat) * 0.95) - 1)]
        print(f"  latency p50 / p95    : {int(p50)}ms / {int(p95)}ms")
    print(f"  routes               : {dict(routes)}")

    if verbose and failures:
        print("\n=== Failure detail ===")
        for qid, payload in failures:
            print(f"  - {qid}")
            for k, v in payload.items():
                print(f"      {k}: {v}")

    # ── Gate ──────────────────────────────────────────────────────────────
    failed_recall = bool(recalls) and mean_recall < recall_threshold
    failed_refusal = bool(refusals) and refusal_acc < refusal_threshold
    if failed_recall or failed_refusal:
        reasons = []
        if failed_recall:
            reasons.append(f"recall {mean_recall:.2f} < {recall_threshold}")
        if failed_refusal:
            reasons.append(f"refusal {refusal_acc:.2f} < {refusal_threshold}")
        print(f"\nFAIL: {', '.join(reasons)}")
        return 1
    print("\nPASS: thresholds met")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--tenant-id", default="s1gpeu5ksyeb")
    parser.add_argument("--recall-threshold", type=float, default=DEFAULT_RECALL_THRESHOLD)
    parser.add_argument("--refusal-threshold", type=float, default=DEFAULT_REFUSAL_THRESHOLD)
    parser.add_argument("--verbose", action="store_true",
                        help="Print answer + citations for failed items.")
    args = parser.parse_args()
    return asyncio.run(
        run(
            args.eval_set,
            args.tenant_id,
            args.recall_threshold,
            args.refusal_threshold,
            args.verbose,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
