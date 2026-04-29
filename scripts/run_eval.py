"""Batch RAGAS eval against a curated questions file.

Reads `data/eval/questions.json` (schema: list of {id, tenant_id, query,
ground_truth}), runs each question through the live agent (Python API,
not HTTP — no Logto round-trip), scores with RAGAS, prints aggregate
stats, and writes per-row results to `eval_logs` so the same dashboards
that track live traffic also see the curated baseline.

The trace_id pattern for batch rows is `eval-batch-<unix_ts>-<question_id>`
so they don't collide with live trace ids.

Usage:
    uv run python scripts/run_eval.py \\
        --questions data/eval/questions.json \\
        [--output data/eval/results-<ts>.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

# Make the repo root importable when running as `python scripts/run_eval.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import build_graph
from app.core.qdrant_client import collection_name_for
from app.db.models import EvalLog
from app.db.session import async_session_maker
from app.observability.ragas_eval import score as ragas_score


async def _run_one(
    graph, question: dict, batch_id: str
) -> dict:
    qid = str(question.get("id"))
    tenant_id = question["tenant_id"]
    query = question["query"]
    ground_truth = question.get("ground_truth")

    initial = {
        "query": query,
        "original_query": query,
        "tenant_id": tenant_id,
        "collection": collection_name_for(tenant_id),
        "iteration": 0,
        "sub_questions": [],
    }

    t0 = time.time()
    state = await graph.ainvoke(initial)
    elapsed = time.time() - t0

    answer = state.get("final_answer") or ""
    contexts = [
        pc.get("text", "")
        for pc in (state.get("parent_chunks") or [])
        if pc.get("text")
    ]

    scores = await ragas_score(
        query, answer, contexts, ground_truth=ground_truth
    )

    trace_id = f"{batch_id}-{qid}"

    async with async_session_maker() as session:
        session.add(
            EvalLog(
                trace_id=trace_id,
                tenant_id=tenant_id,
                query=query,
                faithfulness=scores.get("faithfulness"),
                answer_relevancy=scores.get("answer_relevancy"),
                context_precision=scores.get("context_precision"),
                context_recall=scores.get("context_recall"),
                context_score=state.get("context_score"),
            )
        )
        await session.commit()

    return {
        "id": qid,
        "tenant_id": tenant_id,
        "query": query,
        "ground_truth": ground_truth,
        "answer": answer,
        "context_score": state.get("context_score"),
        "iteration": int(state.get("iteration") or 0),
        "elapsed_s": round(elapsed, 2),
        "scores": scores,
        "trace_id": trace_id,
    }


async def _main(questions_path: Path, output_path: Path | None) -> None:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise SystemExit("questions file must be a non-empty JSON array")

    batch_id = f"eval-batch-{int(time.time())}"
    print(f"[batch] {batch_id}  questions={len(questions)}")

    graph = build_graph()
    results: list[dict] = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q.get('id')}: {q.get('query', '')[:60]}...")
        result = await _run_one(graph, q, batch_id)
        s = result["scores"]
        print(
            f"     score={result['context_score']:.1f}  "
            f"faith={_fmt(s['faithfulness'])}  "
            f"answer_rel={_fmt(s['answer_relevancy'])}  "
            f"ctx_prec={_fmt(s['context_precision'])}  "
            f"ctx_recall={_fmt(s['context_recall'])}  "
            f"({result['elapsed_s']}s)"
        )
        results.append(result)

    # Aggregate
    print()
    print("[aggregate]")
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        vals = [r["scores"][k] for r in results if r["scores"][k] is not None]
        if vals:
            print(f"  {k:<22}  mean={statistics.mean(vals):.3f}  n={len(vals)}")
        else:
            print(f"  {k:<22}  no values")
    print(f"  context_score (LLM)     mean={statistics.mean([r['context_score'] or 0 for r in results]):.2f}")
    print(f"  elapsed_s               mean={statistics.mean([r['elapsed_s'] for r in results]):.2f}")

    if output_path is None:
        output_path = questions_path.parent / f"results-{batch_id}.json"
    output_path.write_text(
        json.dumps({"batch_id": batch_id, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[written] {output_path}")


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "  -  "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions", default="data/eval/questions.json", type=Path,
        help="Path to JSON list of {id, tenant_id, query, ground_truth} objects",
    )
    ap.add_argument(
        "--output", default=None, type=Path,
        help="Where to write the results JSON. Defaults next to --questions.",
    )
    args = ap.parse_args()
    if not args.questions.exists():
        raise SystemExit(f"questions file not found: {args.questions}")
    asyncio.run(_main(args.questions, args.output))


if __name__ == "__main__":
    main()
