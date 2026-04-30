"""Eval the classify node against a labelled set.

Run inside the api container so OpenRouter creds + DB connection are available:

    docker compose exec api uv run python scripts/eval_classifier.py

Reports per-item pass/fail, per-category precision/recall, a confusion
matrix, and an overall accuracy. Exits non-zero if accuracy < threshold so
this script doubles as a CI gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

# Make `app.*` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.nodes.classify import classify_query  # noqa: E402

CATEGORIES = ["conversational", "simple_factual", "multi_part", "procedural"]
DEFAULT_EVAL = Path(__file__).parent / "data" / "classifier_eval.json"
DEFAULT_THRESHOLD = 0.85


async def classify_one(query: str, tenant_id: str) -> str:
    state = {"query": query, "original_query": query, "tenant_id": tenant_id}
    result = await classify_query(state)  # type: ignore[arg-type]
    return result.get("query_type", "")


async def run(eval_path: Path, tenant_id: str, threshold: float) -> int:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    items = payload["items"]

    correct = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    misses: list[tuple[str, str, str]] = []

    print(f"Evaluating {len(items)} items against tenant {tenant_id}...\n")
    for i, item in enumerate(items, 1):
        query, expected = item["query"], item["expected"]
        try:
            actual = await classify_one(query, tenant_id)
        except Exception as exc:
            print(f"[{i:>2}] ERROR  {query!r}: {exc}")
            confusion[expected]["<error>"] += 1
            continue
        ok = actual == expected
        marker = "PASS" if ok else "FAIL"
        if ok:
            correct += 1
        else:
            misses.append((query, expected, actual))
        print(f"[{i:>2}] {marker}   expected={expected:<16} actual={actual:<16}  {query!r}")
        confusion[expected][actual] += 1

    total = len(items)
    accuracy = correct / total if total else 0.0

    print("\n=== Per-category breakdown ===")
    for cat in CATEGORIES:
        row = confusion.get(cat, {})
        total_cat = sum(row.values())
        hit = row.get(cat, 0)
        recall = hit / total_cat if total_cat else 0.0
        # Precision: of items predicted as `cat`, how many actually were?
        predicted_cat = sum(confusion[true].get(cat, 0) for true in CATEGORIES)
        precision = hit / predicted_cat if predicted_cat else 0.0
        print(f"  {cat:<16} recall={recall:.2f}  precision={precision:.2f}  (n={total_cat})")

    print("\n=== Confusion matrix (rows=expected, cols=actual) ===")
    cols = CATEGORIES + ["<error>"]
    header = " " * 18 + " ".join(f"{c:<16}" for c in cols)
    print(header)
    for true in CATEGORIES:
        row_cells = [f"{confusion[true].get(pred, 0):<16}" for pred in cols]
        print(f"  {true:<16} " + " ".join(row_cells))

    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")
    if misses:
        print("\nMisses:")
        for q, exp, act in misses:
            print(f"  - {q!r}\n      expected={exp}, actual={act}")

    if accuracy < threshold:
        print(f"\nFAIL: accuracy {accuracy:.2%} below threshold {threshold:.2%}")
        return 1
    print(f"\nPASS: accuracy {accuracy:.2%} >= threshold {threshold:.2%}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--tenant-id", default="s1gpeu5ksyeb",
                        help="Used for the token-count DB query in classify; pass any existing tenant.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Fail if overall accuracy is below this fraction (0-1).")
    args = parser.parse_args()
    return asyncio.run(run(args.eval_set, args.tenant_id, args.threshold))


if __name__ == "__main__":
    raise SystemExit(main())
