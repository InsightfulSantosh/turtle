from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate_results(
    results: list[dict[str, Any]],
    labels: list[dict[str, str]],
) -> dict[str, float | int | None]:
    relevant: dict[str, dict[str, float]] = defaultdict(dict)
    expected_no_match: dict[str, bool] = {}
    for row in labels:
        query_id = row.get("query_image_id", "").strip()
        if not query_id:
            raise ValueError("every label row requires query_image_id")
        no_match = row.get("no_match", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        expected_no_match[query_id] = no_match
        product_id = row.get("relevant_product_id", "").strip()
        if product_id:
            relevant[query_id][product_id] = float(row.get("relevance") or 1)

    recalls = {1: [], 3: [], 5: [], 10: []}
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    no_match_values: list[float] = []
    latencies: list[float] = []
    evaluated = 0
    for result in results:
        query_id = str(result["query_image_id"])
        if query_id not in expected_no_match:
            continue
        evaluated += 1
        ranked = [str(match["product_id"]) for match in result.get("matches", [])]
        truth = relevant.get(query_id, {})
        if truth:
            for cutoff in recalls:
                recalls[cutoff].append(float(any(item in truth for item in ranked[:cutoff])))
            first = next(
                (index for index, product_id in enumerate(ranked, start=1) if product_id in truth),
                None,
            )
            reciprocal_ranks.append(1 / first if first else 0.0)
            gains = [truth.get(product_id, 0.0) for product_id in ranked[:5]]
            dcg = sum((2**gain - 1) / math.log2(index + 1) for index, gain in enumerate(gains, start=1))
            ideal = sorted(truth.values(), reverse=True)[:5]
            idcg = sum((2**gain - 1) / math.log2(index + 1) for index, gain in enumerate(ideal, start=1))
            ndcg_values.append(dcg / idcg if idcg else 0.0)
        predicted_no_match = bool(result.get("no_suitable_match"))
        no_match_values.append(float(predicted_no_match == expected_no_match[query_id]))
        latencies.append(float(result.get("processing_time_ms") or 0.0))

    return {
        "evaluated_queries": evaluated,
        "recall_at_1": mean(recalls[1]) if recalls[1] else None,
        "recall_at_3": mean(recalls[3]) if recalls[3] else None,
        "recall_at_5": mean(recalls[5]) if recalls[5] else None,
        "recall_at_10": mean(recalls[10]) if recalls[10] else None,
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else None,
        "ndcg_at_5": mean(ndcg_values) if ndcg_values else None,
        "no_match_accuracy": (mean(no_match_values) if no_match_values else None),
        "mean_latency_ms": mean(latencies) if latencies else 0.0,
        "p50_latency_ms": median(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate labelled fashion matching results.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    with args.labels.open(encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    metrics = evaluate_results(payload["results"], labels)
    rendered = json.dumps(metrics, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
