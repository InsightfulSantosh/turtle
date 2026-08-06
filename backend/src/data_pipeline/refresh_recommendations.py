"""Refresh recommendation policy fields without regenerating image embeddings."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from core.config import paths
from machine_learning.demand import (
    annotate_history_rates,
    fit_demand_priors,
    serialize_priors,
)
from machine_learning.model import (
    MIN_CONVINCING_VISUAL_SCORE,
    MODEL_VERSION,
    recommend_one,
)


def backfill_exposure(history: list[dict[str, Any]], processed_csv: Path) -> int:
    """Add missing exposure fields to an artifact built before they existed.

    Without ``ageingDays`` every product falls back to the same assumed
    selling window, which quietly disables the exposure normalisation rather
    than failing loudly. Joining them back from the cleaned CSV keeps an
    existing artifact usable without re-running image embeddings.
    """

    if not processed_csv.exists():
        return 0
    exposure: dict[str, tuple[int, float]] = {}
    with processed_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                exposure[row["product_id"]] = (
                    int(round(float(row["ageing_days"]))),
                    float(row["weekly_sell_through"]),
                )
            except (KeyError, TypeError, ValueError):
                continue

    filled = 0
    for record in history:
        if float(record.get("ageingDays") or 0) > 0:
            continue
        found = exposure.get(str(record.get("sourceId") or record.get("id")))
        if found is None:
            continue
        record["ageingDays"], record["weeklySellThrough"] = found
        filled += 1
    return filled


def refresh_recommendations(path: Path, *, demand_forecast: bool = False) -> dict[str, Any]:
    """Recompute recommendations from the visual matches already in an artifact.

    ``demand_forecast`` swaps the legacy copy-one-analogue rule for the pooled
    predictive estimator without regenerating any image embeddings.
    """

    artifact = json.loads(path.read_text(encoding="utf-8"))
    history = artifact["historical"]
    if demand_forecast:
        filled = backfill_exposure(
            history,
            paths.repository / "data" / "processed" / "historical_cleaned.csv",
        )
        if filled:
            print(f"Backfilled exposure fields onto {filled} historical products")
    demand_priors = fit_demand_priors(history) if demand_forecast else None
    model = artifact["meta"]["model"]
    model.clear()
    model.update(
        {
            "version": MODEL_VERSION,
            "status": (
                "Visual retrieval + pooled demand forecast"
                if demand_priors is not None
                else "Visual-only single-analogue decision model"
            ),
            "algorithm": (
                "FashionSigLIP retrieval + multi-scale DINO, dominant-palette "
                "CIEDE2000 colour and pattern gates + texture reranking"
            ),
            "evidencePolicy": (
                "pooled_visual_analogue_forecast"
                if demand_priors is not None
                else "single_top_visual_analogue"
            ),
            "salesPolicy": (
                "Similarity-weighted, censoring-corrected demand pooled across accepted "
                "analogues and shrunk toward the item-type/category prior"
                if demand_priors is not None
                else "Use cleaned sales from the single accepted historical visual analogue"
            ),
            "orderPolicy": (
                "Newsvendor buy whose expected sell-through equals the planner target "
                "under the predicted demand distribution"
                if demand_priors is not None
                else "Divide the selected analogue's cleaned sales by the target sell-through"
            ),
            "noMachineLearningForecast": demand_priors is None,
            "noAttributeMatching": True,
            "visualOnlyRanking": True,
            "dinoRerankWeight": 0.58,
            "minimumVisualScore": MIN_CONVINCING_VISUAL_SCORE,
            "minimumMatchConfidence": "Medium",
            "noMatchPolicy": (
                "Return no product match, zero system quantity and manual review when the "
                "single best candidate lacks visual evidence or falls below the threshold."
            ),
            "targetSellThrough": 0.70,
            "topK": 4,
        }
    )
    if demand_priors is not None:
        annotate_history_rates(history, demand_priors)
        model["demandModel"] = serialize_priors(demand_priors)

    match_confidence: Counter[str] = Counter()
    for item in artifact["upcoming"]:
        recommendation = recommend_one(
            item,
            history,
            item["matches"],
            model,
            demand_priors=demand_priors,
        )
        item["recommendation"] = recommendation
        item["matches"] = item["matches"][:4]
        flags = [flag for flag in item.get("modelFlags", []) if flag != "no_suitable_match"]
        if recommendation["noSuitableMatch"]:
            flags.append("no_suitable_match")
        item["modelFlags"] = flags
        match_confidence[recommendation["matchConfidence"]] += 1

    confidence_counts = {confidence: match_confidence[confidence] for confidence in ("High", "Medium", "Low")}
    artifact["meta"]["confidenceCounts"] = confidence_counts
    artifact["meta"]["matchConfidenceCounts"] = confidence_counts
    artifact["meta"].pop("demandUncertaintyCounts", None)
    artifact["meta"].pop("attributeScoreRange", None)
    artifact["meta"].pop("attributeAudit", None)
    for historical in artifact["historical"]:
        historical.pop("normalizedDemand", None)
    for item in artifact["upcoming"]:
        for match in item["matches"]:
            match.pop("attributeScore", None)
            match.pop("attributeBreakdown", None)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return artifact


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demand-forecast",
        action="store_true",
        help="Use the pooled predictive estimator instead of copying one analogue's sales",
    )
    arguments = parser.parse_args()

    artifact = refresh_recommendations(
        paths.model_artifact,
        demand_forecast=arguments.demand_forecast,
    )
    no_match_count = sum(bool(item["recommendation"]["noSuitableMatch"]) for item in artifact["upcoming"])
    policy = artifact["meta"]["model"]["evidencePolicy"]
    print(
        f"Refreshed {len(artifact['upcoming'])} recommendations using {policy}; "
        f"{no_match_count} have no suitable visual match"
    )


if __name__ == "__main__":
    main()
