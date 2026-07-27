"""Refresh recommendation policy fields without regenerating image embeddings."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from core.config import paths
from machine_learning.model import (
    MIN_CONVINCING_VISUAL_SCORE,
    fit_demand_pipeline,
    recommend_one,
)


def refresh_recommendations(path: Path) -> dict[str, Any]:
    """Recompute recommendations from the visual matches already in an artifact."""

    artifact = json.loads(path.read_text(encoding="utf-8"))
    history = artifact["historical"]
    model = artifact["meta"]["model"]
    model.update(
        {
            "minimumVisualScore": MIN_CONVINCING_VISUAL_SCORE,
            "minimumMatchConfidence": "Medium",
            "noMatchPolicy": (
                "Show no product match when the best candidate has low "
                "confidence, lacks a visual score, or falls below the visual "
                "similarity threshold. Use the regression forecast without "
                "analogue blending in that case."
            ),
        }
    )
    targets = np.asarray(
        [float(item["salesTarget"]) for item in history],
        dtype=np.float64,
    )
    demand_pipeline = fit_demand_pipeline(
        history,
        targets,
        float(model["ridgeAlpha"]),
    )

    match_confidence: Counter[str] = Counter()
    demand_uncertainty: Counter[str] = Counter()
    for item in artifact["upcoming"]:
        recommendation = recommend_one(
            item,
            history,
            item["matches"],
            targets,
            demand_pipeline,
            model,
        )
        item["recommendation"] = recommendation
        flags = [flag for flag in item.get("modelFlags", []) if flag != "no_suitable_match"]
        if recommendation["noSuitableMatch"]:
            flags.append("no_suitable_match")
        item["modelFlags"] = flags
        match_confidence[recommendation["matchConfidence"]] += 1
        demand_uncertainty[recommendation["demandUncertainty"]] += 1

    confidence_counts = {confidence: match_confidence[confidence] for confidence in ("High", "Medium", "Low")}
    uncertainty_counts = {
        uncertainty: demand_uncertainty[uncertainty] for uncertainty in ("Narrow", "Moderate", "Wide")
    }
    artifact["meta"]["confidenceCounts"] = confidence_counts
    artifact["meta"]["matchConfidenceCounts"] = confidence_counts
    artifact["meta"]["demandUncertaintyCounts"] = uncertainty_counts

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return artifact


def main() -> None:
    artifact = refresh_recommendations(paths.model_artifact)
    no_match_count = sum(bool(item["recommendation"]["noSuitableMatch"]) for item in artifact["upcoming"])
    print(f"Refreshed {len(artifact['upcoming'])} recommendations; {no_match_count} have no suitable visual match")


if __name__ == "__main__":
    main()
