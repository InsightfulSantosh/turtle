"""Refresh recommendation policy fields without regenerating image embeddings."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from core.config import paths
from machine_learning.model import (
    MIN_CONVINCING_VISUAL_SCORE,
    MODEL_VERSION,
    recommend_one,
)


def refresh_recommendations(path: Path) -> dict[str, Any]:
    """Recompute recommendations from the visual matches already in an artifact."""

    artifact = json.loads(path.read_text(encoding="utf-8"))
    history = artifact["historical"]
    model = artifact["meta"]["model"]
    model.clear()
    model.update(
        {
            "version": MODEL_VERSION,
            "status": "Visual-only single-analogue decision model",
            "algorithm": (
                "FashionSigLIP retrieval + multi-scale DINO, dominant-palette "
                "CIEDE2000 colour and pattern gates + texture reranking"
            ),
            "evidencePolicy": "single_top_visual_analogue",
            "salesPolicy": "Use cleaned sales from the single accepted historical visual analogue",
            "orderPolicy": "Divide the selected analogue's cleaned sales by the target sell-through",
            "noMachineLearningForecast": True,
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

    match_confidence: Counter[str] = Counter()
    for item in artifact["upcoming"]:
        recommendation = recommend_one(
            item,
            history,
            item["matches"],
            model,
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
    audit = artifact["meta"].get("attributeAudit", {})
    audit["activeCount"] = 0
    audit["activeAttributes"] = []
    audit["policy"] = (
        "Workbook attributes are retained only for source-data audit and display; "
        "they are not compared, scored or used for forecasting."
    )
    for historical in artifact["historical"]:
        historical.pop("normalizedDemand", None)
    for section in (artifact["meta"], audit):
        for field in section.get("excludedNonComparisonFields", []):
            if field.get("label") == "Demand outcomes":
                field["reason"] = (
                    "Historical order and sales are retained only as the selected "
                    "analogue's output evidence; they never influence visual matching."
                )
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
    artifact = refresh_recommendations(paths.model_artifact)
    no_match_count = sum(bool(item["recommendation"]["noSuitableMatch"]) for item in artifact["upcoming"])
    print(f"Refreshed {len(artifact['upcoming'])} recommendations; {no_match_count} have no suitable visual match")


if __name__ == "__main__":
    main()
