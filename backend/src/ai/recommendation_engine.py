"""Application-facing orchestration for the current recommendation model."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from machine_learning.model import (
    attribute_similarity,
    combined_similarity,
    demand_uncertainty,
    fit_demand_pipeline,
    recommend_one,
)


class RecommendationRuntime:
    """Loads one artifact and coordinates matching, forecasting and ordering."""

    def __init__(self, artifact_path: Path):
        self.path = artifact_path
        self.loaded_at = time.time()
        self.artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.meta = self.artifact["meta"]
        self.model = dict(self.meta["model"])
        self.attribute_weights = {
            str(name): float(weight)
            for name, weight in self.model.get("attributeWeights", {}).items()
        } or None
        self.history = self.artifact["historical"]
        self.targets = np.asarray(
            [float(item["salesTarget"]) for item in self.history],
            dtype=float,
        )
        self.demand_pipeline = fit_demand_pipeline(
            self.history,
            self.targets,
            float(self.model["ridgeAlpha"]),
        )

    def recommend(
        self,
        product: Mapping[str, Any],
        *,
        target_sell_through: float = 0.70,
        visual_similarities: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        item = dict(product)
        visual_scores = visual_similarities or {}
        matches: list[dict[str, Any]] = []
        attribute_weight = float(self.model["attributeWeight"])

        for historical in self.history:
            attribute, breakdown = attribute_similarity(
                item,
                historical,
                self.attribute_weights,
            )
            visual = visual_scores.get(historical["id"])
            hybrid = combined_similarity(attribute, visual, attribute_weight)
            matches.append({
                "historicalId": historical["id"],
                "attributeScore": round(attribute, 4),
                "visualScore": visual,
                "hybridScore": round(hybrid, 4),
                "attributeBreakdown": breakdown,
            })

        matches.sort(key=lambda match: match["hybridScore"], reverse=True)
        result = recommend_one(
            item,
            self.history,
            matches,
            self.targets,
            self.demand_pipeline,
            dict(self.model),
        )

        for source, destination in (
            ("expectedSales", "quantity"),
            ("salesLow", "low"),
            ("salesHigh", "high"),
            ("analogueSales", "analogueQuantity"),
            ("regressionSales", "regressionQuantity"),
            ("salesIntervalHalfWidth", "intervalHalfWidth"),
        ):
            result[destination] = max(
                100,
                min(
                    2_000,
                    int(round((result[source] / target_sell_through) / 25) * 25),
                ),
            )

        result["uncertaintyRatio"] = round(
            result["salesIntervalHalfWidth"] / max(result["expectedSales"], 1),
            4,
        )
        result["demandUncertainty"] = demand_uncertainty(
            result["expectedSales"],
            result["salesIntervalHalfWidth"],
        )

        return {
            "requestId": str(uuid.uuid4()),
            "productId": item["id"],
            "modelVersion": self.model["version"],
            "recommendation": result,
            "matches": matches[: int(self.model["topK"])],
            "warnings": ["attribute_only"] if not visual_scores else [],
        }
