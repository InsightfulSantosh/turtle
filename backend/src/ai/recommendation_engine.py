"""Application-facing orchestration for the current recommendation model."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from machine_learning.model import recommend_one


class RecommendationRuntime:
    """Loads one artifact and coordinates matching, forecasting and ordering."""

    def __init__(self, artifact_path: Path):
        self.path = artifact_path
        self.loaded_at = time.time()
        self.artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.meta = self.artifact["meta"]
        self.model = dict(self.meta["model"])
        self.history = self.artifact["historical"]

    def recommend(
        self,
        product: Mapping[str, Any],
        *,
        target_sell_through: float = 0.70,
        minimum_visual_score: float = 0.50,
        visual_similarities: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        item = dict(product)
        visual_scores = visual_similarities or {}
        matches: list[dict[str, Any]] = []
        for historical in self.history:
            if str(historical.get("itemType")) != str(item.get("itemType")):
                continue
            visual = visual_scores.get(historical["id"])
            if visual is None:
                continue
            matches.append(
                {
                    "historicalId": historical["id"],
                    "visualScore": visual,
                    "fashionVisualScore": None,
                    "dinoVisualScore": None,
                    "colourVisualScore": None,
                    "textureVisualScore": None,
                    "hybridScore": round(float(visual), 4),
                }
            )

        matches.sort(key=lambda match: match["hybridScore"], reverse=True)
        result = recommend_one(
            item,
            self.history,
            matches,
            self.model,
            target_sell_through=target_sell_through,
            minimum_visual_score=minimum_visual_score,
        )
        no_suitable_match = bool(result["noSuitableMatch"])
        warnings = ["no_visual_evidence"] if not visual_scores else []
        if no_suitable_match:
            warnings.append("no_suitable_visual_match")

        return {
            "requestId": str(uuid.uuid4()),
            "productId": item["id"],
            "modelVersion": self.model["version"],
            "recommendation": result,
            "matches": matches[:3],
            "warnings": warnings,
        }
