from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from domain.contracts import Candidate, RankedCandidate
from machine_learning.model import attribute_similarity


RANK_FEATURES = (
    "vector_similarity",
    "attribute_similarity",
    "demand_reliability",
    "planner_feedback",
)


class CandidateReranker:
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.model_name = "transparent-hybrid-fallback"
        if model_path and Path(model_path).exists():
            try:
                from catboost import CatBoostRanker
            except ImportError as exc:  # pragma: no cover - scale image only
                raise RuntimeError("CatBoost model supplied but catboost is not installed") from exc
            model = CatBoostRanker()
            model.load_model(model_path)
            self.model = model
            self.model_name = f"catboost-ranker:{Path(model_path).name}"

    def rank(
        self,
        product: Mapping[str, Any],
        candidates: Sequence[Candidate],
        top_k: int = 10,
    ) -> list[RankedCandidate]:
        feature_rows: list[dict[str, float]] = []
        for candidate in candidates:
            attribute, _ = attribute_similarity(dict(product), candidate.item)
            feature_rows.append({
                "vector_similarity": candidate.vector_similarity,
                "attribute_similarity": attribute,
                "demand_reliability": min(candidate.seasons_observed / 3, 1.0),
                "planner_feedback": min(max(candidate.feedback_score, 0.0), 1.0),
            })
        matrix = np.asarray([[row[name] for name in RANK_FEATURES] for row in feature_rows], dtype=np.float64)
        if self.model is not None and len(matrix):
            scores = np.asarray(self.model.predict(matrix), dtype=np.float64)
            scores = 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
        else:
            weights = np.asarray([0.54, 0.30, 0.08, 0.08], dtype=np.float64)
            scores = matrix @ weights if len(matrix) else np.asarray([])
        ranked = [RankedCandidate(candidate, float(score), row["attribute_similarity"], row)
                  for candidate, score, row in zip(candidates, scores, feature_rows, strict=True)]
        ranked.sort(key=lambda value: value.score, reverse=True)
        return ranked[: min(max(top_k, 1), 50)]
