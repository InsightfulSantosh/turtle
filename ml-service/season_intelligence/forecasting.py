from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from season_intelligence.contracts import DemandForecast, RankedCandidate


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not len(values):
        return 0.0
    order = np.argsort(values)
    ordered_values, ordered_weights = values[order], weights[order]
    cumulative = np.cumsum(ordered_weights) / max(float(ordered_weights.sum()), 1e-9)
    index = min(np.searchsorted(cumulative, quantile, side="left"), len(values) - 1)
    return float(ordered_values[index])


class DemandForecaster:
    """Quantile LightGBM when fitted artifacts exist; analogue quantiles otherwise."""

    def __init__(self, artifact_dir: str | None = None, require_trained: bool = False):
        self.models: dict[str, Any] = {}
        self.feature_names: list[str] = []
        if artifact_dir and Path(artifact_dir).exists():
            manifest_path = Path(artifact_dir) / "manifest.json"
            if manifest_path.exists():
                try:
                    import lightgbm as lgb
                except ImportError as exc:  # pragma: no cover - scale image only
                    raise RuntimeError("LightGBM artifact supplied but lightgbm is not installed") from exc
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.feature_names = list(manifest["featureNames"])
                for quantile in ("p10", "p50", "p90"):
                    self.models[quantile] = lgb.Booster(model_file=str(Path(artifact_dir) / manifest[quantile]))
        if require_trained and len(self.models) != 3:
            raise RuntimeError("trained p10/p50/p90 demand artifacts are required by MODEL_POLICY")

    @property
    def model_name(self) -> str:
        return "lightgbm-quantile" if self.models else "analogue-quantile-fallback"

    def _features(self, product: Mapping[str, Any], matches: Sequence[RankedCandidate]) -> dict[str, float]:
        top = matches[0] if matches else None
        demands = np.asarray([match.candidate.normalized_demand for match in matches], dtype=float)
        return {
            "mrp": float(product.get("mrp") or 0),
            "top_match_score": top.score if top else 0.0,
            "top_attribute_score": top.attribute_score if top else 0.0,
            "top_vector_score": top.candidate.vector_similarity if top else 0.0,
            "analogue_mean": float(demands.mean()) if len(demands) else 0.0,
            "analogue_std": float(demands.std()) if len(demands) else 0.0,
            "analogue_count": float(len(demands)),
        }

    def predict(self, product: Mapping[str, Any], matches: Sequence[RankedCandidate]) -> DemandForecast:
        features = self._features(product, matches)
        if self.models:
            row = np.asarray([[features.get(name, 0.0) for name in self.feature_names]], dtype=np.float64)
            values = {name: max(0.0, float(model.predict(row)[0])) for name, model in self.models.items()}
            ordered = sorted((values["p10"], values["p50"], values["p90"]))
            return DemandForecast(*ordered, model_name=self.model_name)
        valid = [match for match in matches if match.candidate.normalized_demand > 0]
        if not valid:
            return DemandForecast(0, 0, 0, self.model_name, ("no_historical_demand",))
        demands = np.asarray([match.candidate.normalized_demand for match in valid], dtype=np.float64)
        weights = np.asarray([max(match.score, 0.05) ** 2 for match in valid], dtype=np.float64)
        p10 = weighted_quantile(demands, weights, 0.10)
        p50 = weighted_quantile(demands, weights, 0.50)
        p90 = weighted_quantile(demands, weights, 0.90)
        if p10 == p90:
            p10, p90 = p50 * 0.75, p50 * 1.25
        return DemandForecast(p10, p50, p90, self.model_name, ("trained_demand_model_unavailable",))
