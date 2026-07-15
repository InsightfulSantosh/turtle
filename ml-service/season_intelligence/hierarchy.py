from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class HierarchySpec:
    node_names: tuple[str, ...]
    bottom_names: tuple[str, ...]
    summing_matrix: tuple[tuple[float, ...], ...]


class MinTraceReconciler:
    """Reconciles independently forecast hierarchy nodes into coherent totals."""

    def __init__(self, spec: HierarchySpec, residual_covariance: Sequence[Sequence[float]] | None = None):
        self.spec = spec
        self.summing = np.asarray(spec.summing_matrix, dtype=np.float64)
        if self.summing.shape != (len(spec.node_names), len(spec.bottom_names)):
            raise ValueError("summing matrix shape does not match the hierarchy")
        covariance = (
            np.asarray(residual_covariance, dtype=np.float64)
            if residual_covariance is not None
            else np.eye(len(spec.node_names))
        )
        if covariance.shape != (len(spec.node_names), len(spec.node_names)):
            raise ValueError("residual covariance shape does not match the hierarchy")
        inverse = np.linalg.pinv(covariance)
        self.projection = np.linalg.pinv(self.summing.T @ inverse @ self.summing) @ self.summing.T @ inverse

    def reconcile(self, base_forecasts: Mapping[str, float]) -> dict[str, float]:
        missing = set(self.spec.node_names) - set(base_forecasts)
        if missing:
            raise ValueError(f"missing base forecasts for: {', '.join(sorted(missing))}")
        base = np.asarray([base_forecasts[name] for name in self.spec.node_names], dtype=np.float64)
        bottom = np.maximum(self.projection @ base, 0.0)
        coherent = self.summing @ bottom
        return {name: float(value) for name, value in zip(self.spec.node_names, coherent, strict=True)}
