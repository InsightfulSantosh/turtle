from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Candidate:
    item: dict[str, Any]
    vector_similarity: float
    normalized_demand: float
    seasons_observed: int = 1
    feedback_score: float = 0.5


@dataclass(frozen=True)
class RankedCandidate:
    candidate: Candidate
    score: float
    attribute_score: float
    features: dict[str, float]


@dataclass(frozen=True)
class DemandForecast:
    p10: float
    p50: float
    p90: float
    model_name: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BusinessConstraints:
    pack_size: int = 25
    minimum_order: int = 100
    maximum_order: int = 2000
    unit_cost: float | None = None
    budget: float | None = None
    supplier_capacity: int | None = None


@dataclass(frozen=True)
class OptimizedBuy:
    quantity: int
    low: int
    high: int
    binding_constraints: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScaleRecommendation:
    product_id: str
    request_id: str
    model_version: str
    retrieval_mode: str
    recommendation: OptimizedBuy
    forecast: DemandForecast
    matches: tuple[RankedCandidate, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def product_text(product: Mapping[str, Any]) -> str:
    fields = (
        "itemType",
        "gender",
        "sleeve",
        "provision",
        "pattern",
        "range",
        "fit",
        "fabric",
        "fashion",
        "colour",
    )
    return " | ".join(str(product.get(field) or "") for field in fields)
