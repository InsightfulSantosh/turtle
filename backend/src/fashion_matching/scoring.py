from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SignalWeights:
    image: float = 0.70
    attributes: float = 0.20
    text: float = 0.10

    def __post_init__(self) -> None:
        values = (self.image, self.attributes, self.text)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("signal weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one signal weight must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "image": self.image,
            "attributes": self.attributes,
            "text": self.text,
        }


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("embedding must have a finite, non-zero L2 norm")
    return [float(value / norm) for value in vector]


def cosine_to_unit_interval(score: float) -> float:
    if not math.isfinite(score):
        raise ValueError("similarity score must be finite")
    return min(max((score + 1.0) / 2.0, 0.0), 1.0)


def canonical_attribute(value: str) -> str:
    return " ".join(value.upper().replace("–", "-").split())


def attribute_similarity(
    query: Mapping[str, str],
    candidate: Mapping[str, str],
) -> float | None:
    shared = sorted(set(query) & set(candidate))
    comparisons = [
        canonical_attribute(query[name]) == canonical_attribute(candidate[name])
        for name in shared
        if query[name].strip() and candidate[name].strip()
    ]
    if not comparisons:
        return None
    return sum(comparisons) / len(comparisons)


def fuse_signals(
    signals: Mapping[str, float | None],
    weights: SignalWeights,
) -> tuple[float, dict[str, float]]:
    configured = weights.as_dict()
    active: dict[str, tuple[float, float]] = {}
    for name, score in signals.items():
        weight = configured.get(name, 0.0)
        if score is None or weight <= 0:
            continue
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{name} score must be between 0 and 1")
        active[name] = (score, weight)
    total_weight = sum(weight for _, weight in active.values())
    if total_weight <= 0:
        raise ValueError("no scoring signal is available")
    applied = {name: weight / total_weight for name, (_, weight) in active.items()}
    score = sum(active[name][0] * applied[name] for name in active)
    return score, applied
