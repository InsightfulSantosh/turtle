from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "4.2.0"
DEFAULT_TARGET_SELL_THROUGH = 0.70
PACK_SIZE = 25
MIN_BUY = 100
MAX_BUY = 2_000
MIN_CONVINCING_VISUAL_SCORE = 0.50

DESIGN_FAMILY = {
    "DIGITAL PRINT": "PRINT",
    "DISCHARGE PRINT": "PRINT",
    "PIGMENT PRINT": "PRINT",
    "PRINTS": "PRINT",
    "PRINTS (DISCHARGE & PIGMENT)": "PRINT",
    "PRINTS (PIGMENT)": "PRINT",
    "SOLID": "SOLID",
    "SOLIDS": "SOLID",
    "PLAINS": "SOLID",
    "WBC DOBBY": "TEXTURE",
    "DOBBY/STRUCTURE": "TEXTURE",
}

COLOUR_FAMILY = {
    "NAVY BLUE": "BLUE",
    "SKY BLUE": "BLUE",
    "LIGHT BLUE": "BLUE",
    "PEACOCK BLUE": "BLUE",
    "TEAL": "BLUE",
    "INDIGO": "BLUE",
    "DARK GREEN": "GREEN",
    "LIGHT GREEN": "GREEN",
    "MINT": "GREEN",
    "OLIVE": "GREEN",
    "CREAM": "NEUTRAL",
    "IVORY": "NEUTRAL",
    "BEIGE": "NEUTRAL",
    "KHAKI": "NEUTRAL",
    "STONE": "NEUTRAL",
    "WHITE": "NEUTRAL",
    "GREY": "GREY",
    "DARK GREY": "GREY",
    "LIGHT GREY": "GREY",
    "BLACK": "GREY",
    "MAROON": "RED",
    "CORAL": "RED",
    "RUST": "RED",
    "PINK": "RED",
    "OCHRE": "YELLOW",
    "LEMON": "YELLOW",
}

BASE_ATTRIBUTE_WEIGHTS = {
    "item": 0.16,
    "design": 0.17,
    "category_type": 0.14,
    "fabric": 0.14,
    "colour": 0.09,
}

ATTRIBUTE_SCHEMA = {
    "item": (
        "Item",
        "item_type",
        "item_type",
        "Exact match with a strong cross-item penalty",
    ),
    "design": (
        "Design",
        "design",
        "design",
        "Exact or related design-family match",
    ),
    "category_type": (
        "Category Type",
        "category_type",
        "category_type",
        "Exact Formal, Casual, Denim, or Ceremonial match",
    ),
    "fabric": (
        "Fabric",
        "fabric",
        "fabric",
        "Exact or token-overlap match",
    ),
    "colour": (
        "Colour name",
        "colour",
        "colour",
        "Exact or related colour-family match",
    ),
}

EXCLUDED_CONSTANT_ATTRIBUTES: tuple[dict[str, str], ...] = ()

EXCLUDED_NON_COMPARISON_FIELDS = (
    {
        "label": "Identifiers",
        "historicalColumn": "product_id, style_code",
        "upcomingColumn": "product_id, style_code",
        "reason": "Row and style identifiers identify products; they are not reusable product characteristics.",
    },
    {
        "label": "Colour variant code",
        "historicalColumn": "colour_code",
        "upcomingColumn": "colour_code",
        "reason": "Variant codes such as 1001 are not stable colour meanings; colour names are compared instead.",
    },
    {
        "label": "Season",
        "historicalColumn": "season",
        "upcomingColumn": "season",
        "reason": (
            "Season supports tracing and temporal validation but is outside the five approved product attributes."
        ),
    },
    {
        "label": "Demand outcomes",
        "historicalColumn": ("total_order_quantity, dispatch_quantity, sales_quantity, sell_through"),
        "upcomingColumn": "—",
        "reason": (
            "These fields train and validate the quantity forecast; using them "
            "in product similarity would leak outcomes."
        ),
    },
)


def norm(value: Any) -> str:
    return " ".join(str(value or "").upper().split()).strip()


def attribute_value(item: dict[str, Any], name: str) -> Any:
    if name == "item":
        return norm(item.get("itemType"))
    source_key = {
        "design": "design",
        "category_type": "categoryType",
        "fabric": "fabric",
        "colour": "colour",
    }[name]
    return norm(item.get(source_key))


def populated_attribute_values(items: list[dict[str, Any]], name: str) -> set[Any]:
    return {value for item in items if (value := attribute_value(item, name)) not in ("", 0.0)}


def informative_attribute_weights(
    history: list[dict[str, Any]],
    upcoming: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Drop fields that cannot distinguish historical candidates, then renormalize.

    A field with fewer than two populated historical values is not allowed to
    contribute a constant bonus to every match. This keeps future artifact
    rebuilds safe when a newly supplied workbook contains another constant
    field.
    """

    active = {
        name: weight
        for name, weight in BASE_ATTRIBUTE_WEIGHTS.items()
        if len(populated_attribute_values(history, name)) > 1
        and (upcoming is None or len(populated_attribute_values(upcoming, name)) > 0)
    }
    total = sum(active.values())
    if not active or total <= 0:
        raise ValueError("No informative comparable attributes were found in the historical dataset")
    return {name: weight / total for name, weight in active.items()}


def attribute_audit(
    history: list[dict[str, Any]],
    upcoming: list[dict[str, Any]],
    weights: dict[str, float],
    source_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_meta = source_meta or {}
    column_map = source_meta.get("attributeColumnMap", {})
    active = []
    for name, weight in weights.items():
        label, historical_column, upcoming_column, method = ATTRIBUTE_SCHEMA[name]
        mapped_columns = column_map.get(name, {})
        historical_values = populated_attribute_values(history, name)
        upcoming_values = populated_attribute_values(upcoming, name)
        active.append(
            {
                "key": name,
                "label": label,
                "historicalColumn": mapped_columns.get("historicalColumn", historical_column),
                "upcomingColumn": mapped_columns.get("upcomingColumn", upcoming_column),
                "weight": round(weight, 4),
                "historicalUnique": len(historical_values),
                "upcomingUnique": len(upcoming_values),
                "method": method,
            }
        )
    automatically_excluded = []
    for name in sorted(BASE_ATTRIBUTE_WEIGHTS.keys() - weights.keys()):
        label, historical_column, upcoming_column, _ = ATTRIBUTE_SCHEMA[name]
        mapped_columns = column_map.get(name, {})
        automatically_excluded.append(
            {
                "label": label,
                "historicalColumn": mapped_columns.get("historicalColumn", historical_column),
                "upcomingColumn": mapped_columns.get("upcomingColumn", upcoming_column),
                "reason": (
                    "Automatically excluded because the field is missing from one workbook "
                    "or cannot distinguish historical candidates."
                ),
            }
        )
    excluded_constants = source_meta.get(
        "excludedConstantAttributes",
        list(EXCLUDED_CONSTANT_ATTRIBUTES),
    )
    excluded_non_comparison = source_meta.get(
        "excludedNonComparisonFields",
        list(EXCLUDED_NON_COMPARISON_FIELDS),
    )
    return {
        "historicalSourceRange": source_meta.get(
            "historicalSourceRange",
            f"Sheet1!A1:T{len(history) + 1}",
        ),
        "upcomingSourceRange": source_meta.get(
            "upcomingSourceRange",
            f"Sheet1!A1:N{len(upcoming) + 1}",
        ),
        "activeCount": len(active),
        "activeAttributes": active,
        "excludedConstants": [*excluded_constants, *automatically_excluded],
        "excludedNonComparisonFields": list(excluded_non_comparison),
        "policy": (
            "Only fields populated in both workbooks and informative across historical "
            "candidates contribute to similarity; pair-level missing values are reweighted."
        ),
    }


def token_set(value: Any) -> set[str]:
    ignored = {"100%", "100", "PERCENT", "THE"}
    return {part for part in re.findall(r"[A-Z0-9]+", norm(value)) if part not in ignored}


def jaccard(left: Any, right: Any) -> float:
    a, b = token_set(left), token_set(right)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def categorical(left: Any, right: Any) -> float:
    a, b = norm(left), norm(right)
    return 1.0 if a and a == b else 0.0


def design_similarity(left: Any, right: Any) -> float:
    a, b = norm(left), norm(right)
    if a == b:
        return 1.0
    if DESIGN_FAMILY.get(a, a) == DESIGN_FAMILY.get(b, b):
        return 0.62
    return 0.08 if {a, b} <= {"CHECKS", "STRIPES", "PRINTS", "DIGITAL PRINT"} else 0.0


def colour_similarity(left: Any, right: Any) -> float:
    a, b = norm(left), norm(right)
    if a == b:
        return 1.0
    return 0.66 if COLOUR_FAMILY.get(a, a) == COLOUR_FAMILY.get(b, b) else 0.0


def attribute_similarity(
    left: dict[str, Any],
    right: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    active_weights = weights or BASE_ATTRIBUTE_WEIGHTS
    comparable_weights = {
        name: weight
        for name, weight in active_weights.items()
        if attribute_value(left, name) not in ("", 0.0) and attribute_value(right, name) not in ("", 0.0)
    }
    weight_total = sum(comparable_weights.values())
    if weight_total <= 0:
        return 0.0, {}
    comparable_weights = {name: weight / weight_total for name, weight in comparable_weights.items()}
    item_type = categorical(left.get("itemType"), right.get("itemType"))
    values = {
        "item": item_type,
        "design": design_similarity(left.get("design"), right.get("design")),
        "category_type": categorical(
            left.get("categoryType"),
            right.get("categoryType"),
        ),
        "fabric": max(
            categorical(left.get("fabric"), right.get("fabric")),
            jaccard(left.get("fabric"), right.get("fabric")),
        ),
        "colour": colour_similarity(left.get("colour"), right.get("colour")),
    }
    values = {name: values[name] for name in comparable_weights}
    score = sum(values[name] * weight for name, weight in comparable_weights.items())
    if "item" in comparable_weights and item_type == 0:
        score *= 0.42
    return round(score, 4), {name: round(value, 3) for name, value in values.items()}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def round_pack(value: float, pack: int = PACK_SIZE) -> int:
    return int(round(value / pack) * pack)


def quality_flags(item: dict[str, Any]) -> list[str]:
    order = float(item.get("order") or 0)
    dispatch = float(item.get("dispatch") or 0)
    sales = float(item.get("sales") or 0)
    sell_through = float(item.get("sellThrough") or 0)
    flags: list[str] = []
    if order > 0 and dispatch > order * 1.05:
        flags.append("dispatch_above_order")
    if dispatch > 0 and sales > dispatch * 1.02:
        flags.append("sales_above_dispatch")
    if sell_through > 1.0:
        flags.append("sell_through_above_100")
    if min(order, dispatch, sales, sell_through) < 0:
        flags.append("negative_value")
    return flags


def sales_target(item: dict[str, Any]) -> float:
    """Return cleaned observed unit sales for model training.

    The supplied sample has one row where sales exceed both order and dispatch.
    Until opening inventory and transfer data are available, cap that row at the
    strongest observable supply value so it cannot dominate a 33-row pilot.
    """

    order = max(float(item.get("order") or 0), 0)
    dispatch = max(float(item.get("dispatch") or 0), 0)
    sales = max(float(item.get("sales") or 0), 0)
    supply = max(order, dispatch)
    return min(sales, supply) if supply > 0 else sales


def normalized_demand(item: dict[str, Any], target_sell_through: float = DEFAULT_TARGET_SELL_THROUGH) -> float:
    """Compatibility order proxy derived from the cleaned sales target."""

    return sales_target(item) / max(target_sell_through, 0.01)


@dataclass(frozen=True)
class VisionCalibration:
    median: float
    scale: float
    q10: float
    q90: float

    def similarity(self, distance: float | None) -> float | None:
        if distance is None:
            return None
        exponent = clamp((distance - self.median) / max(self.scale, 1e-6), -30, 30)
        return round(1.0 / (1.0 + math.exp(exponent)), 4)


def calibrate_vision(distances: Iterable[float]) -> VisionCalibration:
    values = np.asarray(list(distances), dtype=np.float64)
    if values.size == 0:
        return VisionCalibration(median=0.5, scale=0.1, q10=0.3, q90=0.7)
    q10, median, q90 = np.percentile(values, [10, 50, 90]).tolist()
    scale = max((q90 - q10) / (2 * math.log(9)), 1e-4)
    return VisionCalibration(float(median), float(scale), float(q10), float(q90))


def combined_similarity(attribute: float, visual: float | None, attribute_weight: float) -> float:
    if visual is None:
        return attribute
    return attribute * attribute_weight + visual * (1.0 - attribute_weight)


def demand_features(item: dict[str, Any]) -> dict[str, float]:
    """Create domain features; sklearn owns vocabulary, scaling and regression."""

    design = norm(item.get("design"))
    colour = norm(item.get("colour"))
    categorical_values = {
        "item_type": norm(item.get("itemType")),
        "design": DESIGN_FAMILY.get(design, design),
        "category_type": norm(item.get("categoryType")),
        "colour": COLOUR_FAMILY.get(colour, colour),
    }
    features = {f"{field}={value or 'UNKNOWN'}": 1.0 for field, value in categorical_values.items()}
    features.update({f"fabric={token}": 1.0 for token in token_set(item.get("fabric"))})
    return features


def build_demand_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("features", DictVectorizer(sparse=True)),
            ("scale", StandardScaler(with_mean=False)),
            ("ridge", Ridge(alpha=alpha, solver="lsqr")),
        ]
    )


def fit_demand_pipeline(
    items: list[dict[str, Any]],
    targets: np.ndarray,
    alpha: float,
) -> Pipeline:
    pipeline = build_demand_pipeline(alpha)
    pipeline.fit([demand_features(item) for item in items], targets)
    return pipeline


def analogue_prediction(
    target_index: int,
    candidate_indices: list[int],
    targets: np.ndarray,
    attribute_matrix: np.ndarray,
    visual_matrix: np.ndarray,
    attribute_weight: float,
    top_k: int,
) -> tuple[float, list[tuple[int, float]]]:
    visual_candidates = [
        index
        for index in candidate_indices
        if not np.isnan(visual_matrix[target_index, index])
    ]
    # In a two-stage index, only FashionSigLIP-shortlisted candidates may be
    # considered by the analogue model. If a query has no image candidate,
    # retain the existing attribute-only fallback.
    if visual_candidates:
        candidate_indices = visual_candidates
    ranked: list[tuple[int, float]] = []
    for index in candidate_indices:
        visual = None if np.isnan(visual_matrix[target_index, index]) else float(visual_matrix[target_index, index])
        score = combined_similarity(float(attribute_matrix[target_index, index]), visual, attribute_weight)
        ranked.append((index, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    selected = ranked[:top_k]
    weights = np.asarray([max(score, 0.01) ** 2 for _, score in selected], dtype=np.float64)
    values = np.asarray([targets[index] for index, _ in selected], dtype=np.float64)
    return float(np.average(values, weights=weights)), selected


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(predicted - actual).sum() / max(np.abs(actual).sum(), 1e-9))


def finite_sample_quantile(values: np.ndarray, coverage: float = 0.80) -> float:
    level = min(math.ceil((len(values) + 1) * coverage) / max(len(values), 1), 1.0)
    try:
        return float(np.quantile(values, level, method="higher"))
    except TypeError:
        return float(np.quantile(values, level, interpolation="higher"))


def season_sort_key(value: Any) -> tuple[int, int, str]:
    normalized = norm(value)
    match = re.search(r"\b(SS|AW)\s*(\d{2,4})\b", normalized)
    if not match:
        return (0, 0, normalized)
    year = int(match.group(2))
    if year < 100:
        year += 2000
    phase = 0 if match.group(1) == "SS" else 1
    return (year, phase, normalized)


def temporal_backtest_model(
    history: list[dict[str, Any]],
    attribute_matrix: np.ndarray,
    visual_matrix: np.ndarray,
    targets: np.ndarray,
    latest_season: str,
) -> dict[str, Any]:
    train_indices = np.asarray(
        [index for index, item in enumerate(history) if norm(item.get("season")) != latest_season],
        dtype=int,
    )
    holdout_indices = np.asarray(
        [index for index, item in enumerate(history) if norm(item.get("season")) == latest_season],
        dtype=int,
    )
    if train_indices.size < 20 or holdout_indices.size < 10:
        raise ValueError("Temporal validation requires at least 20 training and 10 holdout rows")

    candidate_alphas = (0.1, 1.0, 10.0, 100.0)
    train_items = [history[int(index)] for index in train_indices]
    holdout_items = [history[int(index)] for index in holdout_indices]
    ridge_by_alpha: dict[float, np.ndarray] = {}
    for alpha in candidate_alphas:
        pipeline = fit_demand_pipeline(train_items, targets[train_indices], alpha)
        ridge_by_alpha[alpha] = pipeline.predict([demand_features(item) for item in holdout_items])

    has_visual = bool(np.isfinite(visual_matrix).any())
    attribute_weights = [round(value / 10, 1) for value in range(1, 10)] if has_visual else [1.0]
    analogue_predictions: dict[tuple[float, int], np.ndarray] = {}
    for config in ParameterGrid({"attributeWeight": attribute_weights, "topK": [3, 5, 8]}):
        attribute_weight = float(config["attributeWeight"])
        top_k = int(config["topK"])
        predictions = []
        for holdout in holdout_indices:
            prediction, _ = analogue_prediction(
                int(holdout),
                [int(index) for index in train_indices],
                targets,
                attribute_matrix,
                visual_matrix,
                attribute_weight,
                top_k,
            )
            predictions.append(prediction)
        analogue_predictions[(attribute_weight, top_k)] = np.asarray(predictions)

    best: dict[str, Any] | None = None
    search = ParameterGrid(
        {
            "attributeWeight": attribute_weights,
            "regressionBlend": [0.15, 0.25, 0.35, 0.50],
            "ridgeAlpha": list(candidate_alphas),
            "topK": [3, 5, 8],
        }
    )
    holdout_targets = targets[holdout_indices]
    for config in search:
        attribute_weight = float(config["attributeWeight"])
        top_k = int(config["topK"])
        alpha = float(config["ridgeAlpha"])
        regression_blend = float(config["regressionBlend"])
        analogue = analogue_predictions[(attribute_weight, top_k)]
        ridge = ridge_by_alpha[alpha]
        ensemble = analogue * (1 - regression_blend) + ridge * regression_blend
        score = wape(holdout_targets, ensemble)
        candidate = {
            "attributeWeight": attribute_weight,
            "visualWeight": 1 - attribute_weight,
            "topK": top_k,
            "ridgeAlpha": alpha,
            "regressionBlend": regression_blend,
            "predictions": ensemble,
            "score": score,
        }
        if best is None or score < best["score"]:
            best = candidate

    assert best is not None
    predictions = np.asarray(best["predictions"])
    residuals = np.abs(predictions - holdout_targets)
    interval = finite_sample_quantile(residuals, coverage=0.80)
    bias = float((predictions - holdout_targets).sum() / max(holdout_targets.sum(), 1e-9))
    coverage = float(np.mean(residuals <= interval))
    demand_pipeline = fit_demand_pipeline(
        history,
        targets,
        float(best["ridgeAlpha"]),
    )
    training_seasons = sorted({norm(history[int(index)].get("season")) for index in train_indices}, key=season_sort_key)
    return {
        **{key: value for key, value in best.items() if not isinstance(value, np.ndarray)},
        "predictions": predictions,
        "residuals": residuals,
        "conformalHalfWidth": interval,
        "metrics": {
            "wape": round(wape(holdout_targets, predictions), 4),
            "mae": round(float(mean_absolute_error(holdout_targets, predictions)), 1),
            "bias": round(bias, 4),
            "intervalCoverage": round(coverage, 4),
        },
        "demandPipeline": demand_pipeline,
        "selectionMethod": "Temporal holdout + ParameterGrid",
        "evaluation": (f"Forward holdout: {', '.join(training_seasons)} used to predict {latest_season}"),
        "validationRows": int(holdout_indices.size),
        "attributeWeightGrid": attribute_weights,
    }


def backtest_model(
    history: list[dict[str, Any]],
    attribute_matrix: np.ndarray,
    visual_matrix: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    seasons = sorted(
        {norm(item.get("season")) for item in history if norm(item.get("season"))},
        key=season_sort_key,
    )
    if len(history) >= 100 and len(seasons) >= 2:
        return temporal_backtest_model(
            history,
            attribute_matrix,
            visual_matrix,
            targets,
            seasons[-1],
        )

    candidate_alphas = (0.1, 1.0, 10.0, 100.0)
    ridge_by_alpha: dict[float, np.ndarray] = {}
    leave_one_out = LeaveOneOut()
    all_indices = np.arange(len(history))
    for alpha in candidate_alphas:
        predictions = np.empty(len(history), dtype=np.float64)
        for train_indices, holdout_indices in leave_one_out.split(history):
            train_items = [history[int(index)] for index in train_indices]
            holdout_items = [history[int(index)] for index in holdout_indices]
            pipeline = fit_demand_pipeline(train_items, targets[train_indices], alpha)
            predictions[holdout_indices] = pipeline.predict([demand_features(item) for item in holdout_items])
        ridge_by_alpha[alpha] = predictions

    best: dict[str, Any] | None = None
    attribute_weights = [round(value / 10, 1) for value in range(1, 10)]
    analogue_predictions: dict[tuple[float, int], np.ndarray] = {}
    for config in ParameterGrid({"attributeWeight": attribute_weights, "topK": [3, 5, 8]}):
        attribute_weight = float(config["attributeWeight"])
        top_k = int(config["topK"])
        predictions = []
        for holdout in all_indices:
            candidates = [int(index) for index in all_indices if index != holdout]
            prediction, _ = analogue_prediction(
                int(holdout),
                candidates,
                targets,
                attribute_matrix,
                visual_matrix,
                attribute_weight,
                top_k,
            )
            predictions.append(prediction)
        analogue_predictions[(attribute_weight, top_k)] = np.asarray(predictions)

    search = ParameterGrid(
        {
            "attributeWeight": attribute_weights,
            "regressionBlend": [0.15, 0.25, 0.35, 0.50],
            "ridgeAlpha": list(candidate_alphas),
            "topK": [3, 5, 8],
        }
    )
    for config in search:
        attribute_weight = float(config["attributeWeight"])
        top_k = int(config["topK"])
        alpha = float(config["ridgeAlpha"])
        regression_blend = float(config["regressionBlend"])
        analogue = analogue_predictions[(attribute_weight, top_k)]
        ridge = ridge_by_alpha[alpha]
        ensemble = analogue * (1 - regression_blend) + ridge * regression_blend
        score = wape(targets, ensemble)
        candidate = {
            "attributeWeight": attribute_weight,
            "visualWeight": 1 - attribute_weight,
            "topK": top_k,
            "ridgeAlpha": alpha,
            "regressionBlend": regression_blend,
            "analoguePredictions": analogue,
            "ridgePredictions": ridge,
            "predictions": ensemble,
            "score": score,
        }
        if best is None or score < best["score"]:
            best = candidate

    assert best is not None
    predictions = np.asarray(best["predictions"])
    residuals = np.abs(predictions - targets)
    interval = finite_sample_quantile(residuals, coverage=0.80)
    bias = float((predictions - targets).sum() / max(targets.sum(), 1e-9))
    coverage = float(np.mean(residuals <= interval))
    demand_pipeline = fit_demand_pipeline(history, targets, float(best["ridgeAlpha"]))
    return {
        **{key: value for key, value in best.items() if not isinstance(value, np.ndarray)},
        "predictions": predictions,
        "residuals": residuals,
        "conformalHalfWidth": interval,
        "metrics": {
            "wape": round(wape(targets, predictions), 4),
            "mae": round(float(mean_absolute_error(targets, predictions)), 1),
            "bias": round(bias, 4),
            "intervalCoverage": round(coverage, 4),
        },
        "demandPipeline": demand_pipeline,
        "selectionMethod": "LeaveOneOut + ParameterGrid",
        "evaluation": "Leave-one-out validation; temporal holdout requires multiple seasons",
        "validationRows": len(history),
        "attributeWeightGrid": attribute_weights,
    }


def match_confidence(
    top_scores: list[float],
    has_visual: bool,
    issue_count: int,
) -> str:
    """Rate analogue relevance without mixing in forecast-range width."""

    top = top_scores[0] if top_scores else 0.0
    mean_top = float(np.mean(top_scores[:3])) if top_scores else 0.0
    if top >= 0.84 and mean_top >= 0.72 and has_visual and issue_count == 0:
        return "High"
    if top >= 0.62 and mean_top >= 0.52:
        return "Medium"
    return "Low"


def demand_uncertainty(quantity: float, interval_half_width: float) -> str:
    """Label uncertainty from the conformal half-width relative to forecast sales."""

    relative_half_width = interval_half_width / max(quantity, 1.0)
    if relative_half_width <= 0.20:
        return "Narrow"
    if relative_half_width <= 0.40:
        return "Moderate"
    return "Wide"


def no_suitable_product_match(
    selected: list[dict[str, Any]],
    relevance: str,
) -> bool:
    """Reject weak or non-visual candidates instead of presenting a false match."""

    if not selected or relevance == "Low":
        return True
    visual_score = selected[0].get("visualScore")
    return visual_score is None or float(visual_score) < MIN_CONVINCING_VISUAL_SCORE


def recommend_one(
    item: dict[str, Any],
    history: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    targets: np.ndarray,
    demand_pipeline: Pipeline,
    model: dict[str, Any],
) -> dict[str, Any]:
    top_k = int(model["topK"])
    selected = matches[:top_k]
    history_index = {historical["id"]: index for index, historical in enumerate(history)}
    weights = np.asarray([max(float(match["hybridScore"]), 0.01) ** 2 for match in selected])
    selected_targets = np.asarray([targets[history_index[match["historicalId"]]] for match in selected])
    analogue_sales = float(np.average(selected_targets, weights=weights))
    regression_sales = float(demand_pipeline.predict([demand_features(item)])[0])
    regression_sales = clamp(regression_sales, 0, MAX_BUY)
    top_scores = [float(match["hybridScore"]) for match in selected]
    top_visual_available = bool(selected and selected[0].get("visualScore") is not None)
    issue_count = sum(len(quality_flags(history[history_index[match["historicalId"]]])) for match in selected[:3])
    relevance = match_confidence(
        top_scores,
        top_visual_available,
        issue_count,
    )
    no_suitable_match = no_suitable_product_match(selected, relevance)
    blend = float(model["regressionBlend"])
    raw_sales = regression_sales if no_suitable_match else analogue_sales * (1 - blend) + regression_sales * blend
    expected_sales = int(clamp(round_pack(raw_sales), 0, MAX_BUY))
    target_sell_through = max(
        float(model.get("targetSellThrough", DEFAULT_TARGET_SELL_THROUGH)),
        0.01,
    )
    quantity = int(clamp(round_pack(expected_sales / target_sell_through), MIN_BUY, MAX_BUY))
    sales_interval = float(model.get("salesConformalHalfWidth", model["conformalHalfWidth"])) * (
        1.0 + max(0.0, 0.7 - (top_scores[0] if top_scores else 0.0))
    )
    uncertainty = demand_uncertainty(expected_sales, sales_interval)
    sales_low = int(clamp(round_pack(expected_sales - sales_interval), 0, MAX_BUY))
    sales_high = int(clamp(round_pack(expected_sales + sales_interval), 0, MAX_BUY))
    order_low = int(clamp(round_pack(sales_low / target_sell_through), MIN_BUY, MAX_BUY))
    order_high = int(clamp(round_pack(sales_high / target_sell_through), MIN_BUY, MAX_BUY))
    return {
        "quantity": quantity,
        "low": order_low,
        "high": order_high,
        "expectedSales": expected_sales,
        "salesLow": sales_low,
        "salesHigh": sales_high,
        "matchConfidence": relevance,
        "noSuitableMatch": no_suitable_match,
        "demandUncertainty": uncertainty,
        "uncertaintyRatio": round(sales_interval / max(expected_sales, 1.0), 4),
        "confidence": relevance,
        "analogueSales": round_pack(analogue_sales),
        "regressionSales": round_pack(regression_sales),
        "salesIntervalHalfWidth": round_pack(sales_interval),
        "analogueQuantity": round_pack(analogue_sales / target_sell_through),
        "regressionQuantity": round_pack(regression_sales / target_sell_through),
        "intervalHalfWidth": round_pack(sales_interval / target_sell_through),
        "topMatchScore": round(top_scores[0] if top_scores else 0.0, 4),
        "modelVersion": MODEL_VERSION,
    }


def blend_two_stage_visual_scores(
    fashion_score: float | None,
    dino_score: float | None,
    dino_weight: float,
) -> float | None:
    """Blend calibrated stage scores only when both visual stages are present."""

    if fashion_score is None:
        return None
    if dino_score is None:
        return fashion_score
    return round(fashion_score * (1 - dino_weight) + dino_score * dino_weight, 4)


def _candidate_calibration_values(
    rows: list[dict[str, Any]],
    *,
    left_ids: set[str],
    right_ids: set[str],
    distance_key: str,
    exclude_self: bool = False,
) -> list[float]:
    return [
        float(row[distance_key])
        for row in rows
        if str(row["leftId"]) in left_ids
        and str(row["rightId"]) in right_ids
        and (not exclude_self or row["leftId"] != row["rightId"])
    ]


def _two_stage_visual_matrix(
    history: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    fashion_calibration: VisionCalibration,
    dino_calibration: VisionCalibration,
    dino_weight: float,
) -> np.ndarray:
    index_by_id = {str(item["id"]): index for index, item in enumerate(history)}
    matrix = np.full((len(history), len(history)), np.nan, dtype=np.float64)
    for row in candidate_rows:
        left_index = index_by_id.get(str(row["leftId"]))
        right_index = index_by_id.get(str(row["rightId"]))
        if left_index is None or right_index is None:
            continue
        score = blend_two_stage_visual_scores(
            fashion_calibration.similarity(float(row["fashionDistance"])),
            dino_calibration.similarity(float(row["dinoDistance"])),
            dino_weight,
        )
        if score is not None:
            matrix[left_index, right_index] = score
    return matrix


def build_model_artifact(source: dict[str, Any], vision_output: dict[str, Any]) -> dict[str, Any]:
    history = [dict(item) for item in source["historical"]]
    upcoming = [dict(item) for item in source["upcoming"]]
    active_attribute_weights = informative_attribute_weights(history, upcoming)
    rows = vision_output.get("distances", [])
    candidate_rows = list(vision_output.get("candidatePairs", []))
    historical_ids = {str(item["id"]) for item in history}
    upcoming_ids = {str(item["id"]) for item in upcoming}
    count = len(history)
    attribute_matrix = np.eye(count, dtype=np.float64)
    for left_index, left in enumerate(history):
        for right_index, right in enumerate(history):
            attribute, _ = attribute_similarity(left, right, active_attribute_weights)
            attribute_matrix[left_index, right_index] = attribute

    targets = np.asarray([sales_target(item) for item in history], dtype=np.float64)
    candidate_history_ids: dict[str, set[str]] = {}
    candidate_pair_map: dict[tuple[str, str], dict[str, Any]] = {}
    distance_map: dict[tuple[str, str], float] = {}
    dino_historical_calibration: VisionCalibration | None = None
    dino_serving_calibration: VisionCalibration | None = None
    selected_dino_weight = 0.0
    if candidate_rows:
        fashion_historical_values = _candidate_calibration_values(
            candidate_rows,
            left_ids=historical_ids,
            right_ids=historical_ids,
            distance_key="fashionDistance",
            exclude_self=True,
        )
        fashion_serving_values = _candidate_calibration_values(
            candidate_rows,
            left_ids=upcoming_ids,
            right_ids=historical_ids,
            distance_key="fashionDistance",
        )
        dino_historical_values = _candidate_calibration_values(
            candidate_rows,
            left_ids=historical_ids,
            right_ids=historical_ids,
            distance_key="dinoDistance",
            exclude_self=True,
        )
        dino_serving_values = _candidate_calibration_values(
            candidate_rows,
            left_ids=upcoming_ids,
            right_ids=historical_ids,
            distance_key="dinoDistance",
        )
        historical_calibration = calibrate_vision(fashion_historical_values)
        serving_calibration = calibrate_vision(fashion_serving_values or fashion_historical_values)
        dino_historical_calibration = calibrate_vision(dino_historical_values)
        dino_serving_calibration = calibrate_vision(dino_serving_values or dino_historical_values)
        weight_grid = tuple(
            float(weight)
            for weight in vision_output.get("reranker", {}).get(
                "weightGrid",
                (0.0, 0.25, 0.5, 0.75, 1.0),
            )
        )
        fitted_candidates: list[tuple[float, dict[str, Any]]] = []
        for dino_weight in weight_grid:
            visual_matrix = _two_stage_visual_matrix(
                history,
                candidate_rows,
                historical_calibration,
                dino_historical_calibration,
                dino_weight,
            )
            fitted_candidates.append(
                (dino_weight, backtest_model(history, attribute_matrix, visual_matrix, targets)))
        selected_dino_weight, fitted = min(
            fitted_candidates,
            key=lambda candidate: (float(candidate[1]["score"]), candidate[0]),
        )
        for row in candidate_rows:
            if str(row["leftId"]) in upcoming_ids and str(row["rightId"]) in historical_ids:
                left_id = str(row["leftId"])
                right_id = str(row["rightId"])
                candidate_history_ids.setdefault(left_id, set()).add(right_id)
                candidate_pair_map[(left_id, right_id)] = row
    else:
        distance_map = {
            (str(row["leftId"]), str(row["rightId"])): float(row["distance"])
            for row in rows
        }
        historical_calibration_values = [
            float(row["distance"])
            for row in rows
            if str(row["leftId"]) in historical_ids
            and str(row["rightId"]) in historical_ids
            and row["leftId"] != row["rightId"]
        ]
        serving_calibration_values = [
            float(row["distance"])
            for row in rows
            if str(row["leftId"]) in upcoming_ids and str(row["rightId"]) in historical_ids
        ]
        historical_calibration = calibrate_vision(historical_calibration_values)
        serving_calibration = calibrate_vision(
            serving_calibration_values or historical_calibration_values,
        )
        visual_matrix = np.full((count, count), np.nan, dtype=np.float64)
        for left_index, left in enumerate(history):
            for right_index, right in enumerate(history):
                visual = historical_calibration.similarity(distance_map.get((left["id"], right["id"])))
                if visual is not None:
                    visual_matrix[left_index, right_index] = visual
        fitted = backtest_model(history, attribute_matrix, visual_matrix, targets)
    for item, target in zip(history, targets, strict=True):
        # Retain enough precision for API-side reproduction after loading the
        # artifact. Sales and final order recommendations are pack-rounded later.
        item["salesTarget"] = round(float(target), 4)
        item["normalizedDemand"] = round(
            float(target) / DEFAULT_TARGET_SELL_THROUGH,
            4,
        )
        item["qualityFlags"] = quality_flags(item)

    attribute_weight = float(fitted["attributeWeight"])
    match_confidence_counts = {"High": 0, "Medium": 0, "Low": 0}
    uncertainty_counts = {"Narrow": 0, "Moderate": 0, "Wide": 0}
    all_attribute_scores: list[float] = []
    all_visual_scores: list[float] = []
    retrieval_history = [historical for historical in history if historical.get("imageUrl")] or history
    for item in upcoming:
        matches: list[dict[str, Any]] = []
        shortlist_ids = candidate_history_ids.get(str(item["id"]))
        eligible_history = (
            [historical for historical in retrieval_history if str(historical["id"]) in shortlist_ids]
            if shortlist_ids
            else retrieval_history
        )
        for historical in eligible_history:
            attribute, breakdown = attribute_similarity(item, historical, active_attribute_weights)
            all_attribute_scores.append(attribute)
            fashion_visual: float | None = None
            dino_visual: float | None = None
            if candidate_rows:
                candidate = candidate_pair_map.get((str(item["id"]), str(historical["id"])))
                if candidate is not None:
                    fashion_visual = serving_calibration.similarity(
                        float(candidate["fashionDistance"]),
                    )
                    assert dino_serving_calibration is not None
                    dino_visual = dino_serving_calibration.similarity(
                        float(candidate["dinoDistance"]),
                    )
                visual = blend_two_stage_visual_scores(
                    fashion_visual,
                    dino_visual,
                    selected_dino_weight,
                )
            else:
                fashion_visual = serving_calibration.similarity(
                    distance_map.get((item["id"], historical["id"])),
                )
                visual = fashion_visual
            if visual is not None:
                all_visual_scores.append(visual)
            hybrid = combined_similarity(attribute, visual, attribute_weight)
            matches.append(
                {
                    "historicalId": historical["id"],
                    "attributeScore": round(attribute, 4),
                    "visualScore": visual,
                    "fashionVisualScore": fashion_visual,
                    "dinoVisualScore": dino_visual,
                    "hybridScore": round(hybrid, 4),
                    "attributeBreakdown": breakdown,
                }
            )
        matches.sort(key=lambda match: match["hybridScore"], reverse=True)
        item["recommendation"] = recommend_one(
            item,
            history,
            matches,
            targets,
            fitted["demandPipeline"],
            fitted,
        )
        item["matches"] = matches[:8]
        model_flags = list(item.get("modelFlags", []))
        if matches and matches[0]["visualScore"] is None:
            model_flags.append("missing_image")
        if item["recommendation"]["noSuitableMatch"]:
            model_flags.append("no_suitable_match")
        item["modelFlags"] = list(dict.fromkeys(model_flags))
        match_confidence_counts[item["recommendation"]["matchConfidence"]] += 1
        uncertainty_counts[item["recommendation"]["demandUncertainty"]] += 1

    anomaly_counts = {
        "dispatchAboveOrder": sum("dispatch_above_order" in item["qualityFlags"] for item in history),
        "salesAboveDispatch": sum("sales_above_dispatch" in item["qualityFlags"] for item in history),
        "sellThroughAbove100": sum("sell_through_above_100" in item["qualityFlags"] for item in history),
    }
    meta = dict(source.get("meta", {}))
    source_quality = dict(meta.get("dataQuality", {}))
    source_quality.update(anomaly_counts)
    meta.update(
        {
            "title": "Turtle Season Intelligence AI",
            "generatedAt": datetime.now(UTC).isoformat(),
            "confidenceCounts": match_confidence_counts,
            "matchConfidenceCounts": match_confidence_counts,
            "demandUncertaintyCounts": uncertainty_counts,
            "attributeScoreRange": (
                [round(min(all_attribute_scores), 3), round(max(all_attribute_scores), 3)]
                if all_attribute_scores
                else [0, 0]
            ),
            "visualScoreRange": (
                [
                    round(min(all_visual_scores), 3),
                    round(max(all_visual_scores), 3),
                ]
                if all_visual_scores
                else [0, 0]
            ),
            "visualMethod": vision_output.get("engine", "FashionCLIP image embedding"),
            "attributeAudit": attribute_audit(
                history,
                upcoming,
                active_attribute_weights,
                source_meta=meta,
            ),
            "visionModel": {
                "modelId": vision_output.get("modelId", "unknown"),
                "modelRevision": vision_output.get("modelRevision"),
                "embeddingDimension": vision_output.get("embeddingDimension"),
                "device": vision_output.get("device", "unknown"),
                "historicalCoverage": vision_output.get("historicalCoverage", 0),
                "upcomingCoverage": vision_output.get("upcomingCoverage", 0),
                "reranker": vision_output.get("reranker"),
            },
            "model": {
                "version": MODEL_VERSION,
                "status": "Real-data pilot — production architecture",
                "trainingRows": count,
                "validationRows": int(fitted["validationRows"]),
                "targetSellThrough": DEFAULT_TARGET_SELL_THROUGH,
                "algorithm": (
                    "Attribute retrieval + scikit-learn Ridge sales forecast + inventory policy"
                    if not all_visual_scores
                    else (
                        "Two-stage FashionSigLIP retrieval + DINOv2 visual reranking + "
                        "attribute constraints + scikit-learn Ridge sales forecast + inventory policy"
                    )
                    if candidate_rows
                    else "Calibrated FashionCLIP retrieval + scikit-learn Ridge sales forecast + inventory policy"
                ),
                "demandLibrary": "scikit-learn",
                "demandPipeline": "DictVectorizer + StandardScaler + Ridge",
                "forecastTarget": "Cleaned positive historical unit sales",
                "orderPolicy": "Expected sales divided by target sell-through",
                "modelSelection": fitted["selectionMethod"],
                "attributeWeights": {name: round(weight, 4) for name, weight in active_attribute_weights.items()},
                "attributeWeightGrid": fitted["attributeWeightGrid"],
                "attributeWeight": round(attribute_weight, 2),
                "visualWeight": round(1 - attribute_weight, 2),
                "dinoRerankWeight": round(selected_dino_weight, 2),
                "minimumVisualScore": MIN_CONVINCING_VISUAL_SCORE,
                "minimumMatchConfidence": "Medium",
                "noMatchPolicy": (
                    "Show no product match when the best candidate has low "
                    "confidence, lacks a visual score, or falls below the visual "
                    "similarity threshold. Use the regression forecast without "
                    "analogue blending in that case."
                ),
                "topK": int(fitted["topK"]),
                "regressionBlend": float(fitted["regressionBlend"]),
                "ridgeAlpha": float(fitted["ridgeAlpha"]),
                "backtest": fitted["metrics"],
                "evaluation": fitted["evaluation"],
                "interval": "Finite-sample 80% conformal interval for expected sales from out-of-fold residuals",
                "salesConformalHalfWidth": round_pack(float(fitted["conformalHalfWidth"])),
                "conformalHalfWidth": round_pack(float(fitted["conformalHalfWidth"])),
            },
            "visionCalibration": {
                "medianDistance": round(serving_calibration.median, 4),
                "q10Distance": round(serving_calibration.q10, 4),
                "q90Distance": round(serving_calibration.q90, 4),
                "historicalMedianDistance": round(historical_calibration.median, 4),
                "historicalQ10Distance": round(historical_calibration.q10, 4),
                "historicalQ90Distance": round(historical_calibration.q90, 4),
                "servingMedianDistance": round(serving_calibration.median, 4),
                "servingQ10Distance": round(serving_calibration.q10, 4),
                "servingQ90Distance": round(serving_calibration.q90, 4),
                "dinoHistoricalMedianDistance": (
                    round(dino_historical_calibration.median, 4)
                    if dino_historical_calibration is not None
                    else None
                ),
                "dinoServingMedianDistance": (
                    round(dino_serving_calibration.median, 4)
                    if dino_serving_calibration is not None
                    else None
                ),
                "method": vision_output.get(
                    "calibrationMethod",
                    "Robust logistic calibration of neural embedding distance",
                ),
            },
            "dataQuality": source_quality,
        }
    )
    return {"meta": meta, "historical": history, "upcoming": upcoming}
