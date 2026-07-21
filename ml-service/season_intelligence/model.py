from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_VERSION = "2.4.0"
DEFAULT_TARGET_SELL_THROUGH = 0.70
PACK_SIZE = 25
MIN_BUY = 100
MAX_BUY = 2_000

PATTERN_FAMILY = {
    "DIGITAL PRINT": "PRINT",
    "DISCHARGE PRINT": "PRINT",
    "PIGMENT PRINT": "PRINT",
    "PRINTS": "PRINT",
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
    "category": 0.16,
    "sleeve": 0.07,
    "provision": 0.07,
    "pattern": 0.17,
    "lifecycle": 0.05,
    "fit": 0.14,
    "fabric": 0.14,
    "colour": 0.09,
    "price": 0.11,
}

ATTRIBUTE_SCHEMA = {
    "category": (
        "Category / item type", "Item Type", "SEGMENT1",
        "Exact match with a strong cross-category penalty",
    ),
    "sleeve": ("Sleeve", "SLEEVS", "SEGMENT5", "Exact categorical match"),
    "provision": ("Provision / fit code", "PROV", "SEGMENT6", "Exact categorical match"),
    "pattern": ("Pattern", "CAT1", "CAT1", "Exact or related pattern-family match"),
    "lifecycle": ("Lifecycle family", "CAT6", "CAT6", "Normalized AW, SS, or CORE family match"),
    "fit": ("Collection / fit", "CAT3", "CAT3", "Exact categorical match"),
    "fabric": ("Fabric", "CAT4", "CAT4", "Exact or token-overlap match"),
    "colour": ("Colour name", "COLOR_NAME", "COLOR", "Exact or related colour-family match"),
    "price": ("MRP / price band", "MRP", "MRP", "Smooth log-price distance"),
}

EXCLUDED_CONSTANT_ATTRIBUTES = (
    {
        "label": "Range code",
        "historicalColumn": "CAT2",
        "upcomingColumn": "CAT2",
        "reason": "Constant after normalization: CMI + VMI and VMI + CMI are the same range.",
    },
    {
        "label": "Merch type",
        "historicalColumn": "CAT5",
        "upcomingColumn": "CAT5",
        "reason": "All historical candidates are FASHION, so this field cannot rank one analogue above another.",
    },
)

EXCLUDED_NON_COMPARISON_FIELDS = (
    {
        "label": "Identifiers",
        "historicalColumn": "SL, CON, SORT",
        "upcomingColumn": "CON, SEGMENT2",
        "reason": "Row and style identifiers identify products; they are not reusable product characteristics.",
    },
    {
        "label": "Colour variant code",
        "historicalColumn": "COLOR",
        "upcomingColumn": "SEGMENT3",
        "reason": "Variant codes such as 1001 are not stable colour meanings; colour names are compared instead.",
    },
    {
        "label": "Historical season label",
        "historicalColumn": "SEASON",
        "upcomingColumn": "—",
        "reason": "There is no upcoming counterpart; the comparable CAT6 lifecycle family is used instead.",
    },
    {
        "label": "Demand outcomes",
        "historicalColumn": "ORDER, DISPATCH, SALE, SALE THRU",
        "upcomingColumn": "—",
        "reason": "These fields train and validate the quantity forecast; using them in product similarity would leak outcomes.",
    },
)


def norm(value: Any) -> str:
    return " ".join(str(value or "").upper().split()).strip()


def lifecycle_family(value: Any) -> str:
    lifecycle = norm(value)
    if lifecycle.startswith("SS"):
        return "SS"
    if lifecycle.startswith("AW"):
        return "AW"
    return lifecycle


def attribute_value(item: dict[str, Any], name: str) -> Any:
    if name == "category":
        return norm(item.get("itemType"))
    if name == "lifecycle":
        return lifecycle_family(item.get("lifecycle"))
    if name == "price":
        return float(item.get("mrp") or 0)
    source_key = {
        "sleeve": "sleeve",
        "provision": "provision",
        "pattern": "pattern",
        "fit": "fit",
        "fabric": "fabric",
        "colour": "colour",
    }[name]
    return norm(item.get(source_key))


def populated_attribute_values(items: list[dict[str, Any]], name: str) -> set[Any]:
    return {
        value
        for item in items
        if (value := attribute_value(item, name)) not in ("", 0.0)
    }


def informative_attribute_weights(history: list[dict[str, Any]]) -> dict[str, float]:
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
    }
    total = sum(active.values())
    if not active or total <= 0:
        raise ValueError("No informative comparable attributes were found in the historical dataset")
    return {name: weight / total for name, weight in active.items()}


def attribute_audit(
    history: list[dict[str, Any]],
    upcoming: list[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    active = []
    for name, weight in weights.items():
        label, historical_column, upcoming_column, method = ATTRIBUTE_SCHEMA[name]
        historical_values = populated_attribute_values(history, name)
        upcoming_values = populated_attribute_values(upcoming, name)
        active.append({
            "key": name,
            "label": label,
            "historicalColumn": historical_column,
            "upcomingColumn": upcoming_column,
            "weight": round(weight, 4),
            "historicalUnique": len(historical_values),
            "upcomingUnique": len(upcoming_values),
            "method": method,
        })
    automatically_excluded = []
    for name in sorted(BASE_ATTRIBUTE_WEIGHTS.keys() - weights.keys()):
        label, historical_column, upcoming_column, _ = ATTRIBUTE_SCHEMA[name]
        automatically_excluded.append({
            "label": label,
            "historicalColumn": historical_column,
            "upcomingColumn": upcoming_column,
            "reason": "Automatically excluded because the historical field has fewer than two populated values.",
        })
    return {
        "historicalSourceRange": "Sheet1!A1:T34",
        "upcomingSourceRange": "Sheet1!A1:N168",
        "activeCount": len(active),
        "activeAttributes": active,
        "excludedConstants": [*EXCLUDED_CONSTANT_ATTRIBUTES, *automatically_excluded],
        "excludedNonComparisonFields": list(EXCLUDED_NON_COMPARISON_FIELDS),
        "policy": "Only comparable fields with at least two populated historical values can contribute to similarity.",
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


def pattern_similarity(left: Any, right: Any) -> float:
    a, b = norm(left), norm(right)
    if a == b:
        return 1.0
    if PATTERN_FAMILY.get(a, a) == PATTERN_FAMILY.get(b, b):
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
    item_type = categorical(left.get("itemType"), right.get("itemType"))
    left_mrp = max(float(left.get("mrp") or 1), 1)
    right_mrp = max(float(right.get("mrp") or 1), 1)
    values = {
        "category": item_type,
        "sleeve": categorical(left.get("sleeve"), right.get("sleeve")),
        "provision": categorical(left.get("provision"), right.get("provision")),
        "pattern": pattern_similarity(left.get("pattern"), right.get("pattern")),
        "lifecycle": categorical(
            lifecycle_family(left.get("lifecycle")),
            lifecycle_family(right.get("lifecycle")),
        ),
        "fit": categorical(left.get("fit"), right.get("fit")),
        "fabric": max(categorical(left.get("fabric"), right.get("fabric")), jaccard(left.get("fabric"), right.get("fabric"))),
        "colour": colour_similarity(left.get("colour"), right.get("colour")),
        "price": math.exp(-abs(math.log(left_mrp / right_mrp)) / 0.30),
    }
    values = {name: values[name] for name in active_weights}
    score = sum(values[name] * weight for name, weight in active_weights.items())
    if item_type == 0:
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

    pattern = norm(item.get("pattern"))
    colour = norm(item.get("colour"))
    season_family = lifecycle_family(item.get("lifecycle"))
    categorical_values = {
        "item_type": norm(item.get("itemType")),
        "sleeve": norm(item.get("sleeve")),
        "provision": norm(item.get("provision")),
        "pattern": PATTERN_FAMILY.get(pattern, pattern),
        "fit": norm(item.get("fit")),
        "colour": COLOUR_FAMILY.get(colour, colour),
        "season": season_family,
    }
    features = {
        f"{field}={value or 'UNKNOWN'}": 1.0
        for field, value in categorical_values.items()
    }
    features.update({f"fabric={token}": 1.0 for token in token_set(item.get("fabric"))})
    features["log_mrp"] = math.log(max(float(item.get("mrp") or 1), 1))
    return features


def build_demand_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("features", DictVectorizer(sparse=True)),
        ("scale", StandardScaler(with_mean=False)),
        ("ridge", Ridge(alpha=alpha, solver="lsqr")),
    ])


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


def backtest_model(
    history: list[dict[str, Any]],
    attribute_matrix: np.ndarray,
    visual_matrix: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
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
            predictions[holdout_indices] = pipeline.predict(
                [demand_features(item) for item in holdout_items]
            )
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

    search = ParameterGrid({
        "attributeWeight": attribute_weights,
        "regressionBlend": [0.15, 0.25, 0.35, 0.50],
        "ridgeAlpha": list(candidate_alphas),
        "topK": [3, 5, 8],
    })
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
    weights = np.asarray([max(float(match["hybridScore"]), 0.01) ** 2 for match in selected])
    history_index = {historical["id"]: index for index, historical in enumerate(history)}
    selected_targets = np.asarray([targets[history_index[match["historicalId"]]] for match in selected])
    analogue_sales = float(np.average(selected_targets, weights=weights))
    regression_sales = float(demand_pipeline.predict([demand_features(item)])[0])
    regression_sales = clamp(regression_sales, 0, MAX_BUY)
    blend = float(model["regressionBlend"])
    raw_sales = analogue_sales * (1 - blend) + regression_sales * blend
    expected_sales = int(clamp(round_pack(raw_sales), 0, MAX_BUY))
    target_sell_through = max(
        float(model.get("targetSellThrough", DEFAULT_TARGET_SELL_THROUGH)),
        0.01,
    )
    quantity = int(clamp(round_pack(expected_sales / target_sell_through), MIN_BUY, MAX_BUY))
    top_scores = [float(match["hybridScore"]) for match in selected]
    top_visual_available = bool(selected and selected[0].get("visualScore") is not None)
    sales_interval = float(
        model.get("salesConformalHalfWidth", model["conformalHalfWidth"])
    ) * (
        1.0 + max(0.0, 0.7 - (top_scores[0] if top_scores else 0.0))
    )
    issue_count = sum(len(quality_flags(history[history_index[match["historicalId"]]])) for match in selected[:3])
    relevance = match_confidence(top_scores, top_visual_available, issue_count)
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


def build_model_artifact(source: dict[str, Any], vision_output: dict[str, Any]) -> dict[str, Any]:
    history = [dict(item) for item in source["historical"]]
    upcoming = [dict(item) for item in source["upcoming"]]
    active_attribute_weights = informative_attribute_weights(history)
    rows = vision_output.get("distances", [])
    distance_map = {
        (str(row["leftId"]), str(row["rightId"])): float(row["distance"])
        for row in rows
    }
    calibration_values = [
        float(row["distance"])
        for row in rows
        if row["leftId"] != row["rightId"]
    ]
    calibration = calibrate_vision(calibration_values)

    count = len(history)
    attribute_matrix = np.eye(count, dtype=np.float64)
    visual_matrix = np.full((count, count), np.nan, dtype=np.float64)
    for left_index, left in enumerate(history):
        for right_index, right in enumerate(history):
            attribute, _ = attribute_similarity(left, right, active_attribute_weights)
            attribute_matrix[left_index, right_index] = attribute
            distance = distance_map.get((left["id"], right["id"]))
            visual = calibration.similarity(distance)
            if visual is not None:
                visual_matrix[left_index, right_index] = visual

    targets = np.asarray([sales_target(item) for item in history], dtype=np.float64)
    fitted = backtest_model(history, attribute_matrix, visual_matrix, targets)
    for item, target in zip(history, targets):
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
    for item in upcoming:
        matches: list[dict[str, Any]] = []
        for historical in history:
            attribute, breakdown = attribute_similarity(item, historical, active_attribute_weights)
            all_attribute_scores.append(attribute)
            visual = calibration.similarity(distance_map.get((item["id"], historical["id"])))
            if visual is not None:
                all_visual_scores.append(visual)
            hybrid = combined_similarity(attribute, visual, attribute_weight)
            matches.append({
                "historicalId": historical["id"],
                "attributeScore": round(attribute, 4),
                "visualScore": visual,
                "hybridScore": round(hybrid, 4),
                "attributeBreakdown": breakdown,
            })
        matches.sort(key=lambda match: match["hybridScore"], reverse=True)
        item["matches"] = matches
        item["recommendation"] = recommend_one(
            item,
            history,
            matches,
            targets,
            fitted["demandPipeline"],
            fitted,
        )
        item["modelFlags"] = ["missing_image"] if matches and matches[0]["visualScore"] is None else []
        match_confidence_counts[item["recommendation"]["matchConfidence"]] += 1
        uncertainty_counts[item["recommendation"]["demandUncertainty"]] += 1

    anomaly_counts = {
        "dispatchAboveOrder": sum("dispatch_above_order" in item["qualityFlags"] for item in history),
        "salesAboveDispatch": sum("sales_above_dispatch" in item["qualityFlags"] for item in history),
        "sellThroughAbove100": sum("sell_through_above_100" in item["qualityFlags"] for item in history),
    }
    meta = dict(source.get("meta", {}))
    meta.update({
        "title": "Turtle Season Intelligence AI",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "confidenceCounts": match_confidence_counts,
        "matchConfidenceCounts": match_confidence_counts,
        "demandUncertaintyCounts": uncertainty_counts,
        "attributeScoreRange": (
            [round(min(all_attribute_scores), 3), round(max(all_attribute_scores), 3)]
            if all_attribute_scores else [0, 0]
        ),
        "visualScoreRange": [round(min(all_visual_scores), 3), round(max(all_visual_scores), 3)] if all_visual_scores else [0, 0],
        "visualMethod": vision_output.get("engine", "FashionCLIP image embedding"),
        "attributeAudit": attribute_audit(history, upcoming, active_attribute_weights),
        "visionModel": {
            "modelId": vision_output.get("modelId", "unknown"),
            "modelRevision": vision_output.get("modelRevision"),
            "embeddingDimension": vision_output.get("embeddingDimension"),
            "device": vision_output.get("device", "unknown"),
            "historicalCoverage": vision_output.get("historicalCoverage", 0),
            "upcomingCoverage": vision_output.get("upcomingCoverage", 0),
        },
        "model": {
            "version": MODEL_VERSION,
            "status": "Pilot validated — production architecture",
            "trainingRows": count,
            "targetSellThrough": DEFAULT_TARGET_SELL_THROUGH,
            "algorithm": "Calibrated FashionCLIP retrieval + scikit-learn Ridge sales forecast + inventory policy",
            "demandLibrary": "scikit-learn",
            "demandPipeline": "DictVectorizer + StandardScaler + Ridge",
            "forecastTarget": "Cleaned historical unit sales",
            "orderPolicy": "Expected sales divided by target sell-through",
            "modelSelection": "LeaveOneOut + ParameterGrid",
            "attributeWeights": {name: round(weight, 4) for name, weight in active_attribute_weights.items()},
            "attributeWeightGrid": [round(value / 10, 1) for value in range(1, 10)],
            "attributeWeight": round(attribute_weight, 2),
            "visualWeight": round(1 - attribute_weight, 2),
            "topK": int(fitted["topK"]),
            "regressionBlend": float(fitted["regressionBlend"]),
            "ridgeAlpha": float(fitted["ridgeAlpha"]),
            "backtest": fitted["metrics"],
            "evaluation": "Leave-one-out validation; temporal holdout requires at least three clean seasons",
            "interval": "Finite-sample 80% conformal interval for expected sales from out-of-fold residuals",
            "salesConformalHalfWidth": round_pack(float(fitted["conformalHalfWidth"])),
            "conformalHalfWidth": round_pack(float(fitted["conformalHalfWidth"])),
        },
        "visionCalibration": {
            "medianDistance": round(calibration.median, 4),
            "q10Distance": round(calibration.q10, 4),
            "q90Distance": round(calibration.q90, 4),
            "method": vision_output.get(
                "calibrationMethod",
                "Robust logistic calibration of neural embedding distance",
            ),
        },
        "dataQuality": anomaly_counts,
    })
    return {"meta": meta, "historical": history, "upcoming": upcoming}
