from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np


MODEL_VERSION = "2.1.0"
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

ATTRIBUTE_WEIGHTS = {
    "category": 0.16,
    "sleeve": 0.07,
    "provision": 0.07,
    "pattern": 0.16,
    "range": 0.04,
    "fit": 0.14,
    "fabric": 0.14,
    "fashion": 0.02,
    "colour": 0.09,
    "price": 0.11,
}


def norm(value: Any) -> str:
    return " ".join(str(value or "").upper().split()).strip()


def canonical_plus(value: Any) -> str:
    return "+".join(sorted(part.strip() for part in norm(value).split("+")))


def token_set(value: Any) -> set[str]:
    ignored = {"100%", "100", "PERCENT", "THE"}
    return {part for part in re.findall(r"[A-Z0-9]+", norm(value)) if part not in ignored}


def jaccard(left: Any, right: Any) -> float:
    a, b = token_set(left), token_set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def categorical(left: Any, right: Any) -> float:
    return 1.0 if norm(left) == norm(right) else 0.0


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


def attribute_similarity(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, dict[str, float]]:
    item_type = categorical(left.get("itemType"), right.get("itemType"))
    left_mrp = max(float(left.get("mrp") or 1), 1)
    right_mrp = max(float(right.get("mrp") or 1), 1)
    values = {
        "category": item_type,
        "sleeve": categorical(left.get("sleeve"), right.get("sleeve")),
        "provision": categorical(left.get("provision"), right.get("provision")),
        "pattern": pattern_similarity(left.get("pattern"), right.get("pattern")),
        "range": 1.0 if canonical_plus(left.get("range")) == canonical_plus(right.get("range")) else 0.0,
        "fit": categorical(left.get("fit"), right.get("fit")),
        "fabric": max(categorical(left.get("fabric"), right.get("fabric")), jaccard(left.get("fabric"), right.get("fabric"))),
        "fashion": categorical(left.get("fashion"), right.get("fashion")),
        "colour": colour_similarity(left.get("colour"), right.get("colour")),
        "price": math.exp(-abs(math.log(left_mrp / right_mrp)) / 0.30),
    }
    score = sum(values[name] * weight for name, weight in ATTRIBUTE_WEIGHTS.items())
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


def normalized_demand(item: dict[str, Any], target_sell_through: float = DEFAULT_TARGET_SELL_THROUGH) -> float:
    """Estimate an ideal initial buy while containing inconsistent sample rows.

    Sales divided by the target sell-through is the demand signal. The result is
    winsorized relative to the strongest available supply observation so a bad
    sell-through or dispatch row cannot dominate the model.
    """

    order = max(float(item.get("order") or 0), 0)
    dispatch = max(float(item.get("dispatch") or 0), 0)
    sales = max(float(item.get("sales") or 0), 0)
    supply = max(order, dispatch, 1.0)
    raw = sales / max(target_sell_through, 0.01)
    if sales == 0:
        raw = (order + dispatch) / 2
    return clamp(raw, supply * 0.45, supply * 1.50)


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


class FeatureEncoder:
    fields = ("itemType", "sleeve", "provision", "pattern", "fit", "fashion", "colour")

    def __init__(self, history: list[dict[str, Any]]):
        self.categories: dict[str, list[str]] = {}
        for field in self.fields:
            if field == "pattern":
                values = {PATTERN_FAMILY.get(norm(item.get(field)), norm(item.get(field))) for item in history}
            elif field == "colour":
                values = {COLOUR_FAMILY.get(norm(item.get(field)), norm(item.get(field))) for item in history}
            else:
                values = {norm(item.get(field)) for item in history}
            self.categories[field] = sorted(values)
        fabric_counts: dict[str, int] = {}
        for item in history:
            for token in token_set(item.get("fabric")):
                fabric_counts[token] = fabric_counts.get(token, 0) + 1
        self.fabric_tokens = sorted(token for token, count in fabric_counts.items() if count >= 2)
        prices = np.log([max(float(item.get("mrp") or 1), 1) for item in history])
        self.price_mean = float(prices.mean())
        self.price_scale = float(prices.std()) or 1.0

    def transform_one(self, item: dict[str, Any]) -> list[float]:
        values: list[float] = []
        for field in self.fields:
            raw = norm(item.get(field))
            if field == "pattern":
                raw = PATTERN_FAMILY.get(raw, raw)
            elif field == "colour":
                raw = COLOUR_FAMILY.get(raw, raw)
            categories = self.categories[field]
            values.extend(1.0 if raw == category else 0.0 for category in categories)
            values.append(1.0 if raw not in categories else 0.0)
        fabric = token_set(item.get("fabric"))
        values.extend(1.0 if token in fabric else 0.0 for token in self.fabric_tokens)
        price = math.log(max(float(item.get("mrp") or 1), 1))
        values.append((price - self.price_mean) / self.price_scale)
        lifecycle = norm(item.get("lifecycle"))
        values.extend([1.0 if lifecycle.startswith("SS") else 0.0, 1.0 if lifecycle.startswith("AW") else 0.0])
        return values

    def transform(self, items: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([self.transform_one(item) for item in items], dtype=np.float64)


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
    means = train_x.mean(axis=0)
    scales = train_x.std(axis=0)
    scales[scales < 1e-8] = 1.0
    x = (train_x - means) / scales
    tx = (test_x - means) / scales
    design = np.column_stack([np.ones(len(x)), x])
    test_design = np.column_stack([np.ones(len(tx)), tx])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ train_y
    return test_design @ coefficients


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
    encoder = FeatureEncoder(history)
    features = encoder.transform(history)
    candidate_alphas = (1.0, 10.0, 50.0)
    ridge_by_alpha: dict[float, np.ndarray] = {}
    all_indices = list(range(len(history)))
    for alpha in candidate_alphas:
        predictions = []
        for holdout in all_indices:
            train = [index for index in all_indices if index != holdout]
            prediction = ridge_predict(features[train], targets[train], features[[holdout]], alpha)[0]
            predictions.append(float(prediction))
        ridge_by_alpha[alpha] = np.asarray(predictions)

    best: dict[str, Any] | None = None
    for attribute_weight in (0.40, 0.50, 0.60, 0.70, 0.80):
        for top_k in (3, 5, 8):
            analogue = []
            for holdout in all_indices:
                candidates = [index for index in all_indices if index != holdout]
                prediction, _ = analogue_prediction(
                    holdout,
                    candidates,
                    targets,
                    attribute_matrix,
                    visual_matrix,
                    attribute_weight,
                    top_k,
                )
                analogue.append(prediction)
            analogue_array = np.asarray(analogue)
            for alpha, ridge in ridge_by_alpha.items():
                for regression_blend in (0.15, 0.25, 0.35, 0.50):
                    ensemble = analogue_array * (1 - regression_blend) + ridge * regression_blend
                    score = wape(targets, ensemble)
                    candidate = {
                        "attributeWeight": attribute_weight,
                        "visualWeight": 1 - attribute_weight,
                        "topK": top_k,
                        "ridgeAlpha": alpha,
                        "regressionBlend": regression_blend,
                        "analoguePredictions": analogue_array,
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
    return {
        **{key: value for key, value in best.items() if not isinstance(value, np.ndarray)},
        "predictions": predictions,
        "residuals": residuals,
        "conformalHalfWidth": interval,
        "metrics": {
            "wape": round(wape(targets, predictions), 4),
            "mae": round(float(np.mean(residuals)), 1),
            "bias": round(bias, 4),
            "intervalCoverage": round(coverage, 4),
        },
        "encoder": encoder,
        "features": features,
    }


def recommendation_confidence(
    top_scores: list[float],
    quantity: float,
    interval_half_width: float,
    has_visual: bool,
    issue_count: int,
) -> str:
    top = top_scores[0] if top_scores else 0.0
    mean_top = float(np.mean(top_scores[:3])) if top_scores else 0.0
    relative_width = interval_half_width / max(quantity, 1.0)
    if top >= 0.84 and mean_top >= 0.72 and relative_width <= 0.50 and has_visual and issue_count == 0:
        return "High"
    if top >= 0.62 and mean_top >= 0.52 and relative_width <= 0.90:
        return "Medium"
    return "Low"


def recommend_one(
    item: dict[str, Any],
    history: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    targets: np.ndarray,
    encoder: FeatureEncoder,
    features: np.ndarray,
    model: dict[str, Any],
) -> dict[str, Any]:
    top_k = int(model["topK"])
    selected = matches[:top_k]
    weights = np.asarray([max(float(match["hybridScore"]), 0.01) ** 2 for match in selected])
    history_index = {historical["id"]: index for index, historical in enumerate(history)}
    selected_targets = np.asarray([targets[history_index[match["historicalId"]]] for match in selected])
    analogue = float(np.average(selected_targets, weights=weights))
    item_features = encoder.transform([item])
    regression = float(ridge_predict(features, targets, item_features, float(model["ridgeAlpha"]))[0])
    regression = clamp(regression, MIN_BUY, MAX_BUY)
    blend = float(model["regressionBlend"])
    raw = analogue * (1 - blend) + regression * blend
    quantity = int(clamp(round_pack(raw), MIN_BUY, MAX_BUY))
    top_scores = [float(match["hybridScore"]) for match in selected]
    top_visual_available = bool(selected and selected[0].get("visualScore") is not None)
    interval = float(model["conformalHalfWidth"]) * (1.0 + max(0.0, 0.7 - (top_scores[0] if top_scores else 0.0)))
    issue_count = sum(len(quality_flags(history[history_index[match["historicalId"]]])) for match in selected[:3])
    confidence = recommendation_confidence(top_scores, quantity, interval, top_visual_available, issue_count)
    return {
        "quantity": quantity,
        "low": int(clamp(round_pack(quantity - interval), MIN_BUY, MAX_BUY)),
        "high": int(clamp(round_pack(quantity + interval), MIN_BUY, MAX_BUY)),
        "confidence": confidence,
        "analogueQuantity": round_pack(analogue),
        "regressionQuantity": round_pack(regression),
        "intervalHalfWidth": round_pack(interval),
        "topMatchScore": round(top_scores[0] if top_scores else 0.0, 4),
        "modelVersion": MODEL_VERSION,
    }


def build_model_artifact(source: dict[str, Any], vision_output: dict[str, Any]) -> dict[str, Any]:
    history = [dict(item) for item in source["historical"]]
    upcoming = [dict(item) for item in source["upcoming"]]
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
            attribute, _ = attribute_similarity(left, right)
            attribute_matrix[left_index, right_index] = attribute
            distance = distance_map.get((left["id"], right["id"]))
            visual = calibration.similarity(distance)
            if visual is not None:
                visual_matrix[left_index, right_index] = visual

    targets = np.asarray([normalized_demand(item) for item in history], dtype=np.float64)
    fitted = backtest_model(history, attribute_matrix, visual_matrix, targets)
    for item, target in zip(history, targets):
        item["normalizedDemand"] = round_pack(float(target))
        item["qualityFlags"] = quality_flags(item)

    attribute_weight = float(fitted["attributeWeight"])
    confidence_counts = {"High": 0, "Medium": 0, "Low": 0}
    all_visual_scores: list[float] = []
    for item in upcoming:
        matches: list[dict[str, Any]] = []
        for historical in history:
            attribute, breakdown = attribute_similarity(item, historical)
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
            fitted["encoder"],
            fitted["features"],
            fitted,
        )
        item["modelFlags"] = ["missing_image"] if matches and matches[0]["visualScore"] is None else []
        confidence_counts[item["recommendation"]["confidence"]] += 1

    anomaly_counts = {
        "dispatchAboveOrder": sum("dispatch_above_order" in item["qualityFlags"] for item in history),
        "salesAboveDispatch": sum("sales_above_dispatch" in item["qualityFlags"] for item in history),
        "sellThroughAbove100": sum("sell_through_above_100" in item["qualityFlags"] for item in history),
    }
    meta = dict(source.get("meta", {}))
    meta.update({
        "title": "Turtle Season Intelligence AI",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "confidenceCounts": confidence_counts,
        "visualScoreRange": [round(min(all_visual_scores), 3), round(max(all_visual_scores), 3)] if all_visual_scores else [0, 0],
        "visualMethod": vision_output.get("engine", "FashionCLIP image embedding"),
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
            "algorithm": "Calibrated FashionCLIP retrieval + regularized demand ensemble",
            "attributeWeight": round(attribute_weight, 2),
            "visualWeight": round(1 - attribute_weight, 2),
            "topK": int(fitted["topK"]),
            "regressionBlend": float(fitted["regressionBlend"]),
            "ridgeAlpha": float(fitted["ridgeAlpha"]),
            "backtest": fitted["metrics"],
            "evaluation": "Leave-one-out validation; temporal holdout requires at least three clean seasons",
            "interval": "Finite-sample 80% conformal interval from out-of-fold residuals",
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
