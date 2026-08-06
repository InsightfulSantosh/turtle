from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from machine_learning.demand import (
    BuyCeilings,
    DemandPriors,
    annotate_history_rates,
    clamp,
    cleaned_sales,
    fit_buy_ceilings,
    fit_demand_priors,
    newsvendor_order,
    predict_demand,
    round_pack,
    serialize_buy_ceilings,
    serialize_priors,
)

MODEL_VERSION = "5.1.0"
# Fallback only, used when no data-derived BuyCeilings is available (the
# legacy no-forecast path, or a caller that didn't fit one). The live
# predictive path uses a per-item-type ceiling from fit_buy_ceilings instead:
# a flat number cannot be right for every item type when this catalogue's own
# per-item-type historical maxima range from ~150 to ~6,700 units.
MAX_BUY = 2_000
MIN_CONVINCING_VISUAL_SCORE = 0.50
DEFAULT_TARGET_SELL_THROUGH = 0.70


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

    Sales above both order and dispatch cannot be demand the business served,
    so they are capped at the strongest observable supply value.
    """

    return cleaned_sales(item)


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


def match_confidence(
    top_scores: list[float],
    has_visual: bool,
    issue_count: int,
    minimum_visual_score: float = MIN_CONVINCING_VISUAL_SCORE,
) -> str:
    """Rate analogue relevance without mixing in forecast-range width."""

    top = top_scores[0] if top_scores else 0.0
    displayed_top = math.floor(top * 100 + 0.5)
    displayed_minimum = math.floor(minimum_visual_score * 100 + 0.5)
    if top >= 0.84 and has_visual and issue_count == 0:
        return "High"
    if displayed_top >= displayed_minimum and has_visual:
        return "Medium"
    return "Low"


def no_suitable_product_match(
    selected: list[dict[str, Any]],
    relevance: str,
    minimum_visual_score: float = MIN_CONVINCING_VISUAL_SCORE,
) -> bool:
    """Reject weak or non-visual candidates instead of presenting a false match."""

    if not selected or relevance == "Low":
        return True
    visual_score = selected[0].get("visualScore")
    if visual_score is None:
        return True
    displayed_score = math.floor(float(visual_score) * 100 + 0.5)
    displayed_minimum = math.floor(minimum_visual_score * 100 + 0.5)
    return displayed_score < displayed_minimum


def recommend_one(
    item: dict[str, Any],
    history: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    model: dict[str, Any],
    *,
    target_sell_through: float = DEFAULT_TARGET_SELL_THROUGH,
    minimum_visual_score: float = MIN_CONVINCING_VISUAL_SCORE,
    demand_priors: DemandPriors | None = None,
    buy_ceilings: BuyCeilings | None = None,
) -> dict[str, Any]:
    """Calculate one buy from the accepted visual analogue evidence.

    With ``demand_priors`` omitted this keeps the legacy single-analogue rule:
    republish the top analogue's cleaned sales and divide by the sell-through
    target. Supplying priors switches on the predictive estimator, which pools
    the accepted analogues, corrects for stock-outs and exposure, and solves
    the buy against the target under uncertainty.

    ``buy_ceilings`` supplies the data-derived, per-item-type sanity cap on
    the forecast and order (see ``demand.fit_buy_ceilings``). Without it, the
    flat ``MAX_BUY`` fallback applies.
    """

    # Rounded once to an int and reused everywhere below (clamping and the
    # reported buyCeiling both read this exact value), so a float ceiling
    # that isn't perfectly integral (e.g. 5700 * 0.35) can't disagree with
    # itself by rounding one way when clamping and another when reporting.
    ceiling = int(round(buy_ceilings.ceiling_for(item))) if buy_ceilings is not None else MAX_BUY

    selected = matches[:1]
    history_by_id = {str(historical["id"]): historical for historical in history}
    top_scores = [float(match["hybridScore"]) for match in selected]
    top_visual_available = bool(selected and selected[0].get("visualScore") is not None)
    selected_history = history_by_id.get(str(selected[0]["historicalId"])) if selected else None
    issue_count = len(quality_flags(selected_history)) if selected_history is not None else 0
    relevance = match_confidence(
        top_scores,
        top_visual_available,
        issue_count,
        minimum_visual_score,
    )
    no_suitable_match = no_suitable_product_match(
        selected,
        relevance,
        minimum_visual_score,
    )
    historical_order = 0.0
    analogue_sales = 0
    prediction = None
    quantity_capped = False
    high_capped = False
    if no_suitable_match or selected_history is None:
        expected_sales = 0
        quantity = 0
        sales_low = sales_high = 0
        order_low = order_high = 0
    else:
        historical_order = max(float(selected_history.get("order") or 0), 0.0)
        # Historical evidence is an observed fact, not a forecast that needs a
        # sanity bound — capping it here would silently understate a real
        # analogue's own sales (this catalogue's top sellers exceed 5,000
        # units, well past the legacy flat cap).
        analogue_sales = int(max(0, round_pack(sales_target(selected_history))))
        accepted = [
            match
            for match in matches
            if match.get("visualScore") is not None
            and math.floor(float(match["visualScore"]) * 100 + 0.5)
            >= math.floor(minimum_visual_score * 100 + 0.5)
        ]
        prediction = (
            predict_demand(item, accepted, history_by_id, demand_priors)
            if demand_priors is not None and accepted
            else None
        )
        if prediction is None:
            # Legacy rule: republish the analogue's own sales as the forecast.
            expected_sales = analogue_sales
            sales_low = sales_high = expected_sales
            raw_quantity = round_pack(expected_sales / max(target_sell_through, 0.01))
            quantity = int(clamp(raw_quantity, 0, ceiling))
            quantity_capped = raw_quantity > ceiling
            order_low = order_high = quantity
            high_capped = quantity_capped
        else:
            # point_demand, not the raw mean: the mean is inflated by a sigma
            # that was tuned for honest interval coverage, which overshoots as
            # a point estimate. See DemandPolicy.point_estimate_skew.
            expected_sales = int(clamp(round_pack(prediction.point_demand), 0, ceiling))
            sales_low = int(clamp(round_pack(prediction.p10), 0, ceiling))
            sales_high = int(clamp(round_pack(prediction.p90), 0, ceiling))
            # Raw (uncapped) solves are kept alongside the clamped ones so the
            # caller can tell a genuine forecast apart from a number the buy
            # ceiling forced, rather than silently presenting both the same
            # way. Before this ceiling was made data-derived, the flat 2,000
            # fallback bound on ~9% of products even at the default 70%
            # target, and the large majority at low targets — so a target
            # sell-through setting could look "wrong" for reasons that were
            # actually the cap, not the forecast.
            raw_quantity = round_pack(
                newsvendor_order(prediction.log_mu, prediction.log_sigma, target_sell_through)
            )
            quantity = int(clamp(raw_quantity, 0, ceiling))
            quantity_capped = raw_quantity > ceiling
            # Order band: what the buy would be if demand landed at the low or
            # high end of the predicted range, not a cosmetic +/- on quantity.
            order_low = int(
                clamp(
                    round_pack(
                        newsvendor_order(
                            math.log(max(prediction.p10, 1e-6)),
                            prediction.log_sigma,
                            target_sell_through,
                        )
                    ),
                    0,
                    ceiling,
                )
            )
            raw_high = round_pack(
                newsvendor_order(
                    math.log(max(prediction.p90, 1e-6)),
                    prediction.log_sigma,
                    target_sell_through,
                )
            )
            order_high = int(clamp(raw_high, 0, ceiling))
            high_capped = raw_high > ceiling

    result = {
        "quantity": quantity,
        "low": order_low,
        "high": order_high,
        "expectedSales": expected_sales,
        "salesLow": sales_low,
        "salesHigh": sales_high,
        "matchConfidence": relevance,
        "noSuitableMatch": no_suitable_match,
        "confidence": relevance,
        "analogueSales": analogue_sales,
        # Historical evidence, not capped — see the analogueSales comment above.
        "analogueQuantity": int(max(0, round_pack(historical_order))) if selected_history else 0,
        "targetSellThrough": target_sell_through,
        "minimumVisualScore": minimum_visual_score,
        "evidencePolicy": "single_top_visual_analogue",
        "topMatchScore": round(top_scores[0] if top_scores else 0.0, 4),
        "modelVersion": MODEL_VERSION,
        # True when the model's own solve wanted more than the buy ceiling
        # and was truncated. A capped quantity is a policy limit, not
        # evidence that the target sell-through was reached — surface it
        # rather than let a hit cap masquerade as a well-calibrated number.
        "quantityCapped": quantity_capped,
        "highCapped": high_capped,
        "buyCeiling": ceiling,
    }
    if prediction is not None:
        result["evidencePolicy"] = "pooled_visual_analogue_forecast"
        # mean/median diverge whenever the distribution is right-skewed
        # (mean/median = exp(logSigma^2/2) for a lognormal), which is exactly
        # when a single headline number stops being self-explanatory: the mean
        # is the statistically correct point estimate to buy against (it is
        # what the newsvendor solve and the validated backtest both use), but
        # it can sit well above the *typical* outcome a planner would expect.
        median_demand = int(clamp(round_pack(prediction.median_demand), 0, ceiling))
        skew_ratio = prediction.mean_demand / max(prediction.median_demand, 1e-6)
        policy = demand_priors.policy if demand_priors is not None else None
        wide_uncertainty = policy is not None and (
            prediction.effective_sample_size < policy.wide_uncertainty_effective_n
            or skew_ratio >= policy.wide_uncertainty_skew_ratio
        )
        # Diagnostics a planner can audit: how much of the number came from the
        # analogues versus the group prior, and how thin that evidence was.
        result["demand"] = {
            "weeklyRate": round(prediction.weekly_rate, 3),
            "horizonWeeks": round(prediction.horizon_weeks, 1),
            "analoguesUsed": prediction.analogues_used,
            "effectiveSampleSize": round(prediction.effective_sample_size, 2),
            "analogueWeight": round(prediction.shrinkage_weight, 3),
            "priorWeeklyRate": round(prediction.prior_weekly_rate, 3),
            "censoredAnalogues": prediction.censored_analogues,
            # medianDemand is the typical (P50) outcome; expectedSales above
            # stays the mean, because that is the unbiased point estimate the
            # order quantity and the validated backtest are built on. Showing
            # both, plus the ratio between them, is what makes a skewed case
            # like "expectedSales far above quantity" self-explanatory instead
            # of looking like a bug.
            "medianDemand": median_demand,
            "skewRatio": round(skew_ratio, 3),
            "wideUncertainty": wide_uncertainty,
            # The predictive distribution itself, so the UI can re-solve the
            # newsvendor when the planner moves the sell-through slider
            # instead of falling back to the legacy division.
            "logMu": round(prediction.log_mu, 6),
            "logSigma": round(prediction.log_sigma, 6),
        }
    return result


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


def blend_hybrid_visual_scores(
    fashion_score: float | None,
    dino_score: float | None,
    colour_score: float | None,
    texture_score: float | None,
    dino_weight: float,
    component_weights: dict[str, float],
) -> float | None:
    """Combine neural, colour and texture evidence without penalising missing signals."""

    neural_score = blend_two_stage_visual_scores(
        fashion_score,
        dino_score,
        dino_weight,
    )
    components = (
        (neural_score, component_weights["neural"]),
        (colour_score, component_weights["colour"]),
        (texture_score, component_weights["texture"]),
    )
    available = [(score, weight) for score, weight in components if score is not None and weight > 0]
    if not available:
        return None
    total_weight = sum(weight for _, weight in available)
    return round(sum(score * weight for score, weight in available) / total_weight, 4)


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
        and row.get(distance_key) is not None
        and (not exclude_self or row["leftId"] != row["rightId"])
    ]


def build_model_artifact(
    source: dict[str, Any],
    vision_output: dict[str, Any],
) -> dict[str, Any]:
    """Build the frontend artifact using the pooled predictive demand estimator.

    Per-item, ``recommend_one`` still falls back to the legacy copy-one-analogue
    rule whenever a product has no accepted visual analogue to pool from — that
    fallback is a property of the evidence available for that item, not a
    switch. There used to be a whole-artifact ``demand_forecast`` flag that
    could disable pooling everywhere; it defaulted to off, which meant the
    artifact people actually rely on only existed when someone remembered to
    pass a flag. Removed rather than fixed, since the off state was never
    the intended behavior.
    """

    history = [dict(item) for item in source["historical"]]
    upcoming = [dict(item) for item in source["upcoming"]]
    rows = vision_output.get("distances", [])
    candidate_rows = list(vision_output.get("candidatePairs", []))
    historical_ids = {str(item["id"]) for item in history}
    upcoming_ids = {str(item["id"]) for item in upcoming}
    targets = np.asarray([sales_target(item) for item in history], dtype=np.float64)
    candidate_history_ids: dict[str, set[str]] = {}
    candidate_pair_map: dict[tuple[str, str], dict[str, Any]] = {}
    distance_map: dict[tuple[str, str], float] = {}
    dino_historical_calibration: VisionCalibration | None = None
    dino_serving_calibration: VisionCalibration | None = None
    colour_historical_calibration: VisionCalibration | None = None
    colour_serving_calibration: VisionCalibration | None = None
    texture_historical_calibration: VisionCalibration | None = None
    texture_serving_calibration: VisionCalibration | None = None
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
        colour_historical_values = _candidate_calibration_values(
            candidate_rows,
            left_ids=historical_ids,
            right_ids=historical_ids,
            distance_key="colourDistance",
            exclude_self=True,
        )
        colour_serving_values = _candidate_calibration_values(
            candidate_rows,
            left_ids=upcoming_ids,
            right_ids=historical_ids,
            distance_key="colourDistance",
        )
        texture_historical_values = _candidate_calibration_values(
            candidate_rows,
            left_ids=historical_ids,
            right_ids=historical_ids,
            distance_key="textureDistance",
            exclude_self=True,
        )
        texture_serving_values = _candidate_calibration_values(
            candidate_rows,
            left_ids=upcoming_ids,
            right_ids=historical_ids,
            distance_key="textureDistance",
        )
        historical_calibration = calibrate_vision(fashion_historical_values)
        serving_calibration = calibrate_vision(fashion_serving_values or fashion_historical_values)
        dino_historical_calibration = calibrate_vision(dino_historical_values)
        dino_serving_calibration = calibrate_vision(dino_serving_values or dino_historical_values)
        colour_historical_calibration = calibrate_vision(colour_historical_values)
        colour_serving_calibration = calibrate_vision(colour_serving_values or colour_historical_values)
        texture_historical_calibration = calibrate_vision(texture_historical_values)
        texture_serving_calibration = calibrate_vision(texture_serving_values or texture_historical_values)
        component_weights = {
            "neural": 0.70,
            "colour": 0.20,
            "texture": 0.10,
            **vision_output.get("reranker", {}).get("appearance", {}).get("weights", {}),
        }
        if set(component_weights) != {"neural", "colour", "texture"}:
            raise ValueError("reranker appearance weights must contain neural, colour and texture")
        if any(weight < 0 for weight in component_weights.values()) or not math.isclose(
            sum(component_weights.values()),
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError("reranker appearance weights must be non-negative and sum to 1")
        weight_grid = tuple(
            float(weight)
            for weight in vision_output.get("reranker", {}).get(
                "weightGrid",
                (0.0, 0.25, 0.5, 0.75, 1.0),
            )
        )
        # The client-approved visual model uses the configured visual blend
        # directly. There is no attribute/forecast model-selection stage.
        selected_dino_weight = weight_grid[0]
        for row in candidate_rows:
            if str(row["leftId"]) in upcoming_ids and str(row["rightId"]) in historical_ids:
                left_id = str(row["leftId"])
                right_id = str(row["rightId"])
                candidate_history_ids.setdefault(left_id, set()).add(right_id)
                candidate_pair_map[(left_id, right_id)] = row
    else:
        distance_map = {(str(row["leftId"]), str(row["rightId"])): float(row["distance"]) for row in rows}
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
    for item, target in zip(history, targets, strict=True):
        # Retain enough precision for API-side reproduction after loading the
        # artifact. Sales and final order recommendations are pack-rounded later.
        item["salesTarget"] = round(float(target), 4)
        item["qualityFlags"] = quality_flags(item)

    # Priors are fitted once from the whole historical catalogue, before any
    # upcoming product is scored, so every buy shares one auditable prior.
    demand_priors = fit_demand_priors(history)
    annotate_history_rates(history, demand_priors)
    # Data-derived sanity ceiling per item type, replacing the flat MAX_BUY
    # fallback — see BuyCeilings for why a single constant cannot be right
    # for every item type on this catalogue.
    buy_ceilings = fit_buy_ceilings(history)
    match_confidence_counts = {"High": 0, "Medium": 0, "Low": 0}
    all_visual_scores: list[float] = []
    retrieval_history = [historical for historical in history if historical.get("imageUrl")] or history
    for item in upcoming:
        matches: list[dict[str, Any]] = []
        shortlist_ids = candidate_history_ids.get(str(item["id"]), set())
        # In two-stage retrieval, an absent shortlist means the product did not
        # pass a mandatory retrieval constraint (for example same item type and
        # design). Do not silently replace that true no-match with the whole
        # historical catalogue.
        eligible_history = (
            [historical for historical in retrieval_history if str(historical["id"]) in shortlist_ids]
            if candidate_rows
            else retrieval_history
        )
        for historical in eligible_history:
            fashion_visual: float | None = None
            dino_visual: float | None = None
            colour_visual: float | None = None
            texture_visual: float | None = None
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
                    if candidate.get("colourDistance") is not None:
                        assert colour_serving_calibration is not None
                        colour_visual = colour_serving_calibration.similarity(
                            float(candidate["colourDistance"]),
                        )
                    if candidate.get("textureDistance") is not None:
                        assert texture_serving_calibration is not None
                        texture_visual = texture_serving_calibration.similarity(
                            float(candidate["textureDistance"]),
                        )
                visual = blend_hybrid_visual_scores(
                    fashion_visual,
                    dino_visual,
                    colour_visual,
                    texture_visual,
                    selected_dino_weight,
                    component_weights,
                )
            else:
                fashion_visual = serving_calibration.similarity(
                    distance_map.get((item["id"], historical["id"])),
                )
                visual = fashion_visual
            if visual is not None:
                all_visual_scores.append(visual)
            matches.append(
                {
                    "historicalId": historical["id"],
                    "visualScore": visual,
                    "fashionVisualScore": fashion_visual,
                    "dinoVisualScore": dino_visual,
                    "colourVisualScore": colour_visual,
                    "textureVisualScore": texture_visual,
                    "hybridScore": round(float(visual or 0.0), 4),
                }
            )
        matches.sort(key=lambda match: match["hybridScore"], reverse=True)
        item["recommendation"] = recommend_one(
            item,
            history,
            matches,
            {"topK": 4},
            demand_priors=demand_priors,
            buy_ceilings=buy_ceilings,
        )
        item["matches"] = matches[:4]
        model_flags = list(item.get("modelFlags", []))
        if matches and matches[0]["visualScore"] is None:
            model_flags.append("missing_image")
        if item["recommendation"]["noSuitableMatch"]:
            model_flags.append("no_suitable_match")
        item["modelFlags"] = list(dict.fromkeys(model_flags))
        match_confidence_counts[item["recommendation"]["matchConfidence"]] += 1

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
            "visualScoreRange": (
                [
                    round(min(all_visual_scores), 3),
                    round(max(all_visual_scores), 3),
                ]
                if all_visual_scores
                else [0, 0]
            ),
            "visualMethod": vision_output.get("engine", "FashionCLIP image embedding"),
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
                "status": (
                    "Visual retrieval + pooled demand forecast"
                    if demand_priors is not None
                    else "Visual-only single-analogue decision model"
                ),
                "algorithm": (
                    "FashionSigLIP retrieval + multi-scale DINO, dominant-palette "
                    "CIEDE2000 colour and pattern gates + texture reranking"
                ),
                "evidencePolicy": (
                    "pooled_visual_analogue_forecast"
                    if demand_priors is not None
                    else "single_top_visual_analogue"
                ),
                "salesPolicy": (
                    "Similarity-weighted, censoring-corrected demand pooled across accepted "
                    "analogues and shrunk toward the item-type/category prior"
                    if demand_priors is not None
                    else "Use cleaned sales from the single accepted historical visual analogue"
                ),
                "orderPolicy": (
                    "Newsvendor buy whose expected sell-through equals the planner target "
                    "under the predicted demand distribution"
                    if demand_priors is not None
                    else "Divide the selected analogue's cleaned sales by the target sell-through"
                ),
                "noMachineLearningForecast": demand_priors is None,
                "noAttributeMatching": True,
                "visualOnlyRanking": True,
                "dinoRerankWeight": round(selected_dino_weight, 2),
                "minimumVisualScore": MIN_CONVINCING_VISUAL_SCORE,
                "minimumMatchConfidence": "Medium",
                "noMatchPolicy": (
                    "Return no product match, zero system quantity and manual review when the "
                    "single best candidate lacks visual evidence or falls below the threshold."
                ),
                "targetSellThrough": DEFAULT_TARGET_SELL_THROUGH,
                "topK": 4,
                **(
                    {"demandModel": serialize_priors(demand_priors)}
                    if demand_priors is not None
                    else {}
                ),
                **(
                    {"buyCeilings": serialize_buy_ceilings(buy_ceilings)}
                    if buy_ceilings is not None
                    else {}
                ),
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
                    round(dino_historical_calibration.median, 4) if dino_historical_calibration is not None else None
                ),
                "dinoServingMedianDistance": (
                    round(dino_serving_calibration.median, 4) if dino_serving_calibration is not None else None
                ),
                "colourHistoricalMedianDistance": (
                    round(colour_historical_calibration.median, 4)
                    if colour_historical_calibration is not None
                    else None
                ),
                "colourServingMedianDistance": (
                    round(colour_serving_calibration.median, 4) if colour_serving_calibration is not None else None
                ),
                "textureHistoricalMedianDistance": (
                    round(texture_historical_calibration.median, 4)
                    if texture_historical_calibration is not None
                    else None
                ),
                "textureServingMedianDistance": (
                    round(texture_serving_calibration.median, 4) if texture_serving_calibration is not None else None
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
