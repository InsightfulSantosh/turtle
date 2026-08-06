from __future__ import annotations

import math

from machine_learning.demand import (
    DemandPolicy,
    cleaned_sales,
    corrected_weekly_rate,
    expected_sales_at,
    fit_demand_priors,
    is_censored,
    newsvendor_order,
    predict_demand,
    weeks_on_floor,
)
from machine_learning.model import recommend_one


def history_row(
    identifier: str,
    *,
    order: int,
    dispatch: int,
    sales: int,
    ageing_days: int = 266,
    item_type: str = "OTSH",
    category: str = "CASUAL",
) -> dict:
    return {
        "id": identifier,
        "itemType": item_type,
        "categoryType": category,
        "order": order,
        "dispatch": dispatch,
        "sales": sales,
        "sellThrough": sales / dispatch if dispatch else 0.0,
        "ageingDays": ageing_days,
    }


def synthetic_history(count: int = 40) -> list[dict]:
    """Build a spread of products so priors have something to fit."""

    return [
        history_row(
            f"H{index}",
            order=600 + 20 * index,
            dispatch=600 + 20 * index,
            sales=int((600 + 20 * index) * (0.45 + 0.01 * (index % 20))),
        )
        for index in range(count)
    ]


def test_newsvendor_reduces_to_the_legacy_rule_without_uncertainty() -> None:
    """The legacy divide-by-target formula is the zero-spread special case."""

    order = newsvendor_order(math.log(700), 0.0, 0.70)

    assert round(order) == 1000


def test_expected_sales_saturate_above_demand() -> None:
    """Ordering more cannot sell more than demand exists for."""

    assert expected_sales_at(500, math.log(700), 0.0) == 500
    assert math.isclose(expected_sales_at(5_000, math.log(700), 0.0), 700)


def test_uncertainty_moves_the_buy_away_from_the_legacy_answer() -> None:
    """A real predictive spread must not return the naive division."""

    naive = newsvendor_order(math.log(700), 0.0, 0.70)
    uncertain = newsvendor_order(math.log(700), 0.60, 0.70)

    assert uncertain != naive
    assert expected_sales_at(uncertain, math.log(700), 0.60) / uncertain == 0.70


def test_exposure_normalisation_separates_equal_sales() -> None:
    """Identical sales over different selling windows are different demand."""

    quick = history_row("QUICK", order=1000, dispatch=1000, sales=500, ageing_days=100)
    slow = history_row("SLOW", order=1000, dispatch=1000, sales=500, ageing_days=380)

    assert cleaned_sales(quick) == cleaned_sales(slow)
    assert weeks_on_floor(quick, 38.0) < weeks_on_floor(slow, 38.0)


def test_stocked_out_products_are_lifted_above_observed_sales() -> None:
    """A sell-out is a censored lower bound on demand, not demand."""

    history = synthetic_history()
    sold_out = history_row("SOLDOUT", order=500, dispatch=500, sales=499)
    priors = fit_demand_priors(history + [sold_out])

    assert is_censored(sold_out, priors.policy)
    corrected, censored = corrected_weekly_rate(sold_out, priors)
    observed = cleaned_sales(sold_out) / weeks_on_floor(sold_out, priors.horizon_weeks)

    assert censored is True
    assert corrected > observed


def test_uncensored_products_keep_their_observed_rate() -> None:
    history = synthetic_history()
    priors = fit_demand_priors(history)
    steady = history_row("STEADY", order=1000, dispatch=1000, sales=600)

    corrected, censored = corrected_weekly_rate(steady, priors)

    assert censored is False
    assert corrected == cleaned_sales(steady) / weeks_on_floor(steady, priors.horizon_weeks)


def test_thin_evidence_shrinks_toward_the_group_prior() -> None:
    """One weak analogue must not be trusted like four strong ones."""

    history = synthetic_history()
    priors = fit_demand_priors(history)
    by_id = {item["id"]: item for item in history}
    item = {"id": "NEW", "itemType": "OTSH", "categoryType": "CASUAL"}

    single = predict_demand(item, [{"historicalId": "H0", "visualScore": 0.55}], by_id, priors)
    several = predict_demand(
        item,
        [{"historicalId": f"H{index}", "visualScore": 0.90} for index in range(4)],
        by_id,
        priors,
    )

    assert single is not None and several is not None
    assert single.shrinkage_weight < several.shrinkage_weight
    assert single.effective_sample_size < several.effective_sample_size


def test_prediction_without_usable_analogues_is_none() -> None:
    priors = fit_demand_priors(synthetic_history())
    item = {"id": "NEW", "itemType": "OTSH", "categoryType": "CASUAL"}

    assert predict_demand(item, [], {}, priors) is None
    assert predict_demand(item, [{"historicalId": "missing", "visualScore": 0.9}], {}, priors) is None


def test_predictive_recommendation_is_not_a_copy_of_the_analogue() -> None:
    """The reported defect: forecast and analogue sales were the same number."""

    history = synthetic_history()
    priors = fit_demand_priors(history)
    matches = [{"historicalId": f"H{index}", "visualScore": 0.88, "hybridScore": 0.88} for index in range(4)]
    item = {"id": "NEW", "itemType": "OTSH", "categoryType": "CASUAL"}

    legacy = recommend_one(item, history, matches, {"topK": 4})
    predictive = recommend_one(item, history, matches, {"topK": 4}, demand_priors=priors)

    assert legacy["expectedSales"] == legacy["analogueSales"]
    assert predictive["expectedSales"] != predictive["analogueSales"]
    assert predictive["evidencePolicy"] == "pooled_visual_analogue_forecast"


def test_predictive_recommendation_publishes_a_real_interval() -> None:
    """Legacy set low == high == quantity, which is the tell of a lookup."""

    history = synthetic_history()
    priors = fit_demand_priors(history)
    matches = [{"historicalId": f"H{index}", "visualScore": 0.88, "hybridScore": 0.88} for index in range(4)]
    item = {"id": "NEW", "itemType": "OTSH", "categoryType": "CASUAL"}

    result = recommend_one(item, history, matches, {"topK": 4}, demand_priors=priors)

    assert result["salesLow"] < result["expectedSales"] < result["salesHigh"]
    assert result["low"] < result["quantity"] < result["high"]
    assert result["demand"]["analoguesUsed"] == 4
    assert 0.0 < result["demand"]["analogueWeight"] <= 1.0
    # Well-supported evidence: mean and median should sit close together and
    # the wide-uncertainty caveat must not fire on a confident forecast.
    assert result["demand"]["skewRatio"] < 1.5
    assert result["demand"]["wideUncertainty"] is False
    assert result["demand"]["medianDemand"] <= result["expectedSales"]


def test_wide_uncertainty_flags_thin_or_conflicting_evidence() -> None:
    """The reported case: mean forecast far above the order looked like a bug.

    One weak, borderline-similarity analogue in a tiny, thinly-populated group
    that disagrees sharply with the wider catalogue must be flagged, not
    presented with the same confidence as a well-supported forecast.
    """

    # A group with a genuinely wide sales spread, so its own prior carries
    # real variance (not floor-clamped), and only one weak analogue backing
    # the new item — mirroring a small item type with erratic history.
    sales_values = [15, 30, 80, 250, 700, 1_800, 4_000, 6_000]
    history = [
        history_row(f"G{index}", order=8_000, dispatch=8_000, sales=sales, item_type="OTGL", category="FORMAL")
        for index, sales in enumerate(sales_values)
    ]
    priors = fit_demand_priors(history)
    item = {"id": "NEW", "itemType": "OTGL", "categoryType": "FORMAL"}
    matches = [{"historicalId": "G0", "visualScore": 0.51, "hybridScore": 0.51}]

    result = recommend_one(item, history, matches, {"topK": 4}, demand_priors=priors)

    assert result["demand"]["analoguesUsed"] == 1
    assert result["demand"]["skewRatio"] >= 1.5
    assert result["demand"]["wideUncertainty"] is True
    # The point this exists to catch: the mean can sit far above the typical
    # (median) outcome, which is what made the raw headline look abnormal.
    assert result["demand"]["medianDemand"] < result["expectedSales"]


def test_predictive_path_respects_the_no_match_policy() -> None:
    """Weak visual evidence must still route to manual review, not a forecast."""

    history = synthetic_history()
    priors = fit_demand_priors(history)
    matches = [{"historicalId": "H0", "visualScore": 0.10, "hybridScore": 0.10}]
    item = {"id": "NEW", "itemType": "OTSH", "categoryType": "CASUAL"}

    result = recommend_one(item, history, matches, {"topK": 4}, demand_priors=priors)

    assert result["noSuitableMatch"] is True
    assert result["quantity"] == 0
    assert result["expectedSales"] == 0
    assert "demand" not in result


def test_sell_through_target_moves_the_buy() -> None:
    """The planner's dial must actually control the outcome."""

    history = synthetic_history()
    priors = fit_demand_priors(history)
    matches = [{"historicalId": f"H{index}", "visualScore": 0.88, "hybridScore": 0.88} for index in range(4)]
    item = {"id": "NEW", "itemType": "OTSH", "categoryType": "CASUAL"}

    conservative = recommend_one(item, history, matches, {}, target_sell_through=0.85, demand_priors=priors)
    aggressive = recommend_one(item, history, matches, {}, target_sell_through=0.55, demand_priors=priors)

    assert aggressive["quantity"] > conservative["quantity"]


def test_priors_need_uncensored_history() -> None:
    policy = DemandPolicy()
    sold_out = [history_row(f"S{index}", order=500, dispatch=500, sales=500) for index in range(5)]

    try:
        fit_demand_priors(sold_out, policy)
    except ValueError as error:
        assert "demand priors" in str(error)
    else:  # pragma: no cover - guard must raise
        raise AssertionError("expected fully censored history to be rejected")
