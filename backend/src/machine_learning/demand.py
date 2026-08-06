"""Predictive demand estimation for upcoming products.

The legacy rule copied one historical row: it published that row's cleaned
sales as the forecast and divided by the sell-through target to get a buy.
That is retrieval plus arithmetic, so the "recommendation" reproduced the
analogue's own order whenever the analogue sold through near target.

This module replaces the arithmetic with an estimator in three stages.

1. ``corrected_weekly_rate`` converts each historical product into a demand
   *rate* (units per week) and lifts rows that stocked out, because their
   observed sales are a censored lower bound on demand rather than demand.
2. ``predict_demand`` pools every accepted analogue by similarity and shrinks
   the pooled rate toward its item-type/category prior, so a thin or weak
   analogue set falls back on the group instead of inheriting one row.
3. ``newsvendor_order`` solves the buy against the planner's sell-through
   target under the predictive distribution. With zero spread it reduces
   exactly to the legacy ``demand / target`` formula, so the existing
   business lever keeps its meaning.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

DAYS_PER_WEEK = 7.0
P10_Z = 1.2815515655446004
# Garments are ordered and packed in fixed lot sizes; every buy quantity in
# this catalogue is rounded to a multiple of this.
PACK_SIZE = 25


@dataclass(frozen=True)
class DemandPolicy:
    """Tunable estimator settings kept out of the calculation code."""

    censor_sell_through: float = 0.95
    # Not a hard cutoff: a group's own fit is blended toward its parent with
    # weight rows/(rows + group_shrinkage_rows), so a category with, say, 7
    # rows still keeps most of its own signal instead of a cliff at 8
    # discarding it outright in favour of a much noisier parent.
    group_shrinkage_rows: float = 8.0
    similarity_exponent: float = 4.0
    # Calibrated on the leave-one-out backtest: the sigma floor is what stops a
    # single strong analogue from implying a precision the data cannot support.
    minimum_log_sigma: float = 0.55
    maximum_log_sigma: float = 1.50
    fallback_horizon_weeks: float = 38.0
    maximum_analogues: int = 4
    # Below this effective analogue count, or above this mean/median skew
    # ratio, a lognormal forecast's mean stops being representative of the
    # likely outcome (mean/median = exp(sigma^2/2), so this grows with
    # uncertainty alone). 1.5 sits almost exactly at sigma ~= 0.9 — comfortably
    # above the calibrated floor (1.16x at minimum_log_sigma) and well below
    # the ceiling (2.7x at maximum_log_sigma) — so it only fires on genuinely
    # thin or conflicting evidence, not on ordinary uncertainty.
    wide_uncertainty_effective_n: float = 2.0
    wide_uncertainty_skew_ratio: float = 1.5
    # A statistical sanity backstop against a degenerate model output, scaled
    # per item type from what has actually been observed — not a substitute
    # for a real manufacturing capacity limit, which can only come from the
    # business. A flat catalogue-wide cap cannot be right for every item
    # type: on this catalogue, per-item-type historical maxima range from
    # ~150 to ~6,700 units, a 44x spread.
    #
    # 1.25x sits above every order and sales figure this catalogue has ever
    # recorded, so the ceiling only ever catches a degenerate model output —
    # it never truncates an outcome the business has actually achieved. An
    # earlier 0.35x value scored marginally better on WMAPE but did so by
    # clipping below real observed sales for 16.7% of historical products
    # (6 of 7 OTWC products), which is a systematic distortion rather than a
    # sanity bound. Forecast skew is corrected at its source instead, via
    # point_estimate_skew below.
    buy_ceiling_multiplier: float = 1.25
    buy_ceiling_floor: float = 500.0
    # The lognormal mean, exp(mu + sigma^2/2), is highly sensitive to sigma —
    # and sigma here is deliberately inflated (floor 0.55, ceiling 1.50) to
    # make the p10-p90 interval cover honestly. Using that same inflated sigma
    # to build a *point* estimate overshoots: it pushed portfolio bias to +67
    # units on the leave-one-out backtest. This exponent scales the skew
    # correction (1.0 = full mean, 0.0 = median); 0.75 is the calibrated
    # near-zero-bias point.
    point_estimate_skew: float = 0.75


DEFAULT_POLICY = DemandPolicy()


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def round_pack(value: float, pack: int = PACK_SIZE) -> int:
    return int(round(value / pack) * pack)


def cleaned_sales(item: dict[str, Any]) -> float:
    """Return observed unit sales capped by the strongest observable supply.

    Sales above both order and dispatch cannot be demand that the business
    actually served, so they are treated as a data fault rather than signal.
    """

    order = max(float(item.get("order") or 0), 0.0)
    dispatch = max(float(item.get("dispatch") or 0), 0.0)
    sales = max(float(item.get("sales") or 0), 0.0)
    supply = max(order, dispatch)
    return min(sales, supply) if supply > 0 else sales


@dataclass(frozen=True)
class BuyCeilings:
    """Data-derived sanity ceiling per item type.

    Replaces a fixed constant with a value that scales with what each item
    type has actually shown it can sell or be ordered in, and that recomputes
    automatically every time the artifact is rebuilt from new data — instead
    of relying on someone to remember to update a hardcoded number as the
    business changes.
    """

    by_item_type: dict[str, float]
    global_ceiling: float
    multiplier: float
    floor: float

    def ceiling_for(self, item: dict[str, Any]) -> float:
        item_type = str(item.get("itemType") or "")
        return self.by_item_type.get(item_type, self.global_ceiling)


def fit_buy_ceilings(
    history: Iterable[dict[str, Any]],
    policy: DemandPolicy = DEFAULT_POLICY,
) -> BuyCeilings:
    """Derive a per-item-type buy ceiling from observed order/sales scale.

    This is a statistical backstop against a degenerate model output, not a
    substitute for a real production-capacity limit: nothing in sales history
    reveals how many units a factory can actually run, so that number has to
    come from the business, not from this function.
    """

    by_item_type_max: dict[str, float] = {}
    global_max = 0.0
    for item in history:
        observed = max(
            max(float(item.get("order") or 0), 0.0),
            max(float(item.get("sales") or 0), 0.0),
        )
        item_type = str(item.get("itemType") or "")
        by_item_type_max[item_type] = max(by_item_type_max.get(item_type, 0.0), observed)
        global_max = max(global_max, observed)

    def ceiling(observed_max: float) -> float:
        return max(observed_max * policy.buy_ceiling_multiplier, policy.buy_ceiling_floor)

    return BuyCeilings(
        by_item_type={item_type: ceiling(observed) for item_type, observed in by_item_type_max.items()},
        global_ceiling=ceiling(global_max),
        multiplier=policy.buy_ceiling_multiplier,
        floor=policy.buy_ceiling_floor,
    )


def weeks_on_floor(item: dict[str, Any], fallback_weeks: float) -> float:
    """Return how long a historical product was actually sellable.

    Two products with the same sales are not the same product if one had 109
    days on the floor and the other 382. ``ageingDays`` is preferred; older
    artifacts that predate it can still recover the window from the ratio of
    seasonal to weekly sell-through.
    """

    days = item.get("ageingDays")
    if days is not None and float(days) > 0:
        return max(float(days) / DAYS_PER_WEEK, 1.0)
    sell_through = float(item.get("sellThrough") or 0)
    weekly = float(item.get("weeklySellThrough") or 0)
    if sell_through > 0 and weekly > 0:
        return max(sell_through / weekly, 1.0)
    return max(fallback_weeks, 1.0)


def is_censored(item: dict[str, Any], policy: DemandPolicy = DEFAULT_POLICY) -> bool:
    """Report whether a product plausibly ran out before demand did."""

    return float(item.get("sellThrough") or 0) >= policy.censor_sell_through


def observed_weekly_rate(
    item: dict[str, Any],
    fallback_weeks: float,
) -> float:
    return cleaned_sales(item) / weeks_on_floor(item, fallback_weeks)


def lognormal_conditional_mean(mu: float, sigma: float, floor: float) -> float:
    """Return ``E[X | X > floor]`` for a lognormal rate distribution.

    A stocked-out product only tells us demand was *at least* what it sold.
    Rather than invent a stock-out date the pipeline does not record, take the
    group's fitted rate distribution and use the expectation conditional on
    exceeding the observed rate.
    """

    if floor <= 0:
        return math.exp(mu + sigma**2 / 2.0)
    if sigma <= 1e-9:
        return max(math.exp(mu), floor)
    z = (math.log(floor) - mu) / sigma
    tail = 1.0 - _normal_cdf(z)
    upper = 1.0 - _normal_cdf(z - sigma)
    if tail <= 1e-9 or upper <= 0:
        return floor
    return max(math.exp(mu + sigma**2 / 2.0) * upper / tail, floor)


@dataclass(frozen=True)
class RatePrior:
    """A fitted lognormal prior over weekly demand rate for one group."""

    mu: float
    sigma: float
    rows: int
    level: str

    @property
    def weekly_rate(self) -> float:
        return math.exp(self.mu)


def _fit_prior(log_rates: Sequence[float], level: str, floor_sigma: float) -> RatePrior:
    count = len(log_rates)
    mean = sum(log_rates) / count
    variance = (
        sum((value - mean) ** 2 for value in log_rates) / (count - 1) if count > 1 else floor_sigma**2
    )
    sigma = max(math.sqrt(max(variance, 0.0)), floor_sigma)
    return RatePrior(mu=mean, sigma=sigma, rows=count, level=level)


def _shrink_prior(raw: RatePrior, parent: RatePrior, shrink_rows: float, level: str) -> RatePrior:
    """Blend a group's own fit toward its parent, weighted by its row count.

    weight = rows / (rows + shrink_rows) grows smoothly from 0 toward 1 as
    evidence accumulates. A group that falls one row short of "enough data"
    still keeps most of its own signal instead of a hard cutoff discarding it
    outright in favour of a parent that may span far more diverse products.
    """

    weight = raw.rows / (raw.rows + shrink_rows)
    mu = weight * raw.mu + (1.0 - weight) * parent.mu
    variance = weight * raw.sigma**2 + (1.0 - weight) * parent.sigma**2
    sigma = math.sqrt(max(variance, 0.0))
    return RatePrior(mu=mu, sigma=sigma, rows=raw.rows, level=level)


def _group_keys(item: dict[str, Any]) -> list[tuple[str, ...]]:
    item_type = str(item.get("itemType") or "UNSPECIFIED")
    category = str(item.get("categoryType") or "UNSPECIFIED")
    return [(item_type, category), (item_type,), ()]


@dataclass(frozen=True)
class DemandPriors:
    """Fitted rate priors plus the empirical-Bayes shrinkage constant."""

    by_group: dict[tuple[str, ...], RatePrior]
    horizon_weeks: float
    tau: float
    policy: DemandPolicy

    def prior_for(self, item: dict[str, Any]) -> RatePrior:
        # No row-count gate here: by_group already holds hierarchically
        # shrunk priors (see fit_demand_priors), so the most specific key
        # present is always safe to use directly.
        for key in _group_keys(item):
            prior = self.by_group.get(key)
            if prior is not None:
                return prior
        return self.by_group[()]


def fit_demand_priors(
    history: Iterable[dict[str, Any]],
    policy: DemandPolicy = DEFAULT_POLICY,
) -> DemandPriors:
    """Fit per-group rate priors and the shrinkage strength from history.

    Only uncensored rows train the prior: including stocked-out products would
    drag the group mean toward the supply cap that censored them.
    """

    rows = list(history)
    horizon_candidates = [
        weeks_on_floor(item, policy.fallback_horizon_weeks)
        for item in rows
        if float(item.get("ageingDays") or 0) > 0 or float(item.get("weeklySellThrough") or 0) > 0
    ]
    horizon_weeks = (
        sorted(horizon_candidates)[len(horizon_candidates) // 2]
        if horizon_candidates
        else policy.fallback_horizon_weeks
    )

    grouped: dict[tuple[str, ...], list[float]] = {}
    for item in rows:
        if is_censored(item, policy):
            continue
        rate = observed_weekly_rate(item, horizon_weeks)
        if rate <= 0:
            continue
        log_rate = math.log(rate)
        for key in _group_keys(item):
            grouped.setdefault(key, []).append(log_rate)

    if not grouped.get(()):
        raise ValueError("Cannot fit demand priors: no uncensored positive-rate history")

    global_prior = _fit_prior(grouped[()], "global", policy.minimum_log_sigma)

    def _level_name(key: tuple[str, ...]) -> str:
        return "category" if len(key) == 2 else "itemType" if key else "global"

    raw_fits = {
        key: _fit_prior(values, _level_name(key), policy.minimum_log_sigma) for key, values in grouped.items()
    }

    # Hierarchical shrinkage: a category blends its own rows toward its
    # item-type's prior, which itself blends toward the global prior. This
    # replaces an all-or-nothing row-count cutoff with a graduated one, so a
    # category that falls one row short of "enough data" keeps most of its
    # own signal instead of inheriting the full spread of every other product
    # in the catalogue.
    by_group: dict[tuple[str, ...], RatePrior] = {(): global_prior}
    for key in sorted(key for key in raw_fits if len(key) == 1):
        by_group[key] = _shrink_prior(raw_fits[key], global_prior, policy.group_shrinkage_rows, "itemType")
    for key in sorted(key for key in raw_fits if len(key) == 2):
        parent = by_group.get((key[0],), global_prior)
        by_group[key] = _shrink_prior(raw_fits[key], parent, policy.group_shrinkage_rows, "category")

    # Empirical-Bayes shrinkage: how many analogues are worth as much as the
    # group prior. Estimated from the raw (unblended) category fits so the
    # hierarchy above doesn't feed back into its own calibration. Groups that
    # differ sharply from one another (large between variance) trust their
    # own analogues sooner.
    leaf_groups = [
        (key, raw_fits[key])
        for key in raw_fits
        if len(key) == 2 and raw_fits[key].rows >= policy.group_shrinkage_rows
    ]
    if len(leaf_groups) >= 2:
        within = [prior.sigma ** 2 for _, prior in leaf_groups]
        means = [prior.mu for _, prior in leaf_groups]
        mean_of_means = sum(means) / len(means)
        between = sum((value - mean_of_means) ** 2 for value in means) / (len(means) - 1)
        average_within = sum(within) / len(within)
        tau = clamp(average_within / between, 0.5, 20.0) if between > 1e-9 else 20.0
    else:
        tau = 4.0

    return DemandPriors(
        by_group=by_group,
        horizon_weeks=horizon_weeks,
        tau=float(tau),
        policy=policy,
    )


def corrected_weekly_rate(
    item: dict[str, Any],
    priors: DemandPriors,
) -> tuple[float, bool]:
    """Return the censoring-corrected weekly demand rate and a censored flag."""

    rate = observed_weekly_rate(item, priors.horizon_weeks)
    if rate <= 0 or not is_censored(item, priors.policy):
        return rate, False
    prior = priors.prior_for(item)
    return lognormal_conditional_mean(prior.mu, prior.sigma, rate), True


def annotate_history_rates(
    history: Iterable[dict[str, Any]],
    priors: DemandPriors,
) -> int:
    """Stamp each historical product with its corrected log demand rate.

    The rate is a property of the historical product, not of any pairing, so
    it is stored once here rather than repeated on every match. Publishing it
    lets the frontend repool the forecast when the planner moves a slider,
    instead of falling back to the legacy division.
    """

    censored_total = 0
    for item in history:
        rate, censored = corrected_weekly_rate(item, priors)
        item["weeklyLogRate"] = round(math.log(rate), 6) if rate > 0 else None
        item["demandCensored"] = censored
        censored_total += int(censored)
    return censored_total


def serialize_priors(priors: DemandPriors) -> dict[str, Any]:
    """Export the fitted priors so the frontend can reproduce the estimator."""

    return {
        "horizonWeeks": round(priors.horizon_weeks, 4),
        "shrinkageTau": round(priors.tau, 4),
        "similarityExponent": priors.policy.similarity_exponent,
        "minimumLogSigma": priors.policy.minimum_log_sigma,
        "maximumLogSigma": priors.policy.maximum_log_sigma,
        "censorSellThrough": priors.policy.censor_sell_through,
        "groupShrinkageRows": priors.policy.group_shrinkage_rows,
        "maximumAnalogues": priors.policy.maximum_analogues,
        "wideUncertaintyEffectiveN": priors.policy.wide_uncertainty_effective_n,
        "wideUncertaintySkewRatio": priors.policy.wide_uncertainty_skew_ratio,
        "pointEstimateSkew": priors.policy.point_estimate_skew,
        "groups": {
            "|".join(key): {
                "mu": round(prior.mu, 6),
                "sigma": round(prior.sigma, 6),
                "rows": prior.rows,
            }
            for key, prior in priors.by_group.items()
        },
    }


def serialize_buy_ceilings(ceilings: BuyCeilings) -> dict[str, Any]:
    """Export the fitted buy ceilings so the frontend mirrors the same cap."""

    return {
        "byItemType": {item_type: round(value, 2) for item_type, value in ceilings.by_item_type.items()},
        "globalCeiling": round(ceilings.global_ceiling, 2),
        "multiplier": ceilings.multiplier,
        "floor": ceilings.floor,
    }


@dataclass(frozen=True)
class DemandPrediction:
    """A predictive demand distribution, not a copied observation."""

    weekly_rate: float
    horizon_weeks: float
    median_demand: float
    mean_demand: float
    # The headline point forecast: the mean's skew correction scaled by
    # policy.point_estimate_skew, sitting between median_demand (no
    # correction) and mean_demand (full). See DemandPolicy for why the raw
    # mean overshoots here.
    point_demand: float
    p10: float
    p90: float
    log_mu: float
    log_sigma: float
    analogues_used: int
    effective_sample_size: float
    shrinkage_weight: float
    prior_weekly_rate: float
    censored_analogues: int


def predict_demand(
    item: dict[str, Any],
    matches: Sequence[dict[str, Any]],
    history_by_id: dict[str, dict[str, Any]],
    priors: DemandPriors,
) -> DemandPrediction | None:
    """Pool accepted analogues into a shrunk predictive demand distribution.

    Returns ``None`` when no analogue carries usable demand, which the caller
    surfaces as manual review rather than as a zero-confidence number.
    """

    policy = priors.policy
    prior = priors.prior_for(item)
    weights: list[float] = []
    log_rates: list[float] = []
    censored_count = 0

    for match in matches[: policy.maximum_analogues]:
        historical = history_by_id.get(str(match.get("historicalId")))
        if historical is None:
            continue
        score = match.get("visualScore")
        if score is None:
            continue
        rate, censored = corrected_weekly_rate(historical, priors)
        if rate <= 0:
            continue
        censored_count += int(censored)
        weights.append(max(float(score), 1e-6) ** policy.similarity_exponent)
        log_rates.append(math.log(rate))

    if not log_rates:
        return None

    total_weight = sum(weights)
    weighted_mean = sum(w * y for w, y in zip(weights, log_rates, strict=True)) / total_weight
    # Kish effective sample size: four near-identical analogues carry more
    # evidence than four where one dominates the weighting.
    effective_n = total_weight**2 / sum(w**2 for w in weights)

    if len(log_rates) > 1:
        observed_variance = (
            sum(w * (y - weighted_mean) ** 2 for w, y in zip(weights, log_rates, strict=True)) / total_weight
        )
    else:
        observed_variance = prior.sigma**2

    shrinkage_weight = effective_n / (effective_n + priors.tau)
    log_mu = shrinkage_weight * weighted_mean + (1.0 - shrinkage_weight) * prior.mu
    core_variance = (effective_n * observed_variance + priors.tau * prior.sigma**2) / (effective_n + priors.tau)
    # A single analogue is not certainty. Carrying the estimation variance of
    # the pooled mean is what stops one row from implying a point forecast.
    log_sigma = clamp(
        math.sqrt(core_variance * (1.0 + 1.0 / (effective_n + priors.tau))),
        policy.minimum_log_sigma,
        policy.maximum_log_sigma,
    )

    horizon = priors.horizon_weeks
    weekly_rate = math.exp(log_mu)
    demand_log_mu = log_mu + math.log(horizon)
    return DemandPrediction(
        weekly_rate=weekly_rate,
        horizon_weeks=horizon,
        median_demand=math.exp(demand_log_mu),
        mean_demand=math.exp(demand_log_mu + log_sigma**2 / 2.0),
        point_demand=math.exp(demand_log_mu + policy.point_estimate_skew * log_sigma**2 / 2.0),
        p10=math.exp(demand_log_mu - P10_Z * log_sigma),
        p90=math.exp(demand_log_mu + P10_Z * log_sigma),
        log_mu=demand_log_mu,
        log_sigma=log_sigma,
        analogues_used=len(log_rates),
        effective_sample_size=effective_n,
        shrinkage_weight=shrinkage_weight,
        prior_weekly_rate=prior.weekly_rate,
        censored_analogues=censored_count,
    )


def expected_sales_at(order_quantity: float, log_mu: float, log_sigma: float) -> float:
    """Return ``E[min(D, Q)]`` for lognormal demand ``D``.

    Ordering more never sells more than demand, so expected sales saturate.
    This is what makes the buy a decision under uncertainty instead of a
    division.
    """

    if order_quantity <= 0:
        return 0.0
    if log_sigma <= 1e-9:
        return min(math.exp(log_mu), order_quantity)
    z = (math.log(order_quantity) - log_mu) / log_sigma
    mean = math.exp(log_mu + log_sigma**2 / 2.0)
    return mean * _normal_cdf(z - log_sigma) + order_quantity * (1.0 - _normal_cdf(z))


def newsvendor_order(
    log_mu: float,
    log_sigma: float,
    target_sell_through: float,
) -> float:
    """Return the buy whose *expected* sell-through equals the planner target.

    Expected sell-through ``E[min(D, Q)] / Q`` decreases monotonically in Q, so
    a bisection is safe. When ``log_sigma`` is zero this returns exactly
    ``demand / target`` — the legacy rule is the no-uncertainty special case.
    """

    target = clamp(target_sell_through, 0.01, 0.99)
    low = 1e-6
    high = max(math.exp(log_mu + log_sigma**2 / 2.0) / target, 1.0)
    for _ in range(60):
        if expected_sales_at(high, log_mu, log_sigma) / high <= target:
            break
        high *= 2.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if expected_sales_at(middle, log_mu, log_sigma) / middle > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0
