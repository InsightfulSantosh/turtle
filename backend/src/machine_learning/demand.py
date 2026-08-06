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


@dataclass(frozen=True)
class DemandPolicy:
    """Tunable estimator settings kept out of the calculation code."""

    censor_sell_through: float = 0.95
    minimum_group_rows: int = 8
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


DEFAULT_POLICY = DemandPolicy()


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


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
        for key in _group_keys(item):
            prior = self.by_group.get(key)
            if prior is not None and prior.rows >= self.policy.minimum_group_rows:
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
    by_group = {
        key: _fit_prior(values, "group" if len(key) == 2 else "itemType" if key else "global", policy.minimum_log_sigma)
        for key, values in grouped.items()
    }
    by_group[()] = global_prior

    # Empirical-Bayes shrinkage: how many analogues are worth as much as the
    # group prior. Groups that differ sharply from one another (large between
    # variance) trust their own analogues sooner.
    leaf_groups = [
        (key, values)
        for key, values in grouped.items()
        if len(key) == 2 and len(values) >= policy.minimum_group_rows
    ]
    if len(leaf_groups) >= 2:
        within = [by_group[key].sigma ** 2 for key, _ in leaf_groups]
        means = [by_group[key].mu for key, _ in leaf_groups]
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
        "minimumGroupRows": priors.policy.minimum_group_rows,
        "maximumAnalogues": priors.policy.maximum_analogues,
        "wideUncertaintyEffectiveN": priors.policy.wide_uncertainty_effective_n,
        "wideUncertaintySkewRatio": priors.policy.wide_uncertainty_skew_ratio,
        "groups": {
            "|".join(key): {
                "mu": round(prior.mu, 6),
                "sigma": round(prior.sigma, 6),
                "rows": prior.rows,
            }
            for key, prior in priors.by_group.items()
        },
    }


@dataclass(frozen=True)
class DemandPrediction:
    """A predictive demand distribution, not a copied observation."""

    weekly_rate: float
    horizon_weeks: float
    median_demand: float
    mean_demand: float
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
