"""Leave-one-out backtest for the order-quantity recommendation.

The point of this harness is to stop "the number changed" from being mistaken
for "the number improved". Every method below sees exactly the same retrieved
analogues, so differences in the reported metrics come from the estimator and
nothing else.

Retrieval is held fixed deliberately. Visual similarity between two historical
products is not persisted in the model artifact, so the default neighbour
function ranks by shared catalogue attributes. Swap ``neighbour_fn`` for real
historical-to-historical visual scores when they are available; the comparison
between methods stays valid either way because all methods share it.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_pipeline.feature_engineering import historical_identifier
from machine_learning.demand import (
    DemandPolicy,
    cleaned_sales,
    fit_demand_priors,
    is_censored,
    newsvendor_order,
    predict_demand,
    weeks_on_floor,
)

PACK_SIZE = 25
MAX_BUY = 2_000
TARGET_SELL_THROUGH = 0.70


def round_pack(value: float, pack: int = PACK_SIZE) -> int:
    return int(round(value / pack) * pack)


def load_history(path: Path) -> list[dict[str, Any]]:
    """Read the cleaned historical CSV into artifact-shaped product records.

    Ids and deduplication mirror ``feature_engineering.build_historical_features``
    exactly (``SEASON-PRODUCT_ID``, first occurrence wins), because that is the
    id space every other artifact-derived source — visual retrieval pairs
    included — is keyed by. A raw ``product_id`` id here would silently never
    match real visual candidates.
    """

    def number(row: dict[str, str], key: str) -> float:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return 0.0

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identifier = historical_identifier(row)
            if identifier in seen_ids:
                continue
            seen_ids.add(identifier)
            records.append(
                {
                    "id": identifier,
                    "itemType": row["item_type"],
                    "categoryType": row["category_type"],
                    "design": row["design"],
                    "colour": row["colour"],
                    "fabric": row["fabric"],
                    "style": row["style_code"],
                    "order": number(row, "total_order_quantity"),
                    "dispatch": number(row, "dispatch_quantity"),
                    "sales": number(row, "sales_quantity"),
                    "sellThrough": number(row, "sell_through"),
                    "ageingDays": number(row, "ageing_days"),
                    "weeklySellThrough": number(row, "weekly_sell_through"),
                }
            )
    return records


def attribute_neighbours(
    target: dict[str, Any],
    pool: Sequence[dict[str, Any]],
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Rank same-item-type products by shared catalogue attributes.

    This stands in for visual retrieval so the estimator can be measured. The
    returned shape matches the artifact's ``matches`` entries.
    """

    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in pool:
        if candidate["id"] == target["id"]:
            continue
        if candidate["itemType"] != target["itemType"]:
            continue
        shared = sum(
            candidate.get(key) == target.get(key)
            for key in ("categoryType", "design", "fabric", "colour")
        )
        # Map 0-4 shared attributes onto the same 0.5-0.9 band the visual
        # scores occupy, so similarity weighting behaves comparably.
        score = 0.50 + 0.10 * shared
        scored.append((score, candidate))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [
        {"historicalId": candidate["id"], "visualScore": score, "hybridScore": score}
        for score, candidate in scored[:limit]
    ]


@dataclass
class MethodResult:
    """Accumulated scores for one recommendation method."""

    name: str
    absolute_errors: list[float] = field(default_factory=list)
    actuals: list[float] = field(default_factory=list)
    signed_errors: list[float] = field(default_factory=list)
    realized_sell_through: list[float] = field(default_factory=list)
    covered: list[bool] = field(default_factory=list)

    def record(
        self,
        predicted: float,
        actual: float,
        order_quantity: float,
        interval: tuple[float, float] | None,
    ) -> None:
        self.absolute_errors.append(abs(predicted - actual))
        self.signed_errors.append(predicted - actual)
        self.actuals.append(actual)
        if order_quantity > 0:
            self.realized_sell_through.append(min(actual, order_quantity) / order_quantity)
        if interval is not None:
            self.covered.append(interval[0] <= actual <= interval[1])

    def summary(self) -> dict[str, Any]:
        total_actual = sum(self.actuals) or 1.0
        count = len(self.actuals) or 1
        sell_through = self.realized_sell_through or [0.0]
        return {
            "method": self.name,
            "products": len(self.actuals),
            "wmape": sum(self.absolute_errors) / total_actual,
            "mae": sum(self.absolute_errors) / count,
            "bias": sum(self.signed_errors) / count,
            "meanSellThrough": sum(sell_through) / len(sell_through),
            "sellThroughGap": abs(sum(sell_through) / len(sell_through) - TARGET_SELL_THROUGH),
            "coverage": (sum(self.covered) / len(self.covered)) if self.covered else None,
        }


def _group_mean_sales(history: Sequence[dict[str, Any]], target: dict[str, Any]) -> float:
    same_group = [
        cleaned_sales(item)
        for item in history
        if item["itemType"] == target["itemType"] and item["categoryType"] == target["categoryType"]
    ]
    if not same_group:
        same_group = [cleaned_sales(item) for item in history if item["itemType"] == target["itemType"]]
    if not same_group:
        same_group = [cleaned_sales(item) for item in history]
    return sum(same_group) / len(same_group)


NeighbourFn = Callable[[dict[str, Any], Sequence[dict[str, Any]], int], list[dict[str, Any]]]


def run_backtest(
    history: Sequence[dict[str, Any]],
    *,
    policy: DemandPolicy | None = None,
    neighbour_fn: NeighbourFn = attribute_neighbours,
    target_sell_through: float = TARGET_SELL_THROUGH,
    uncensored_only: bool = True,
) -> dict[str, Any]:
    """Hold out each historical product and score every method on it.

    Censored products are excluded from scoring by default: their observed
    sales are a supply cap, so a model that correctly predicts the demand they
    could not serve would be penalised for being right.
    """

    policy = policy or DemandPolicy()
    records = list(history)
    legacy = MethodResult("legacy_copy_analogue")
    group = MethodResult("group_mean")
    pooled = MethodResult("pooled_predictive")
    skipped = 0

    for index, target in enumerate(records):
        pool = records[:index] + records[index + 1 :]
        actual = cleaned_sales(target)
        if actual <= 0:
            skipped += 1
            continue
        if uncensored_only and is_censored(target, policy):
            skipped += 1
            continue

        matches = neighbour_fn(target, pool, policy.maximum_analogues)
        if not matches:
            skipped += 1
            continue

        # Priors are refit without the held-out product so no product informs
        # its own prediction, directly or through the group mean.
        priors = fit_demand_priors(pool, policy)
        pool_by_id = {item["id"]: item for item in pool}

        top = pool_by_id[matches[0]["historicalId"]]
        legacy_prediction = min(round_pack(cleaned_sales(top)), MAX_BUY)
        legacy_order = min(round_pack(legacy_prediction / target_sell_through), MAX_BUY)
        legacy.record(legacy_prediction, actual, legacy_order, None)

        group_prediction = min(round_pack(_group_mean_sales(pool, target)), MAX_BUY)
        group_order = min(round_pack(group_prediction / target_sell_through), MAX_BUY)
        group.record(group_prediction, actual, group_order, None)

        prediction = predict_demand(target, matches, pool_by_id, priors)
        if prediction is None:
            skipped += 1
            continue
        # Score the rate estimate over the window the held-out product actually
        # had, so exposure differences are measured rather than assumed away.
        horizon = weeks_on_floor(target, priors.horizon_weeks)
        scale = horizon / prediction.horizon_weeks
        # The mean, not the median, is the headline forecast: the median of a
        # lognormal sits ~20% below its mean, which would under-buy the whole
        # portfolio systematically even while scoring well on per-item error.
        pooled_prediction = min(round_pack(prediction.mean_demand * scale), MAX_BUY)
        pooled_order = min(
            round_pack(
                newsvendor_order(
                    prediction.log_mu + math.log(scale),
                    prediction.log_sigma,
                    target_sell_through,
                )
            ),
            MAX_BUY,
        )
        pooled.record(
            pooled_prediction,
            actual,
            pooled_order,
            (prediction.p10 * scale, prediction.p90 * scale),
        )

    return {
        "scoredProducts": len(pooled.actuals),
        "skipped": skipped,
        "targetSellThrough": target_sell_through,
        "methods": [legacy.summary(), group.summary(), pooled.summary()],
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Leave-one-out backtest  ({report['scoredProducts']} products scored, {report['skipped']} skipped)",
        f"Target sell-through: {report['targetSellThrough']:.0%}",
        "",
        f"{'method':<24}{'WMAPE':>9}{'MAE':>9}{'bias':>10}{'sell-thr':>10}{'p10-p90':>10}",
        "-" * 72,
    ]
    for entry in report["methods"]:
        coverage = f"{entry['coverage']:.0%}" if entry["coverage"] is not None else "n/a"
        lines.append(
            f"{entry['method']:<24}"
            f"{entry['wmape']:>8.1%} "
            f"{entry['mae']:>8.0f} "
            f"{entry['bias']:>+9.0f} "
            f"{entry['meanSellThrough']:>9.1%} "
            f"{coverage:>9}"
        )
    return "\n".join(lines)


def main() -> None:
    from core.config import paths

    history = load_history(paths.repository / "data" / "processed" / "historical_cleaned.csv")
    report = run_backtest(history)
    print(format_report(report))


if __name__ == "__main__":
    main()
