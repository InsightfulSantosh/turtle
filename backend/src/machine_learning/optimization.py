from __future__ import annotations

import math

from domain.contracts import (
    BusinessConstraints,
    DemandForecast,
    OptimizedBuy,
)


def _nearest_pack(value: float, pack: int) -> int:
    return int(round(value / pack) * pack)


def _floor_pack(value: float, pack: int) -> int:
    return int(math.floor(value / pack) * pack)


class BuyOptimizer:
    def optimize(self, forecast: DemandForecast, constraints: BusinessConstraints) -> OptimizedBuy:
        pack = max(int(constraints.pack_size), 1)
        lower_bound = max(int(constraints.minimum_order), 0)
        upper_bound = max(int(constraints.maximum_order), 0)
        binding: list[str] = []
        warnings: list[str] = []
        if constraints.supplier_capacity is not None and constraints.supplier_capacity < upper_bound:
            upper_bound = max(0, int(constraints.supplier_capacity))
            binding.append("supplier_capacity")
        if constraints.budget is not None and constraints.unit_cost:
            affordable = int(constraints.budget / constraints.unit_cost)
            if affordable < upper_bound:
                upper_bound = max(0, affordable)
                binding.append("budget")
        upper_bound = _floor_pack(upper_bound, pack)
        if upper_bound < lower_bound:
            warnings.append("minimum_order_infeasible")
            lower_bound = upper_bound
        quantity = min(max(_nearest_pack(forecast.p50, pack), lower_bound), upper_bound)
        low = min(max(_nearest_pack(forecast.p10, pack), lower_bound), upper_bound)
        high = min(max(_nearest_pack(forecast.p90, pack), quantity), upper_bound)
        if quantity == lower_bound and forecast.p50 < lower_bound:
            binding.append("minimum_order")
        if quantity == upper_bound and forecast.p50 > upper_bound:
            binding.append("maximum_order")
        return OptimizedBuy(
            quantity,
            min(low, quantity),
            max(high, quantity),
            tuple(dict.fromkeys(binding)),
            tuple(warnings),
        )
