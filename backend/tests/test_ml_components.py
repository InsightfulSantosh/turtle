from __future__ import annotations

import math

from domain.contracts import (
    BusinessConstraints,
    Candidate,
    DemandForecast,
)
from machine_learning.hierarchy import (
    HierarchySpec,
    MinTraceReconciler,
)
from machine_learning.optimization import BuyOptimizer
from machine_learning.ranking import CandidateReranker


def item(item_id: str, item_type: str = "OTSH") -> dict:
    return {
        "id": item_id,
        "itemType": item_type,
        "design": "CHECKS",
        "categoryType": "FORMAL",
        "fabric": "100% Cotton",
        "colour": "BLUE",
    }


def test_ranker_combines_available_product_evidence() -> None:
    product = item("NEW")
    candidates = [
        Candidate(item("GOOD"), 0.82, 500, seasons_observed=3, feedback_score=1.0),
        Candidate(
            item("WRONG", item_type="OTTS"),
            0.90,
            700,
            seasons_observed=3,
            feedback_score=1.0,
        ),
    ]
    ranked = CandidateReranker().rank(product, candidates)
    assert ranked[0].candidate.item["id"] == "GOOD"
    assert ranked[0].attribute_score > ranked[1].attribute_score


def test_optimizer_respects_budget_capacity_and_pack() -> None:
    optimized = BuyOptimizer().optimize(
        DemandForecast(200, 400, 600, "test"),
        BusinessConstraints(
            pack_size=25,
            minimum_order=50,
            maximum_order=1_000,
            unit_cost=10,
            budget=1_000,
            supplier_capacity=130,
        ),
    )
    assert optimized.quantity == 100
    assert optimized.high == 100
    assert {"budget", "supplier_capacity"} <= set(optimized.binding_constraints)


def test_mintrace_output_is_coherent() -> None:
    spec = HierarchySpec(
        node_names=("total", "shirts", "tees"),
        bottom_names=("shirts", "tees"),
        summing_matrix=((1, 1), (1, 0), (0, 1)),
    )
    result = MinTraceReconciler(spec).reconcile(
        {"total": 140, "shirts": 80, "tees": 70}
    )
    assert math.isclose(
        result["total"],
        result["shirts"] + result["tees"],
        rel_tol=1e-9,
    )
