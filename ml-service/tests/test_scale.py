from __future__ import annotations

import math
from pathlib import Path

from season_intelligence.contracts import BusinessConstraints, Candidate, DemandForecast
from season_intelligence.embeddings import SuppliedEmbeddingProvider
from season_intelligence.forecasting import DemandForecaster
from season_intelligence.hierarchy import HierarchySpec, MinTraceReconciler
from season_intelligence.optimization import BuyOptimizer
from season_intelligence.platform import ScaleEngine, serialize_recommendation
from season_intelligence.ranking import CandidateReranker
from season_intelligence.retrieval import vector_literal


def item(item_id: str, item_type: str = "OTSH", mrp: float = 1_899) -> dict:
    return {
        "id": item_id, "itemType": item_type, "sleeve": "F", "provision": "SF",
        "pattern": "CHECKS", "range": "CMI + VMI", "fit": "TAILORED",
        "fabric": "100% Cotton", "fashion": "FASHION", "colour": "BLUE", "mrp": mrp,
    }


class FakeRepository:
    def __init__(self, candidates: list[Candidate]):
        self.candidates = candidates

    def search(self, embedding, product, limit=200):
        assert embedding == [1.0, 0.0]
        return self.candidates[:limit]


def test_ranker_combines_visual_and_attribute_evidence() -> None:
    product = item("NEW")
    candidates = [
        Candidate(item("GOOD"), 0.82, 500, seasons_observed=3, feedback_score=1.0),
        Candidate(item("WRONG", item_type="OTTS"), 0.90, 700, seasons_observed=3, feedback_score=1.0),
    ]
    ranked = CandidateReranker().rank(product, candidates)
    assert ranked[0].candidate.item["id"] == "GOOD"
    assert ranked[0].attribute_score > ranked[1].attribute_score


def test_engine_returns_quantiles_and_constraint_safe_buy() -> None:
    candidates = [
        Candidate(item("A"), 0.90, 300, seasons_observed=3),
        Candidate(item("B"), 0.85, 500, seasons_observed=2),
        Candidate(item("C"), 0.80, 700, seasons_observed=1),
    ]
    engine = ScaleEngine(
        FakeRepository(candidates), SuppliedEmbeddingProvider(dimension=2),
        CandidateReranker(), DemandForecaster(),
    )
    product = dict(item("NEW"), embedding=[1.0, 0.0])
    result = engine.recommend(product, BusinessConstraints(pack_size=25, maximum_order=450), top_k=3)
    payload = serialize_recommendation(result)
    assert payload["recommendation"]["quantity"] % 25 == 0
    assert payload["recommendation"]["quantity"] <= 450
    assert payload["forecast"]["p10"] <= payload["forecast"]["p50"] <= payload["forecast"]["p90"]
    assert len(payload["matches"]) == 3


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
    result = MinTraceReconciler(spec).reconcile({"total": 140, "shirts": 80, "tees": 70})
    assert math.isclose(result["total"], result["shirts"] + result["tees"], rel_tol=1e-9)


def test_vector_literal_validates_dimension_and_finite_values() -> None:
    assert vector_literal([0.25, -0.5], 2) == "[0.25,-0.5]"
    try:
        vector_literal([float("nan"), 0.0], 2)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite embedding was accepted")


def test_scale_schema_contains_operational_guards() -> None:
    sql = (Path(__file__).resolve().parents[1] / "migrations" / "001_scale_platform.sql").read_text(encoding="utf-8")
    assert "halfvec(512)" in sql
    assert "USING hnsw" in sql
    assert "similarity_feedback" in sql
    assert "batch_jobs" in sql
