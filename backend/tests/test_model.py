from __future__ import annotations

import json

import numpy as np

from core.config import paths
from machine_learning.model import (
    attribute_similarity,
    calibrate_vision,
    demand_features,
    demand_uncertainty,
    fit_demand_pipeline,
    match_confidence,
    no_suitable_product_match,
    normalized_demand,
    recommend_one,
    sales_target,
)


def test_vision_calibration_is_monotonic() -> None:
    calibration = calibrate_vision([0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    assert calibration.similarity(0.15) > calibration.similarity(0.50) > calibration.similarity(0.90)


def test_category_mismatch_is_strongly_penalized() -> None:
    base = {
        "itemType": "OTSH",
        "design": "CHECKS",
        "categoryType": "FORMAL",
        "fabric": "100% Cotton",
        "colour": "BLUE",
    }
    same = dict(base)
    different = dict(base, itemType="OTTS")
    assert attribute_similarity(base, same)[0] == 1.0
    assert attribute_similarity(base, different)[0] < 0.5


def test_only_five_approved_attributes_are_compared() -> None:
    left = {
        "itemType": "OTSH",
        "design": "CHECKS",
        "categoryType": "FORMAL",
        "fabric": "100% Cotton",
        "colour": "BLUE",
        "season": "AW25",
    }
    _, breakdown = attribute_similarity(left, dict(left, season="SS27"))
    assert set(breakdown) == {
        "item",
        "design",
        "category_type",
        "fabric",
        "colour",
    }


def test_only_five_approved_attributes_are_sent_to_demand_pipeline() -> None:
    features = demand_features(
        {
            "itemType": "OTSH",
            "design": "CHECKS",
            "categoryType": "FORMAL",
            "fabric": "COTTON",
            "colour": "BLUE",
            "season": "AW25",
        }
    )
    assert set(features) == {
        "item_type=OTSH",
        "design=CHECKS",
        "category_type=FORMAL",
        "colour=BLUE",
        "fabric=COTTON",
    }


def test_sales_target_contains_bad_sales_row() -> None:
    item = {"order": 400, "dispatch": 390, "sales": 900, "sellThrough": 2.3}
    assert sales_target(item) == 400
    assert normalized_demand(item) == 400 / 0.70


def test_demand_model_uses_sklearn_pipeline() -> None:
    rows = [
        {
            "itemType": "OTSH",
            "design": "CHECKS",
            "categoryType": "FORMAL",
            "fabric": "COTTON",
            "colour": "BLUE",
        },
        {
            "itemType": "OTSH",
            "design": "PLAINS",
            "categoryType": "CASUAL",
            "fabric": "LINEN",
            "colour": "WHITE",
        },
        {
            "itemType": "OTTS",
            "design": "PRINTS",
            "categoryType": "CASUAL",
            "fabric": "COTTON",
            "colour": "BLACK",
        },
    ]
    pipeline = fit_demand_pipeline(rows, np.asarray([500.0, 350.0, 200.0]), alpha=1.0)
    assert list(pipeline.named_steps) == ["features", "scale", "ridge"]
    assert np.isfinite(pipeline.predict([rows[0]])[0])


def test_match_confidence_is_separate_from_demand_uncertainty() -> None:
    assert match_confidence([0.90, 0.80, 0.75], has_visual=True, issue_count=0) == "High"
    assert demand_uncertainty(quantity=650, interval_half_width=325) == "Wide"


def test_weak_visual_candidates_are_rejected() -> None:
    assert no_suitable_product_match(
        [{"visualScore": 0.49}],
        "Medium",
    )
    assert no_suitable_product_match(
        [{"visualScore": 0.90}],
        "Low",
    )
    assert not no_suitable_product_match(
        [{"visualScore": 0.70}],
        "Medium",
    )


def test_generated_artifact_contract() -> None:
    data = json.loads(paths.model_artifact.read_text(encoding="utf-8"))
    assert data["meta"]["model"]["version"] == "4.2.0"
    assert data["meta"]["dataMode"] == "real"
    assert data["meta"]["upcomingSeason"] == "SS27"
    assert data["meta"]["model"]["demandLibrary"] == "scikit-learn"
    assert "image embeddings" in data["meta"]["visualMethod"]
    assert data["meta"]["visionModel"]["modelId"] == "Marqo/marqo-fashionSigLIP"
    assert data["meta"]["visionModel"]["embeddingDimension"] == 768
    assert data["meta"]["visionModel"]["historicalCoverage"] == data["meta"]["historicalImageCoverage"] == 508
    assert data["meta"]["visionModel"]["upcomingCoverage"] == data["meta"]["upcomingImageCoverage"] == 36
    assert data["meta"]["visionCalibration"]["servingMedianDistance"] > (
        data["meta"]["visionCalibration"]["historicalMedianDistance"]
    )
    assert data["meta"]["model"]["trainingRows"] == len(data["historical"])
    assert data["meta"]["model"]["modelSelection"] == "Temporal holdout + ParameterGrid"
    assert data["meta"]["model"]["minimumVisualScore"] == 0.50
    assert data["meta"]["model"]["minimumMatchConfidence"] == "Medium"
    assert data["meta"]["model"]["validationRows"] > 0
    assert len(data["historical"]) == 665
    assert len(data["upcoming"]) == 1_752
    assert data["meta"]["dataQuality"]["zeroSalesHistoricalRowsExcluded"] == 142
    assert data["meta"]["dataQuality"]["upcomingRowsExcludedUnseenItem"] == 114
    assert all(item["salesTarget"] > 0 for item in data["historical"])
    assert all(item["itemType"] != "OTJT" for item in data["upcoming"])
    assert data["meta"]["attributeAudit"]["activeCount"] == 5
    assert set(data["meta"]["model"]["attributeWeights"]) == {
        "item",
        "design",
        "category_type",
        "fabric",
        "colour",
    }
    assert data["meta"]["attributeAudit"]["excludedConstants"] == []
    assert 0 <= data["meta"]["model"]["backtest"]["wape"] <= 1
    assert data["meta"]["model"]["forecastTarget"] == "Cleaned positive historical unit sales"
    assert len(data["upcoming"]) == data["meta"]["upcomingItems"]
    for item in data["upcoming"]:
        assert "mrp" not in item
        recommendation = item["recommendation"]
        assert recommendation["confidence"] == recommendation["matchConfidence"]
        assert isinstance(recommendation["noSuitableMatch"], bool)
        if recommendation["noSuitableMatch"]:
            assert recommendation["expectedSales"] == recommendation["regressionSales"]
        assert recommendation["demandUncertainty"] in {"Narrow", "Moderate", "Wide"}
        assert recommendation["low"] <= recommendation["quantity"] <= recommendation["high"]
        assert recommendation["salesLow"] <= recommendation["expectedSales"] <= recommendation["salesHigh"]
        assert recommendation["quantity"] % 25 == 0
        assert recommendation["expectedSales"] % 25 == 0
        assert len(item["matches"]) == 8
        assert all(
            set(match["attributeBreakdown"]) == {"item", "design", "category_type", "fabric", "colour"}
            for match in item["matches"]
        )
        assert all(0 <= match["attributeScore"] <= 1 for match in item["matches"])
        if item["imageUrl"]:
            assert item["hasVisualFeature"] is True
            assert all(match["visualScore"] is not None for match in item["matches"])
        else:
            assert item["hasVisualFeature"] is False
            assert all(match["visualScore"] is None for match in item["matches"])
        assert "design" in item
        assert "categoryType" in item
        assert "pattern" not in item
        assert "fit" not in item
        assert "lifecycle" not in item
    history_by_id = {item["id"]: item for item in data["historical"]}
    image_backed_upcoming = [item for item in data["upcoming"] if item["imageUrl"]]
    assert image_backed_upcoming
    assert all(
        not item["recommendation"]["noSuitableMatch"]
        for item in image_backed_upcoming
    )
    assert all(
        item["matches"][0]["visualScore"] >= data["meta"]["model"]["minimumVisualScore"]
        for item in image_backed_upcoming
    )
    assert all("mrp" not in item for item in data["historical"])
    assert all(
        history_by_id[match["historicalId"]]["imageUrl"] for item in data["upcoming"] for match in item["matches"]
    )

    model = data["meta"]["model"]
    history = data["historical"]
    targets = np.asarray([float(item["salesTarget"]) for item in history])
    pipeline = fit_demand_pipeline(history, targets, float(model["ridgeAlpha"]))
    representative = data["upcoming"][0]
    reproduced = recommend_one(
        representative,
        history,
        representative["matches"],
        targets,
        pipeline,
        model,
    )
    assert reproduced["quantity"] == representative["recommendation"]["quantity"]
    assert reproduced["expectedSales"] == representative["recommendation"]["expectedSales"]
    assert reproduced["matchConfidence"] == representative["recommendation"]["matchConfidence"]
    assert reproduced["noSuitableMatch"] == representative["recommendation"]["noSuitableMatch"]
    assert reproduced["demandUncertainty"] == representative["recommendation"]["demandUncertainty"]

    stricter_inventory_policy = dict(model, targetSellThrough=0.80)
    policy_scenario = recommend_one(
        representative,
        history,
        representative["matches"],
        targets,
        pipeline,
        stricter_inventory_policy,
    )
    assert policy_scenario["expectedSales"] == reproduced["expectedSales"]
    assert policy_scenario["salesLow"] == reproduced["salesLow"]
    assert policy_scenario["salesHigh"] == reproduced["salesHigh"]
    assert policy_scenario["quantity"] != reproduced["quantity"]
