from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from season_intelligence.model import (
    attribute_similarity,
    calibrate_vision,
    demand_features,
    demand_uncertainty,
    fit_demand_pipeline,
    match_confidence,
    normalized_demand,
    recommend_one,
)


APP_ROOT = Path(__file__).resolve().parents[2]


def test_vision_calibration_is_monotonic() -> None:
    calibration = calibrate_vision([0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    assert calibration.similarity(0.15) > calibration.similarity(0.50) > calibration.similarity(0.90)


def test_category_mismatch_is_strongly_penalized() -> None:
    base = {
        "itemType": "OTSH",
        "sleeve": "F",
        "provision": "SF",
        "pattern": "CHECKS",
        "range": "CMI + VMI",
        "lifecycle": "AW2026",
        "fit": "TAILORED",
        "fabric": "100% Cotton",
        "fashion": "FASHION",
        "colour": "BLUE",
        "mrp": 1_899,
    }
    same = dict(base)
    different = dict(base, itemType="OTTS")
    assert attribute_similarity(base, same)[0] == 1.0
    assert attribute_similarity(base, different)[0] < 0.5


def test_constant_fields_are_removed_and_season_family_is_retained() -> None:
    left = {
        "itemType": "OTSH", "sleeve": "F", "provision": "SF", "pattern": "CHECKS",
        "range": "CMI + VMI", "lifecycle": "AW2026", "fit": "TAILORED",
        "fabric": "100% Cotton", "fashion": "FASHION", "colour": "BLUE", "mrp": 1_899,
    }
    same_season = dict(left, range="ANOTHER INTERNAL CODE", lifecycle="AW2025")
    different_season = dict(left, lifecycle="SS2026")
    _, same_breakdown = attribute_similarity(left, same_season)
    _, different_breakdown = attribute_similarity(left, different_season)
    assert "range" not in same_breakdown
    assert "fashion" not in same_breakdown
    assert same_breakdown["lifecycle"] == 1.0
    assert different_breakdown["lifecycle"] == 0.0


def test_constant_fields_are_not_sent_to_the_demand_pipeline() -> None:
    features = demand_features({
        "itemType": "OTSH", "range": "CMI + VMI", "fashion": "FASHION",
        "lifecycle": "AW2026", "pattern": "CHECKS", "mrp": 1_899,
    })
    assert not any(name.startswith("range=") for name in features)
    assert not any(name.startswith("fashion=") for name in features)


def test_normalized_demand_contains_bad_sales_row() -> None:
    item = {"order": 400, "dispatch": 390, "sales": 900, "sellThrough": 2.3}
    assert normalized_demand(item) <= 600


def test_demand_model_uses_sklearn_pipeline() -> None:
    rows = [
        {"itemType": "OTSH", "pattern": "CHECKS", "fabric": "COTTON", "mrp": 1_499},
        {"itemType": "OTSH", "pattern": "PLAINS", "fabric": "LINEN", "mrp": 1_999},
        {"itemType": "OTTS", "pattern": "PRINTS", "fabric": "COTTON", "mrp": 999},
    ]
    pipeline = fit_demand_pipeline(rows, np.asarray([500.0, 350.0, 200.0]), alpha=1.0)
    assert list(pipeline.named_steps) == ["features", "scale", "ridge"]
    assert np.isfinite(pipeline.predict([rows[0]])[0])


def test_match_confidence_is_separate_from_demand_uncertainty() -> None:
    assert match_confidence([0.90, 0.80, 0.75], has_visual=True, issue_count=0) == "High"
    assert demand_uncertainty(quantity=650, interval_half_width=325) == "Wide"


def test_generated_artifact_contract() -> None:
    data = json.loads((APP_ROOT / "app" / "generated-data.json").read_text(encoding="utf-8"))
    assert data["meta"]["model"]["version"] == "2.3.2"
    assert data["meta"]["model"]["demandLibrary"] == "scikit-learn"
    assert "FashionCLIP" in data["meta"]["visualMethod"]
    assert data["meta"]["visionModel"]["modelId"] == "patrickjohncyh/fashion-clip"
    assert data["meta"]["visionModel"]["embeddingDimension"] == 512
    assert data["meta"]["visionModel"]["historicalCoverage"] == data["meta"]["historicalImageCoverage"]
    assert data["meta"]["visionModel"]["upcomingCoverage"] == data["meta"]["upcomingImageCoverage"]
    assert data["meta"]["model"]["trainingRows"] == len(data["historical"])
    assert data["meta"]["attributeAudit"]["activeCount"] == 9
    assert set(data["meta"]["model"]["attributeWeights"]) == {
        "category", "sleeve", "provision", "pattern", "lifecycle",
        "fit", "fabric", "colour", "price",
    }
    assert {row["historicalColumn"] for row in data["meta"]["attributeAudit"]["excludedConstants"]} == {"CAT2", "CAT5"}
    assert 0 <= data["meta"]["model"]["backtest"]["wape"] <= 1
    assert len(data["upcoming"]) == data["meta"]["upcomingItems"]
    for item in data["upcoming"]:
        recommendation = item["recommendation"]
        assert recommendation["confidence"] == recommendation["matchConfidence"]
        assert recommendation["demandUncertainty"] in {"Narrow", "Moderate", "Wide"}
        assert recommendation["low"] <= recommendation["quantity"] <= recommendation["high"]
        assert recommendation["quantity"] % 25 == 0
        assert all(0 <= match["attributeScore"] <= 1 for match in item["matches"])
        assert all(match["visualScore"] is None or 0 <= match["visualScore"] <= 1 for match in item["matches"])
        assert all("range" not in match["attributeBreakdown"] and "fashion" not in match["attributeBreakdown"] for match in item["matches"])

    model = data["meta"]["model"]
    history = data["historical"]
    targets = np.asarray([float(item["normalizedDemand"]) for item in history])
    pipeline = fit_demand_pipeline(history, targets, float(model["ridgeAlpha"]))
    representative = next(item for item in data["upcoming"] if item["id"] == "OTSH-98427-1001")
    reproduced = recommend_one(
        representative,
        history,
        representative["matches"],
        targets,
        pipeline,
        model,
    )
    assert reproduced["quantity"] == representative["recommendation"]["quantity"]
    assert reproduced["matchConfidence"] == representative["recommendation"]["matchConfidence"]
    assert reproduced["demandUncertainty"] == representative["recommendation"]["demandUncertainty"]
