from __future__ import annotations

import json
from pathlib import Path

from season_intelligence.model import (
    attribute_similarity,
    calibrate_vision,
    normalized_demand,
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


def test_normalized_demand_contains_bad_sales_row() -> None:
    item = {"order": 400, "dispatch": 390, "sales": 900, "sellThrough": 2.3}
    assert normalized_demand(item) <= 600


def test_generated_artifact_contract() -> None:
    data = json.loads((APP_ROOT / "app" / "generated-data.json").read_text(encoding="utf-8"))
    assert data["meta"]["model"]["version"] == "2.1.0"
    assert "FashionCLIP" in data["meta"]["visualMethod"]
    assert data["meta"]["visionModel"]["modelId"] == "patrickjohncyh/fashion-clip"
    assert data["meta"]["visionModel"]["embeddingDimension"] == 512
    assert data["meta"]["visionModel"]["historicalCoverage"] == data["meta"]["historicalImageCoverage"]
    assert data["meta"]["visionModel"]["upcomingCoverage"] == data["meta"]["upcomingImageCoverage"]
    assert data["meta"]["model"]["trainingRows"] == len(data["historical"])
    assert 0 <= data["meta"]["model"]["backtest"]["wape"] <= 1
    assert len(data["upcoming"]) == data["meta"]["upcomingItems"]
    for item in data["upcoming"]:
        recommendation = item["recommendation"]
        assert recommendation["low"] <= recommendation["quantity"] <= recommendation["high"]
        assert recommendation["quantity"] % 25 == 0
        assert all(0 <= match["attributeScore"] <= 1 for match in item["matches"])
        assert all(match["visualScore"] is None or 0 <= match["visualScore"] <= 1 for match in item["matches"])
