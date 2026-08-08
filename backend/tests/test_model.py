from __future__ import annotations

from machine_learning.model import (
    blend_hybrid_visual_scores,
    blend_two_stage_visual_scores,
    calibrate_vision,
    match_confidence,
    no_suitable_product_match,
    recommend_one,
    sales_target,
)


def test_vision_calibration_is_monotonic() -> None:
    calibration = calibrate_vision([0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    assert calibration.similarity(0.15) > calibration.similarity(0.50) > calibration.similarity(0.90)


def test_visual_blends_use_only_image_signals() -> None:
    assert blend_two_stage_visual_scores(None, 1.0, 0.75) is None
    assert blend_two_stage_visual_scores(0.8, None, 0.75) == 0.8
    assert blend_hybrid_visual_scores(
        0.8,
        0.6,
        0.5,
        None,
        0.5,
        {"neural": 0.7, "colour": 0.2, "texture": 0.1},
    ) == 0.6556


def test_sales_target_caps_impossible_sales() -> None:
    item = {"order": 400, "dispatch": 390, "sales": 900, "sellThrough": 2.3}
    assert sales_target(item) == 400


def test_confidence_uses_one_visual_product() -> None:
    assert match_confidence([0.90], has_visual=True, issue_count=0) == "High"
    assert match_confidence([0.50], has_visual=True, issue_count=0) == "Medium"
    assert match_confidence([0.4954], has_visual=True, issue_count=0) == "Medium"
    assert match_confidence([0.49], has_visual=True, issue_count=0) == "Low"
    assert no_suitable_product_match([{"visualScore": 0.49}], "Medium")
    assert not no_suitable_product_match([{"visualScore": 0.70}], "Medium")


def test_single_visual_analogue_supplies_sales_and_sell_through_buy() -> None:
    history = [
        {"id": "AW25-OTTR-1", "order": 625, "dispatch": 600, "sales": 525, "sellThrough": 0.875},
        {"id": "AW25-OTTR-2", "order": 900, "dispatch": 850, "sales": 800, "sellThrough": 0.94},
    ]
    matches = [
        {"historicalId": "AW25-OTTR-1", "visualScore": 0.70, "hybridScore": 0.70},
        {"historicalId": "AW25-OTTR-2", "visualScore": 0.69, "hybridScore": 0.69},
    ]
    result = recommend_one({}, history, matches)
    assert result["expectedSales"] == 525
    assert result["quantity"] == 750
    assert result["analogueSales"] == 525
    assert result["analogueQuantity"] == 625
    assert result["targetSellThrough"] == 0.70
    assert result["evidencePolicy"] == "single_top_visual_analogue"
    assert "regressionSales" not in result


def test_no_visual_match_returns_manual_review_zeroes() -> None:
    result = recommend_one({}, [], [])
    assert result["noSuitableMatch"] is True
    assert result["expectedSales"] == 0
    assert result["quantity"] == 0
