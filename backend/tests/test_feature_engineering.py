from __future__ import annotations

from data_pipeline.feature_engineering import (
    engineer_features,
    exclude_upcoming_without_historical_item,
    exclude_zero_sales_rows,
)


def historical_row() -> dict:
    return {
        "product_id": "OTSH-23850-1001",
        "season": "SS26",
        "item_type": "OTSH",
        "style_code": "23850",
        "colour_code": "1001",
        "design": "DOBBY/STRUCTURE",
        "category_type": "CEREMONIAL",
        "fabric": "Cotton – 100%",
        "merchandise_type": "FASHION",
        "colour": "PINK",
        "total_order_quantity": 414,
        "dispatch_quantity": 555,
        "sales_quantity": 280,
        "sell_through": 0.5045,
    }


def upcoming_row() -> dict:
    return {
        "product_id": "OTTR-300171-1001",
        "season": "SS27",
        "item_type": "OTTR",
        "style_code": "300171",
        "colour_code": "1001",
        "design": "CHECKS",
        "colour": "GREY",
        "fabric": "Polyester – Viscose Stretch",
        "category_type": "FORMAL",
        "collection_world": "PROFESSIONAL",
    }


def test_feature_engineering_uses_clean_identifier_and_fabric_values() -> None:
    result = engineer_features(
        [historical_row()],
        [upcoming_row()],
    )
    historical = result.historical[0]
    upcoming = result.upcoming[0]

    assert historical["sourceId"] == "OTSH-23850-1001"
    assert historical["fabric"] == "Cotton – 100%"
    assert "mrp" not in historical
    assert upcoming["id"] == "OTTR-300171-1001"
    assert upcoming["fabric"] == "Polyester – Viscose Stretch"
    assert "mrp" not in upcoming
    assert upcoming["modelFlags"] == ["unseen_item"]


def test_zero_sales_rows_are_excluded_before_training() -> None:
    positive = historical_row()
    zero = dict(
        historical_row(),
        product_id="OTSH-23850-1002",
        sales_quantity=0,
    )

    retained, excluded = exclude_zero_sales_rows([zero, positive])

    assert retained == [positive]
    assert excluded == 1


def test_upcoming_rows_without_historical_item_are_excluded() -> None:
    represented = upcoming_row()
    represented["item_type"] = "OTSH"
    unseen = dict(upcoming_row(), item_type="OTJT")

    retained, excluded = exclude_upcoming_without_historical_item(
        [historical_row()],
        [unseen, represented],
    )

    assert retained == [represented]
    assert excluded == 1
