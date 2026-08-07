"""Transform validated workbook rows into model-ready product records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_pipeline.preprocessing import (
    normalize_text,
    parse_integer,
    parse_number,
)


@dataclass(frozen=True)
class FeatureEngineeringResult:
    historical: list[dict[str, Any]]
    upcoming: list[dict[str, Any]]
    duplicate_historical_rows_removed: int
    upcoming_without_historical_item: int


def exclude_zero_sales_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Exclude historical rows with no observed unit sales from model training."""

    retained = [
        row
        for row in rows
        if parse_number(row.get("sales_quantity")) != 0
    ]
    excluded = len(rows) - len(retained)
    if not retained:
        raise ValueError("Historical training data contains no positive-sales rows")
    return retained, excluded


def exclude_upcoming_without_historical_item(
    historical_rows: list[dict[str, Any]],
    upcoming_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Exclude upcoming rows whose item type has no retained historical data."""

    historical_items = {
        normalize_text(row.get("item_type"))
        for row in historical_rows
    }
    retained = [
        row
        for row in upcoming_rows
        if normalize_text(row.get("item_type")) in historical_items
    ]
    excluded = len(upcoming_rows) - len(retained)
    if not retained:
        raise ValueError(
            "Upcoming data contains no item types represented in historical training"
        )
    return retained, excluded


def historical_identifier(row: dict[str, Any]) -> str:
    return (
        f"{normalize_text(row['season'])}-"
        f"{normalize_text(row['product_id'])}"
    )


def product_attributes(row: dict[str, Any]) -> dict[str, Any]:
    """Build the shared product contract from a canonical processed row."""

    return {
        "itemType": normalize_text(row["item_type"]),
        "style": normalize_text(row["style_code"]),
        "colourCode": normalize_text(row["colour_code"]),
        "design": normalize_text(row["design"]),
        "categoryType": normalize_text(row["category_type"]),
        "fabric": row["fabric"],
        "season": normalize_text(row["season"]),
        "colour": normalize_text(row["colour"]),
        "imageUrl": None,
        "hasVisualFeature": False,
    }


def build_historical_features(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    historical: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates_removed = 0

    for row in rows:
        identifier = historical_identifier(row)
        if identifier in seen:
            duplicates_removed += 1
            continue
        seen.add(identifier)
        historical.append({
            "id": identifier,
            "sourceId": row["product_id"],
            **product_attributes(row),
            "order": parse_integer(row["total_order_quantity"]),
            "dispatch": parse_integer(row["dispatch_quantity"]),
            "sales": parse_integer(row["sales_quantity"]),
            "sellThrough": round(parse_number(row["sell_through"]), 6),
            # Exposure fields. Demand estimation needs to know how long a
            # product was sellable: identical sales over 109 and 382 days are
            # not identical demand.
            "ageingDays": parse_integer(row.get("ageing_days")),
            "weeklySellThrough": round(parse_number(row.get("weekly_sell_through")), 8),
        })
    return historical, duplicates_removed


def build_upcoming_features(
    rows: list[dict[str, Any]],
    historical_items: set[str],
) -> tuple[list[dict[str, Any]], int]:
    upcoming: list[dict[str, Any]] = []
    unseen_item = 0

    for row in rows:
        attributes = product_attributes(row)
        flags: list[str] = []
        if attributes["itemType"] not in historical_items:
            unseen_item += 1
            flags.append("unseen_item")
        upcoming.append({
            "id": row["product_id"],
            **attributes,
            "world": normalize_text(row["collection_world"]),
            "modelFlags": flags,
        })
    return upcoming, unseen_item


def engineer_features(
    historical_rows: list[dict[str, Any]],
    upcoming_rows: list[dict[str, Any]],
) -> FeatureEngineeringResult:
    historical, duplicates_removed = build_historical_features(historical_rows)
    upcoming, unseen_item = build_upcoming_features(
        upcoming_rows,
        {item["itemType"] for item in historical},
    )
    return FeatureEngineeringResult(
        historical=historical,
        upcoming=upcoming,
        duplicate_historical_rows_removed=duplicates_removed,
        upcoming_without_historical_item=unseen_item,
    )
