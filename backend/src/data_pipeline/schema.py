"""Canonical processed-data schemas shared by every downstream component."""

from __future__ import annotations

from typing import Any


class SchemaStandardizationError(ValueError):
    """Raised when a source row cannot be mapped safely to the canonical schema."""


CATEGORY_TYPE_MAP = {
    "TAILORED": "FORMAL",
    "RELAXED WASH": "CASUAL",
    "DENIM BRAND": "DENIM",
    "CEREMONIAL": "CEREMONIAL",
}

HISTORICAL_COLUMN_MAP = {
    "CON": "product_id",
    "SEASON": "season",
    "ITEM TYPE": "item_type",
    "SORT": "style_code",
    "COLOR": "colour_code",
    "CAT1": "design",
    "CAT3": "category_type",
    "CAT4": "fabric",
    "CAT5": "merchandise_type",
    "COLOR__2": "colour",
    "MAX": "max_quantity",
    "PT": "pt_quantity",
    "RIL": "ril_quantity",
    "TOTAL ORDER": "total_order_quantity",
    "DISPATCH": "dispatch_quantity",
    "SALE": "sales_quantity",
    "1ST DISPATCH": "first_dispatch_date",
    "AGEING": "ageing_days",
    "SELL THR": "sell_through",
    "WKLY SELL THRU": "weekly_sell_through",
}
HISTORICAL_IGNORED_SOURCE_COLUMNS = frozenset({
    "MRP",
    "SLEEVS",
    "PROV",
})

UPCOMING_COLUMN_MAP = {
    "CC (SEG-1+2+3)": "product_id",
    "SEGMENT1": "item_type",
    "SEGMENT2": "style_code",
    "SEGMENT3": "colour_code",
    "CAT1": "design",
    "COLOUR NAME": "colour",
    "CAT4": "fabric",
    "CAT-3": "category_type",
    "WORLD": "collection_world",
}
UPCOMING_IGNORED_SOURCE_COLUMNS = frozenset({
    "PROPOSED MRP",
    "SL",
})

HISTORICAL_COLUMN_ORDER = tuple(HISTORICAL_COLUMN_MAP.values())
UPCOMING_COLUMN_ORDER = (
    "product_id",
    "season",
    "item_type",
    "style_code",
    "colour_code",
    "design",
    "category_type",
    "fabric",
    "colour",
    "collection_world",
)


def _standardize_rows(
    dataset: str,
    rows: list[dict[str, Any]],
    column_map: dict[str, str],
    *,
    derived_values: dict[str, Any] | None = None,
    ignored_source_columns: frozenset[str] = frozenset(),
    column_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not rows:
        raise SchemaStandardizationError(f"{dataset} contains no rows")

    source_columns = set(rows[0])
    expected_columns = set(column_map)
    missing = expected_columns - source_columns
    unexpected = source_columns - expected_columns - ignored_source_columns
    if missing or unexpected:
        raise SchemaStandardizationError(
            f"{dataset} schema mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    targets = list(column_map.values())
    if len(targets) != len(set(targets)):
        raise SchemaStandardizationError(
            f"{dataset} column map contains duplicate canonical names"
        )

    derived_values = derived_values or {}
    output_columns = set(targets) | set(derived_values)
    if output_columns != set(column_order):
        raise SchemaStandardizationError(
            f"{dataset} canonical order does not match mapped columns"
        )

    standardized: list[dict[str, Any]] = []
    for row in rows:
        canonical = {
            target: row[source]
            for source, target in column_map.items()
        }
        category_type = str(canonical["category_type"] or "").strip().upper()
        if category_type not in CATEGORY_TYPE_MAP:
            raise SchemaStandardizationError(
                f"{dataset} contains an unknown category type: "
                f"{canonical['category_type']!r}"
            )
        canonical["category_type"] = CATEGORY_TYPE_MAP[category_type]
        canonical.update(derived_values)
        standardized.append({
            column: canonical[column]
            for column in column_order
        })
    return standardized


def standardize_historical_schema(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _standardize_rows(
        "Historical",
        rows,
        HISTORICAL_COLUMN_MAP,
        ignored_source_columns=HISTORICAL_IGNORED_SOURCE_COLUMNS,
        column_order=HISTORICAL_COLUMN_ORDER,
    )


def standardize_upcoming_schema(
    rows: list[dict[str, Any]],
    season: str,
) -> list[dict[str, Any]]:
    return _standardize_rows(
        "Upcoming",
        rows,
        UPCOMING_COLUMN_MAP,
        derived_values={
            "season": season,
        },
        ignored_source_columns=UPCOMING_IGNORED_SOURCE_COLUMNS,
        column_order=UPCOMING_COLUMN_ORDER,
    )
