from __future__ import annotations

import pytest

from data_pipeline.schema import (
    HISTORICAL_COLUMN_MAP,
    HISTORICAL_COLUMN_ORDER,
    UPCOMING_COLUMN_MAP,
    UPCOMING_COLUMN_ORDER,
    SchemaStandardizationError,
    standardize_historical_schema,
    standardize_upcoming_schema,
)


def test_historical_schema_uses_only_canonical_column_names() -> None:
    source = {
        column: f"value-{index}"
        for index, column in enumerate(HISTORICAL_COLUMN_MAP)
    }
    source["CAT3"] = "TAILORED"
    source.update({
        "MRP": 1999,
        "SLEEVS": "FULL",
        "PROV": "YES",
    })
    standardized = standardize_historical_schema([source])[0]

    assert tuple(standardized) == HISTORICAL_COLUMN_ORDER
    assert standardized["product_id"] == source["CON"]
    assert standardized["fabric"] == source["CAT4"]
    assert not set(standardized) & set(HISTORICAL_COLUMN_MAP)
    assert not {"MRP", "SLEEVS", "PROV"} & set(standardized)


def test_upcoming_schema_adds_shared_fields_and_removes_source_naming() -> None:
    source = {
        column: f"value-{index}"
        for index, column in enumerate(UPCOMING_COLUMN_MAP)
    }
    source["CAT-3"] = "RELAXED WASH"
    source.update({
        "PROPOSED MRP": 1999,
        "SL": "FULL",
    })
    standardized = standardize_upcoming_schema([source], "SS27")[0]

    assert tuple(standardized) == UPCOMING_COLUMN_ORDER
    assert standardized["product_id"] == source["CC (SEG-1+2+3)"]
    assert standardized["season"] == "SS27"
    assert standardized["category_type"] in {
        "FORMAL",
        "CASUAL",
        "DENIM",
        "CEREMONIAL",
    }
    assert not set(standardized) & set(UPCOMING_COLUMN_MAP)
    assert not {"PROPOSED MRP", "SL"} & set(standardized)


def test_schema_standardization_fails_on_unmapped_columns() -> None:
    source = {
        column: f"value-{index}"
        for index, column in enumerate(UPCOMING_COLUMN_MAP)
    }
    source["CAT-3"] = "TAILORED"
    source["UNEXPECTED"] = "schema drift"

    with pytest.raises(SchemaStandardizationError, match="unexpected"):
        standardize_upcoming_schema([source], "SS27")
