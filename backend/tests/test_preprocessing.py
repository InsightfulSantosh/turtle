from __future__ import annotations

import pytest

from data_pipeline.preprocessing import (
    PreprocessingError,
    clean_product_identifier,
    preprocess_rows,
    standardize_fabric,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("OTSU51021001F2PC", "OTSU-5102-1001"),
        ("OTSU5102T1001F3PC", "OTSU-5102T-1001"),
        ("OTSH59312V1003FNA", "OTSH-59312V-1003"),
        ("OTSH-59915-1001", "OTSH-59915-1001"),
        (" ottr 300171 1001 ", "OTTR-300171-1001"),
    ],
)
def test_identifier_parser_standardizes_supported_formats(
    raw: str,
    expected: str,
) -> None:
    assert clean_product_identifier(raw) == expected


def test_identifier_parser_rejects_ambiguous_values() -> None:
    with pytest.raises(PreprocessingError, match="Unable to parse"):
        clean_product_identifier("BAD-CODE")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("100% Cotton", "Cotton – 100%"),
        ("Cotton 2% Elastane", "Cotton – Stretch"),
        ("PV LINEN", "Polyester – Viscose Linen Blend"),
        ("PVS", "Polyester – Viscose Stretch"),
        ("AS PER SAMPLE", "Unspecified"),
        ("KNITTED", "Unspecified"),
        ("PVL", "Review Required"),
        ("TR", "Review Required"),
        ("EXCEL", "Review Required"),
    ],
)
def test_fabric_mapping_uses_controlled_vocabulary(
    raw: str,
    expected: str,
) -> None:
    assert standardize_fabric(raw) == expected


def test_preprocessing_reports_identifier_collisions_and_fabric_review() -> None:
    rows = [
        {"CON": "OTSU51021001F2PC", "CAT4": "AS PER SAMPLE"},
        {"CON": "OTSU51021001F3PC", "CAT4": "TR"},
    ]
    cleaned, report = preprocess_rows("Historical", rows, "CON")
    assert [row["CON"] for row in cleaned] == [
        "OTSU-5102-1001",
        "OTSU-5102-1001",
    ]
    assert report.identifier_collisions_introduced == 1
    assert report.review_required_fabric == 1
    assert report.unspecified_fabric == 1
