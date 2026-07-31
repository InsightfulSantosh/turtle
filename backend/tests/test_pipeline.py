from __future__ import annotations

import csv
from pathlib import Path

from data_pipeline.feature_engineering import FeatureEngineeringResult
from data_pipeline.pipeline import RealDataPipeline
from data_pipeline.preprocessing import CleaningReport
from data_pipeline.settings import PipelineSettings


def cleaning_report(dataset: str, identifier_column: str) -> CleaningReport:
    return CleaningReport(
        dataset=dataset,
        identifier_column=identifier_column,
        rows=1,
        identifier_values_changed=1,
        unique_identifiers_before=1,
        unique_identifiers_after=1,
        identifier_collisions_introduced=0,
        review_required_fabric=0,
        unspecified_fabric=0,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_processed_datasets_are_written_below_data_processed(
    tmp_path: Path,
) -> None:
    settings = PipelineSettings(
        data_root=tmp_path / "DATA",
        temporary_root=tmp_path / "tmp",
        output_path=tmp_path / "artifact.json",
    )
    pipeline = RealDataPipeline(settings)
    historical_rows = [
        {
            "product_id": "OTSH-23850-1001",
            "fabric": "Cotton – 100%",
            "category_type": "FORMAL",
        },
    ]
    upcoming_rows = [
        {
            "product_id": "OTTR-300171-1001",
            "fabric": "Polyester – Viscose Stretch",
            "category_type": "CASUAL",
        },
    ]

    pipeline._write_processed_datasets(
        historical_rows,
        upcoming_rows,
        cleaning_report("Historical", "product_id"),
        cleaning_report("Upcoming", "product_id"),
    )

    assert settings.raw_root == tmp_path / "DATA" / "raw"
    assert read_csv(settings.historical_processed_output) == [
        {
            "product_id": "OTSH-23850-1001",
            "fabric": "Cotton – 100%",
            "category_type": "FORMAL",
        },
    ]
    assert read_csv(settings.upcoming_processed_output) == [
        {
            "product_id": "OTTR-300171-1001",
            "fabric": "Polyester – Viscose Stretch",
            "category_type": "CASUAL",
        },
    ]
    validation = read_csv(settings.validation_output)
    assert [row["Dataset"] for row in validation] == ["Historical", "Upcoming"]
    assert all(row["Canonical schema applied"] == "True" for row in validation)


def test_catalogue_images_are_mapped_by_source_identifier(
    tmp_path: Path,
) -> None:
    settings = PipelineSettings(
        data_root=tmp_path / "DATA",
        temporary_root=tmp_path / "tmp",
        output_path=tmp_path / "artifact.json",
    )
    settings.historical_image_root.mkdir(parents=True)
    settings.upcoming_image_root.mkdir(parents=True)
    (settings.historical_image_root / "OTSH-100-1001.JPG").write_bytes(b"image")
    (settings.upcoming_image_root / "OTSH-200-1002.jpg").write_bytes(b"image")
    features = FeatureEngineeringResult(
        historical=[
            {
                "id": "AW25-OTSH-100-1001",
                "sourceId": "otsh-100-1001",
            },
            {
                "id": "AW25-OTSH-101-1001",
                "sourceId": "OTSH-101-1001",
            },
        ],
        upcoming=[
            {"id": "OTSH-200-1002"},
            {"id": "OTSH-201-1002"},
        ],
        duplicate_historical_rows_removed=0,
        upcoming_without_historical_item=0,
    )

    coverage = RealDataPipeline(settings)._attach_product_images(features)

    assert coverage == (1, 1)
    assert features.historical[0]["imageUrl"] == ("/product-images/historical/otsh-100-1001")
    assert features.historical[1]["imageUrl"] is None
    assert features.upcoming[0]["imageUrl"] == ("/product-images/upcoming/OTSH-200-1002")
    assert features.upcoming[1]["imageUrl"] is None
    assert all(item["hasVisualFeature"] is False for item in [*features.historical, *features.upcoming])


def test_item_type_scope_keeps_only_requested_catalogue_items(tmp_path: Path) -> None:
    settings = PipelineSettings(
        data_root=tmp_path / "DATA",
        temporary_root=tmp_path / "tmp",
        output_path=tmp_path / "artifact.json",
    )
    source = {
        "meta": {"title": "test"},
        "historical": [
            {"id": "AW25-OTSH-1", "itemType": "OTSH", "imageUrl": "/historical/1"},
            {"id": "AW25-OTTR-1", "itemType": "OTTR", "imageUrl": "/historical/2"},
        ],
        "upcoming": [
            {"id": "OTSH-2", "itemType": "OTSH", "imageUrl": "/upcoming/2"},
            {"id": "OTTR-2", "itemType": "OTTR", "imageUrl": None},
        ],
    }

    scoped = RealDataPipeline(settings)._restrict_source_to_item_type(source, "otsh")

    assert [item["id"] for item in scoped["historical"]] == ["AW25-OTSH-1"]
    assert [item["id"] for item in scoped["upcoming"]] == ["OTSH-2"]
    assert scoped["meta"]["artifactScope"] == {
        "itemType": "OTSH",
        "historicalItems": 1,
        "upcomingItems": 1,
    }
