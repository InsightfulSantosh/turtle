from __future__ import annotations

import csv
from pathlib import Path

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
