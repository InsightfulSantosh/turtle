"""End-to-end orchestration for ingestion, preprocessing and model export."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_pipeline.feature_engineering import (
    engineer_features,
    exclude_upcoming_without_historical_item,
    exclude_zero_sales_rows,
)
from data_pipeline.ingestion import WorkbookIngestor
from data_pipeline.preprocessing import (
    CleaningReport,
    preprocess_rows,
)
from data_pipeline.schema import (
    standardize_historical_schema,
    standardize_upcoming_schema,
)
from data_pipeline.settings import PipelineSettings
from data_pipeline.validation import validate_cleaned_rows
from machine_learning.model import build_model_artifact


@dataclass(frozen=True)
class PipelineRunSummary:
    output_path: Path
    historical_items: int
    upcoming_items: int
    model_version: str
    selection_method: str
    backtest_wape: float


class RealDataPipeline:
    """Validated path from raw workbooks to the frontend model artifact."""

    def __init__(self, settings: PipelineSettings | None = None):
        self.settings = settings or PipelineSettings.from_project()
        self.ingestor = WorkbookIngestor(self.settings)

    def build_source(self) -> dict[str, Any]:
        ingested = self.ingestor.ingest()
        historical_rows, historical_report = preprocess_rows(
            "Historical",
            ingested.historical_rows,
            "CON",
        )
        upcoming_rows, upcoming_report = preprocess_rows(
            "Upcoming",
            ingested.upcoming_rows,
            "CC (SEG-1+2+3)",
        )
        historical_rows = standardize_historical_schema(historical_rows)
        upcoming_rows = standardize_upcoming_schema(
            upcoming_rows,
            self.settings.upcoming_season,
        )
        historical_rows, zero_sales_rows_excluded = exclude_zero_sales_rows(
            historical_rows
        )
        upcoming_rows, unseen_item_rows_excluded = (
            exclude_upcoming_without_historical_item(
                historical_rows,
                upcoming_rows,
            )
        )
        validate_cleaned_rows(
            "Historical",
            historical_rows,
            require_unique_identifiers=False,
        )
        validate_cleaned_rows(
            "Upcoming",
            upcoming_rows,
            require_unique_identifiers=True,
        )
        self._write_processed_datasets(
            historical_rows,
            upcoming_rows,
            historical_report,
            upcoming_report,
            zero_sales_rows_excluded,
            unseen_item_rows_excluded,
        )

        features = engineer_features(
            historical_rows,
            upcoming_rows,
        )
        return {
            "meta": self._metadata(
                len(ingested.historical_rows),
                len(ingested.upcoming_rows),
                features,
                historical_report,
                upcoming_report,
                zero_sales_rows_excluded,
                unseen_item_rows_excluded,
            ),
            "historical": features.historical,
            "upcoming": features.upcoming,
        }

    def run(self) -> PipelineRunSummary:
        source = self.build_source()
        artifact = build_model_artifact(source, self._vision_metadata())
        self._write_artifact(artifact)
        model = artifact["meta"]["model"]
        return PipelineRunSummary(
            output_path=self.settings.output_path,
            historical_items=len(artifact["historical"]),
            upcoming_items=len(artifact["upcoming"]),
            model_version=str(model["version"]),
            selection_method=str(model["modelSelection"]),
            backtest_wape=float(model["backtest"]["wape"]),
        )

    def _write_artifact(self, artifact: dict[str, Any]) -> None:
        output = self.settings.output_path
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(artifact, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, output)

    def _metadata(
        self,
        historical_row_count: int,
        upcoming_row_count: int,
        features,
        historical_report: CleaningReport,
        upcoming_report: CleaningReport,
        zero_sales_rows_excluded: int,
        unseen_item_rows_excluded: int,
    ) -> dict[str, Any]:
        return {
            "title": "Turtle Season Intelligence AI",
            "dataMode": "real",
            "upcomingSeason": self.settings.upcoming_season,
            "historicalSource": self.settings.historical_source.name,
            "upcomingSource": self.settings.upcoming_source.name,
            "rawDataPath": "DATA/raw",
            "processedDataPath": "DATA/processed",
            "sourceImageArchive": "COLLATED.zip",
            "imageMappingStatus": (
                "Attribute-only mode: supplied image filenames do not match "
                "the SS27 workbook identifiers."
            ),
            "historicalItems": len(features.historical),
            "upcomingItems": len(features.upcoming),
            "historicalImageCoverage": 0,
            "upcomingImageCoverage": 0,
            "missingUpcomingImages": [item["id"] for item in features.upcoming],
            "historicalSourceRange": (
                f"{self.settings.sheet_name}!A1:W{historical_row_count + 1}"
            ),
            "upcomingSourceRange": (
                f"{self.settings.sheet_name}!A1:K{upcoming_row_count + 1}"
            ),
            "attributeColumnMap": self._attribute_column_map(),
            "excludedConstantAttributes": [],
            "excludedNonComparisonFields": self._excluded_fields(),
            "preprocessing": {
                "identifierFormat": "PREFIX-STYLE-COLOUR",
                "columnSchema": "Canonical snake_case",
                "fabricVocabulary": "Controlled fabric families",
                "historical": historical_report.as_dict(),
                "upcoming": upcoming_report.as_dict(),
            },
            "dataQuality": {
                "duplicateHistoricalRowsRemoved": (
                    features.duplicate_historical_rows_removed
                ),
                "upcomingWithoutHistoricalItem": (
                    features.upcoming_without_historical_item
                ),
                "zeroSalesHistoricalRowsExcluded": zero_sales_rows_excluded,
                "upcomingRowsExcludedUnseenItem": unseen_item_rows_excluded,
                "historicalIdentifierCollisions": (
                    historical_report.identifier_collisions_introduced
                ),
                "historicalFabricReviewRequired": (
                    historical_report.review_required_fabric
                ),
                "historicalFabricUnspecified": historical_report.unspecified_fabric,
                "upcomingFabricReviewRequired": (
                    upcoming_report.review_required_fabric
                ),
                "upcomingFabricUnspecified": upcoming_report.unspecified_fabric,
            },
        }

    def _write_processed_datasets(
        self,
        historical_rows: list[dict[str, Any]],
        upcoming_rows: list[dict[str, Any]],
        historical_report: CleaningReport,
        upcoming_report: CleaningReport,
        zero_sales_rows_excluded: int = 0,
        unseen_item_rows_excluded: int = 0,
    ) -> None:
        self.settings.processed_root.mkdir(parents=True, exist_ok=True)
        self._write_csv(
            self.settings.historical_processed_output,
            historical_rows,
        )
        self._write_csv(
            self.settings.upcoming_processed_output,
            upcoming_rows,
        )
        validation_rows = [
            self._validation_row(
                historical_report,
                len(historical_rows[0]),
                historical_rows,
                zero_sales_rows_excluded,
            ),
            self._validation_row(
                upcoming_report,
                len(upcoming_rows[0]),
                upcoming_rows,
                unseen_item_rows_excluded=unseen_item_rows_excluded,
            ),
        ]
        self._write_csv(self.settings.validation_output, validation_rows)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError(f"Cannot write an empty processed dataset: {path}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    @staticmethod
    def _validation_row(
        report: CleaningReport,
        column_count: int,
        rows: list[dict[str, Any]],
        zero_sales_rows_excluded: int = 0,
        unseen_item_rows_excluded: int = 0,
    ) -> dict[str, Any]:
        return {
            "Dataset": report.dataset,
            "Identifier column": report.identifier_column,
            "Source rows": report.rows,
            "Rows": len(rows),
            "Rows excluded: zero sales": zero_sales_rows_excluded,
            "Rows excluded: unseen item": unseen_item_rows_excluded,
            "Columns": column_count,
            "Identifier values changed": report.identifier_values_changed,
            "Unique identifiers before": report.unique_identifiers_before,
            "Unique identifiers after": report.unique_identifiers_after,
            "Identifier collisions introduced": (
                report.identifier_collisions_introduced
            ),
            "Invalid formatted identifiers": 0,
            "Blank fabric": 0,
            "Review Required fabric": report.review_required_fabric,
            "Unspecified fabric": report.unspecified_fabric,
            "Category types": ", ".join(sorted({
                str(row["category_type"])
                for row in rows
            })),
            "Canonical schema applied": True,
        }

    @staticmethod
    def _vision_metadata() -> dict[str, Any]:
        return {
            "engine": (
                "Attribute-only matching; the supplied archive has no SS27 "
                "identifier matches"
            ),
            "modelId": "not-available",
            "modelRevision": None,
            "embeddingDimension": 0,
            "device": "not-used",
            "historicalCoverage": 0,
            "upcomingCoverage": 0,
            "calibrationMethod": "Not applicable without mapped product images",
            "distances": [],
        }

    @staticmethod
    def _attribute_column_map() -> dict[str, dict[str, str]]:
        return {
            "item": {
                "historicalColumn": "item_type",
                "upcomingColumn": "item_type",
            },
            "design": {
                "historicalColumn": "design",
                "upcomingColumn": "design",
            },
            "category_type": {
                "historicalColumn": "category_type",
                "upcomingColumn": "category_type",
            },
            "fabric": {
                "historicalColumn": "fabric",
                "upcomingColumn": "fabric",
            },
            "colour": {
                "historicalColumn": "colour",
                "upcomingColumn": "colour",
            },
        }

    @staticmethod
    def _excluded_fields() -> list[dict[str, str]]:
        return [
            {
                "label": "Identifiers",
                "historicalColumn": "product_id, style_code",
                "upcomingColumn": "product_id, style_code",
                "reason": (
                    "Identifiers locate products but do not describe reusable "
                    "product similarity."
                ),
            },
            {
                "label": "Colour variant code",
                "historicalColumn": "colour_code",
                "upcomingColumn": "colour_code",
                "reason": (
                    "Variant codes are identifiers; colour names are used for "
                    "similarity."
                ),
            },
            {
                "label": "Season",
                "historicalColumn": "season",
                "upcomingColumn": "season",
                "reason": (
                    "Season supports tracing and temporal validation but is "
                    "outside the five approved product attributes."
                ),
            },
            {
                "label": "Demand outcomes",
                "historicalColumn": (
                    "total_order_quantity, dispatch_quantity, "
                    "sales_quantity, sell_through"
                ),
                "upcomingColumn": "—",
                "reason": (
                    "Historical outcomes train and validate demand; they cannot "
                    "be product-similarity inputs."
                ),
            },
            {
                "label": "Commercial operations",
                "historicalColumn": (
                    "max_quantity, pt_quantity, ril_quantity, "
                    "first_dispatch_date, ageing_days, weekly_sell_through"
                ),
                "upcomingColumn": "collection_world",
                "reason": (
                    "These fields have no like-for-like counterpart in the "
                    "other workbook."
                ),
            },
        ]
