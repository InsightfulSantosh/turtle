"""End-to-end orchestration for ingestion, preprocessing and model export."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_pipeline.feature_engineering import (
    FeatureEngineeringResult,
    engineer_features,
    exclude_upcoming_without_historical_item,
    exclude_zero_sales_rows,
)
from data_pipeline.images import attach_catalog_images
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
    evidence_policy: str
    accepted_matches: int


class RealDataPipeline:
    """Validated path from raw workbooks to the frontend model artifact."""

    def __init__(self, settings: PipelineSettings | None = None):
        self.settings = settings or PipelineSettings.from_project()
        self.ingestor = WorkbookIngestor(self.settings)

    def build_source(self) -> dict[str, Any]:
        ingested = self.ingestor.ingest()
        historical_input = self._normalize_historical_source_layout(
            ingested.historical_rows
        )
        historical_rows, historical_report = preprocess_rows(
            "Historical",
            historical_input,
            "CON",
        )
        # Support the current SS27 master layout, which provides the product
        # identifier and category fields under the workbook's display names.
        # Normalize it into the legacy canonical source names before cleaning.
        upcoming_input = []
        for row in ingested.upcoming_rows:
            if "IMAGE ID" in row:
                normalized = dict(row)
                normalized["CC (SEG-1+2+3)"] = row.get("IMAGE ID")
                normalized["COLOUR NAME"] = row.get("COLOR_NAME")
                normalized["CAT-3"] = row.get("CAT3")
                normalized.pop("IMAGE ID", None)
                normalized.pop("COLOR_NAME", None)
                normalized.pop("CAT3", None)
                upcoming_input.append(normalized)
            else:
                upcoming_input.append(row)
        upcoming_rows, upcoming_report = preprocess_rows(
            "Upcoming",
            upcoming_input,
            "CC (SEG-1+2+3)",
        )
        historical_rows = standardize_historical_schema(historical_rows)
        upcoming_rows = standardize_upcoming_schema(
            upcoming_rows,
            self.settings.upcoming_season,
        )
        historical_rows, zero_sales_rows_excluded = exclude_zero_sales_rows(historical_rows)
        upcoming_rows, unseen_item_rows_excluded = exclude_upcoming_without_historical_item(
            historical_rows,
            upcoming_rows,
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
        historical_image_coverage, upcoming_image_coverage = self._attach_product_images(features)
        return {
            "meta": self._metadata(
                len(ingested.historical_rows),
                len(ingested.upcoming_rows),
                features,
                historical_report,
                upcoming_report,
                zero_sales_rows_excluded,
                unseen_item_rows_excluded,
                historical_image_coverage,
                upcoming_image_coverage,
            ),
            "historical": features.historical,
            "upcoming": features.upcoming,
        }

    @staticmethod
    def _normalize_historical_source_layout(
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Map the current historical workbook headers to the legacy source schema.

        The latest workbook uses the same catalogue-oriented headers as the SS27
        master. Keeping this small compatibility layer allows the pipeline to
        accept both supplied historical layouts without weakening schema checks.
        """

        if not rows or "IMAGE ID" not in rows[0]:
            return rows

        renamed_columns = {
            "IMAGE ID": "CON",
            "SEGMENT1": "ITEM TYPE",
            "SEGMENT2": "SORT",
            "SEGMENT3": "COLOR",
            "COLOR_NAME": "COLOR__2",
            "SELL THROUGH": "SELL THR",
            "WEEKLY SELL THROUGH": "WKLY SELL THRU",
        }
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            normalized = dict(row)
            for source, target in renamed_columns.items():
                normalized[target] = normalized.pop(source)
            normalized_rows.append(normalized)
        return normalized_rows

    def run(
        self,
        vision_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        *,
        item_type: str | None = None,
        sample_size: int | None = None,
        sample_seed: int = 27,
    ) -> PipelineRunSummary:
        source = self.build_source()
        if item_type is not None and sample_size is not None:
            raise ValueError("item_type and sample_size cannot be used together")
        if item_type is not None:
            source = self._restrict_source_to_item_type(source, item_type)
        if sample_size is not None:
            source = self._sample_upcoming_catalogue(source, sample_size, sample_seed)
        vision_output = vision_builder(source) if vision_builder is not None else self._vision_metadata(source)
        artifact = build_model_artifact(source, vision_output)
        self._write_artifact(artifact)
        model = artifact["meta"]["model"]
        return PipelineRunSummary(
            output_path=self.settings.output_path,
            historical_items=len(artifact["historical"]),
            upcoming_items=len(artifact["upcoming"]),
            model_version=str(model["version"]),
            evidence_policy=str(model["evidencePolicy"]),
            accepted_matches=sum(
                not item["recommendation"]["noSuitableMatch"]
                for item in artifact["upcoming"]
            ),
        )

    @staticmethod
    def _restrict_source_to_item_type(source: dict[str, Any], item_type: str) -> dict[str, Any]:
        """Create an auditable, self-contained item-type artifact scope."""

        selected_type = " ".join(item_type.upper().split())
        if not selected_type:
            raise ValueError("item_type must not be blank")

        def belongs_to_scope(item: dict[str, Any]) -> bool:
            return " ".join(str(item.get("itemType") or "").upper().split()) == selected_type

        historical = [item for item in source["historical"] if belongs_to_scope(item)]
        upcoming = [item for item in source["upcoming"] if belongs_to_scope(item)]
        if not historical:
            raise ValueError(f"No historical products found for item type {selected_type}")
        if not upcoming:
            raise ValueError(f"No upcoming products found for item type {selected_type}")

        meta = dict(source["meta"])
        historical_coverage = sum(bool(item.get("imageUrl")) for item in historical)
        upcoming_coverage = sum(bool(item.get("imageUrl")) for item in upcoming)
        meta.update(
            {
                "historicalItems": len(historical),
                "upcomingItems": len(upcoming),
                "historicalImageCoverage": historical_coverage,
                "upcomingImageCoverage": upcoming_coverage,
                "missingUpcomingImages": [item["id"] for item in upcoming if not item.get("imageUrl")],
                "imageMappingStatus": (
                    f"Mapped {historical_coverage} historical and {upcoming_coverage} upcoming "
                    f"{selected_type} product images by catalogue identifier."
                ),
                "artifactScope": {
                    "itemType": selected_type,
                    "historicalItems": len(historical),
                    "upcomingItems": len(upcoming),
                },
            }
        )
        return {"meta": meta, "historical": historical, "upcoming": upcoming}

    @staticmethod
    def _sample_upcoming_catalogue(
        source: dict[str, Any],
        sample_size: int,
        sample_seed: int = 27,
    ) -> dict[str, Any]:
        """Select a deterministic, category-balanced catalogue preview.

        The historical catalogue remains complete so preview products are matched
        against the same evidence pool that will be used by the production build.
        """

        upcoming = source["upcoming"]
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than zero")
        if sample_size > len(upcoming):
            raise ValueError(
                f"sample_size {sample_size} exceeds the {len(upcoming)} upcoming products"
            )

        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in upcoming:
            key = (
                str(item.get("itemType") or "UNSPECIFIED"),
                str(item.get("categoryType") or "UNSPECIFIED"),
            )
            strata[key].append(item)
        if sample_size < len(strata):
            raise ValueError(
                f"sample_size {sample_size} cannot cover all {len(strata)} "
                "item-type/category strata"
            )

        def balanced_allocation(
            capacities: dict[Any, int], total: int
        ) -> dict[Any, int]:
            """Round-robin allocation keeps small catalogue groups visible."""

            allocation = {key: 1 for key in capacities}
            remaining = total - len(allocation)
            while remaining:
                available = [
                    key
                    for key in sorted(capacities)
                    if allocation[key] < capacities[key]
                ]
                if not available:
                    raise ValueError("Not enough catalogue products for requested sample")
                for key in available:
                    if remaining == 0:
                        break
                    allocation[key] += 1
                    remaining -= 1
            return allocation

        item_capacities: dict[str, int] = defaultdict(int)
        for (item_type, _), items in strata.items():
            item_capacities[item_type] += len(items)
        item_allocations = balanced_allocation(dict(item_capacities), sample_size)

        allocations: dict[tuple[str, str], int] = {}
        for item_type in sorted(item_allocations):
            category_capacities = {
                key: len(items)
                for key, items in strata.items()
                if key[0] == item_type
            }
            allocations.update(
                balanced_allocation(
                    category_capacities,
                    item_allocations[item_type],
                )
            )

        selected: list[dict[str, Any]] = []
        for key in sorted(strata):
            ranked = sorted(
                strata[key],
                key=lambda item: hashlib.sha256(
                    f"{sample_seed}:{item['id']}".encode("utf-8")
                ).hexdigest(),
            )
            selected.extend(ranked[: allocations[key]])
        selected.sort(key=lambda item: (item["itemType"], item["categoryType"], item["id"]))

        meta = dict(source["meta"])
        full_upcoming_count = len(upcoming)
        full_upcoming_coverage = int(meta.get("upcomingImageCoverage", 0))
        sample_coverage = sum(bool(item.get("imageUrl")) for item in selected)
        stratum_counts = [
            {
                "itemType": key[0],
                "categoryType": key[1],
                "sampleItems": allocations[key],
                "fullItems": len(strata[key]),
            }
            for key in sorted(strata)
        ]
        meta.update(
            {
                "upcomingItems": len(selected),
                "upcomingImageCoverage": sample_coverage,
                "missingUpcomingImages": [
                    item["id"] for item in selected if not item.get("imageUrl")
                ],
                "imageMappingStatus": (
                    f"Preview mapped {sample_coverage} of {len(selected)} sampled upcoming "
                    f"images; full SS27 catalogue coverage is {full_upcoming_coverage} "
                    f"of {full_upcoming_count}."
                ),
                "previewSample": True,
                "previewSampleSize": len(selected),
                "previewSampleSeed": sample_seed,
                "previewSamplingMethod": (
                    "Deterministic balanced stratification by itemType, then categoryType"
                ),
                "fullUpcomingItems": full_upcoming_count,
                "fullUpcomingImageCoverage": full_upcoming_coverage,
                "previewStrata": stratum_counts,
                "artifactScope": {
                    "mode": "preview",
                    "historicalItems": len(source["historical"]),
                    "upcomingItems": len(selected),
                    "fullUpcomingItems": full_upcoming_count,
                },
            }
        )
        return {
            "meta": meta,
            "historical": source["historical"],
            "upcoming": selected,
        }

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
        historical_image_coverage: int,
        upcoming_image_coverage: int,
    ) -> dict[str, Any]:
        missing_upcoming_images = [item["id"] for item in features.upcoming if not item.get("imageUrl")]
        return {
            "title": "Turtle Season Intelligence AI",
            "dataMode": "real",
            "upcomingSeason": self.settings.upcoming_season,
            "historicalSource": self.settings.historical_source.name,
            "upcomingSource": self.settings.upcoming_source.name,
            "rawDataPath": "DATA/raw",
            "processedDataPath": "DATA/processed",
            "sourceImageArchive": (
                f"{self.settings.historical_image_root.name}, {self.settings.upcoming_image_root.name}"
            ),
            "imageMappingStatus": (
                f"Mapped {historical_image_coverage} historical and "
                f"{upcoming_image_coverage} upcoming product images by "
                "catalogue identifier."
            ),
            "historicalItems": len(features.historical),
            "upcomingItems": len(features.upcoming),
            "historicalImageCoverage": historical_image_coverage,
            "upcomingImageCoverage": upcoming_image_coverage,
            "missingUpcomingImages": missing_upcoming_images,
            "historicalSourceRange": (f"{self.settings.sheet_name}!A1:W{historical_row_count + 1}"),
            "upcomingSourceRange": (f"{self.settings.sheet_name}!A1:K{upcoming_row_count + 1}"),
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
                "duplicateHistoricalRowsRemoved": (features.duplicate_historical_rows_removed),
                "upcomingWithoutHistoricalItem": (features.upcoming_without_historical_item),
                "zeroSalesHistoricalRowsExcluded": zero_sales_rows_excluded,
                "upcomingRowsExcludedUnseenItem": unseen_item_rows_excluded,
                "historicalIdentifierCollisions": (historical_report.identifier_collisions_introduced),
                "historicalFabricReviewRequired": (historical_report.review_required_fabric),
                "historicalFabricUnspecified": historical_report.unspecified_fabric,
                "upcomingFabricReviewRequired": (upcoming_report.review_required_fabric),
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
            "Identifier collisions introduced": (report.identifier_collisions_introduced),
            "Invalid formatted identifiers": 0,
            "Blank fabric": 0,
            "Review Required fabric": report.review_required_fabric,
            "Unspecified fabric": report.unspecified_fabric,
            "Category types": ", ".join(sorted({str(row["category_type"]) for row in rows})),
            "Canonical schema applied": True,
        }

    def _attach_product_images(
        self,
        features: FeatureEngineeringResult,
    ) -> tuple[int, int]:
        historical_coverage = attach_catalog_images(
            features.historical,
            image_directory=self.settings.historical_image_root,
            catalog="historical",
            identifier_field="sourceId",
        )
        upcoming_coverage = attach_catalog_images(
            features.upcoming,
            image_directory=self.settings.upcoming_image_root,
            catalog="upcoming",
            identifier_field="id",
        )
        return historical_coverage, upcoming_coverage

    @staticmethod
    def _vision_metadata(source: dict[str, Any]) -> dict[str, Any]:
        return {
            "engine": (
                "Attribute-only matching; image files are mapped but visual "
                "embeddings were not requested for this build"
            ),
            "modelId": "not-available",
            "modelRevision": None,
            "embeddingDimension": 0,
            "device": "not-used",
            "historicalCoverage": 0,
            "upcomingCoverage": 0,
            "mappedHistoricalImages": source["meta"]["historicalImageCoverage"],
            "mappedUpcomingImages": source["meta"]["upcomingImageCoverage"],
            "calibrationMethod": "Not applicable without generated embeddings",
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
                "reason": ("Identifiers locate products but do not describe reusable product similarity."),
            },
            {
                "label": "Colour variant code",
                "historicalColumn": "colour_code",
                "upcomingColumn": "colour_code",
                "reason": ("Variant codes are identifiers; colour names are used for similarity."),
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
                "historicalColumn": ("total_order_quantity, dispatch_quantity, sales_quantity, sell_through"),
                "upcomingColumn": "—",
                "reason": ("Historical outcomes train and validate demand; they cannot be product-similarity inputs."),
            },
            {
                "label": "Commercial operations",
                "historicalColumn": (
                    "max_quantity, pt_quantity, ril_quantity, first_dispatch_date, ageing_days, weekly_sell_through"
                ),
                "upcomingColumn": "collection_world",
                "reason": ("These fields have no like-for-like counterpart in the other workbook."),
            },
        ]
