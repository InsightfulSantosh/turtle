"""Asynchronous dual-mode analysis orchestration."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from analysis_service.catalogs import (
    SUPPORTED_CATALOGUE_SUFFIXES,
    CatalogValidationError,
    ValidationIssue,
    build_feature_source,
    validate_catalog,
)
from analysis_service.store import RunStore, ist_now
from data_pipeline.feature_engineering import build_upcoming_features
from data_pipeline.preprocessing import normalize_text
from fashion_matching.artifact_vision import (
    VisionPaths,
    build_artifact_vision_output,
    load_catalogue_embeddings,
    save_catalogue_embeddings,
)
from fashion_matching.config import MatchingSettings
from fashion_matching.encoders import create_dino_encoder, create_encoder
from fashion_matching.preprocessing import ImagePreprocessor
from machine_learning.model import build_model_artifact


def write_validation_report(path: Path, records: list[ValidationIssue]) -> None:
    fieldnames = ["recordType", "catalog", "row", "productId", "field", "code", "message", "severity"]
    record_order = {"summary": 0, "check": 1, "issue": 2}
    ordered = sorted(
        records,
        key=lambda record: (
            record_order.get(record.record_type, 3),
            record.catalog,
            record.product_id,
            record.code,
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.as_dict() for record in ordered)


class AnalysisProcessor:
    def __init__(
        self,
        store: RunStore,
        *,
        vision_enabled: bool | None = None,
        max_workers: int = 2,
    ):
        self.store = store
        self.vision_enabled = (
            os.getenv("TURTLE_RUN_VISION", "1").strip().lower() not in {"0", "false", "no", "off"}
            if vision_enabled is None
            else vision_enabled
        )
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="turtle-analysis")
        self._models: tuple[Any, Any, ImagePreprocessor, MatchingSettings] | None = None
        self._model_lock = threading.Lock()

    def submit(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        if run["status"] != "uploading":
            raise ValueError("run is not accepting uploads")
        self.store.update_run(
            run_id,
            status="queued",
            stage="queued",
            message="Waiting for an analysis worker",
            analysis_started_at=ist_now(),
            completed_at=None,
        )
        self.executor.submit(self.process, run_id)

    def resume_incomplete(self) -> None:
        """Restart durable queued/processing work after a service restart."""
        for run in self.store.incomplete_analysis_runs():
            self.store.update_run(
                run["id"],
                status="queued",
                stage="queued",
                message="Resuming analysis after service restart",
                eta_seconds=None,
            )
            self.executor.submit(self.process, run["id"])

    def _check_cancelled(self, run_id: str) -> None:
        if self.store.get_run(run_id)["cancelRequested"]:
            self.store.update_run(
                run_id,
                status="cancelled",
                stage="cancelled",
                message="Run cancelled",
                completed_at=ist_now(),
            )
            raise InterruptedError("run cancelled")

    @staticmethod
    def _single_catalogue(staging: Path, catalog: str) -> Path:
        directories = [staging / catalog / "catalogue", staging / catalog / "csv"]
        candidates = sorted(
            path
            for directory in directories
            if directory.is_dir()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_CATALOGUE_SUFFIXES
        )
        if len(candidates) != 1:
            raise CatalogValidationError(f"{catalog} requires exactly one catalogue file")
        return candidates[0]

    def _write_issues(self, run_id: str, issues: list[ValidationIssue]) -> Path:
        path = self.store.staging_path(run_id) / "validation-report.csv"
        write_validation_report(path, issues)
        self.store.update_run(run_id, issues_path=str(path))
        return path

    def _load_models(self) -> tuple[Any, Any, ImagePreprocessor, MatchingSettings]:
        with self._model_lock:
            if self._models is None:
                settings = MatchingSettings.from_environment()
                encoder = create_encoder(settings.model_id, revision=settings.model_revision, device=settings.device)
                reranker = (
                    create_dino_encoder(
                        settings.dino_model_id,
                        revision=settings.dino_model_revision,
                        device=settings.device,
                    )
                    if settings.dino_reranker_enabled
                    else None
                )
                preprocessor = ImagePreprocessor(
                    version=settings.preprocessing_version,
                    max_bytes=settings.max_image_bytes,
                    max_pixels=settings.max_image_pixels,
                    minimum_dimension=settings.min_image_dimension,
                    pad_to_square=settings.pad_to_square,
                )
                self._models = (encoder, reranker, preprocessor, settings)
            return self._models

    @staticmethod
    def _attribute_only_vision(source: dict[str, Any]) -> dict[str, Any]:
        return {
            "engine": "Upload validated; visual inference disabled by TURTLE_RUN_VISION",
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

    def _vision(
        self,
        source: dict[str, Any],
        *,
        paths: VisionPaths,
        historical_features_path: Path,
        reuse_historical: bool,
        run_id: str,
    ) -> dict[str, Any]:
        historical_total = 0 if reuse_historical else int(source["meta"]["historicalImageCoverage"])
        upcoming_total = int(source["meta"]["upcomingImageCoverage"])
        total_images = historical_total + upcoming_total
        # Publish the denominators before the encoders load. A cold model load
        # takes tens of seconds, and reporting the real scale of the build up
        # front is the difference between a stalled-looking bar and a run the
        # user can see the size of.
        self.store.update_run(
            run_id,
            historical_total=historical_total,
            upcoming_total=upcoming_total,
            historical_processed=0,
            upcoming_processed=0,
            total_count=total_images,
            processed_count=0,
            cache_hits=0,
            eta_seconds=None,
        )
        if not self.vision_enabled:
            return self._attribute_only_vision(source)
        self.store.update_run(run_id, message="Loading visual encoders")
        encoder, reranker, preprocessor, matching = self._load_models()
        feature_kwargs = {
            "model_id": encoder.model_id,
            "model_revision": encoder.revision,
            "reranker_model_id": reranker.model_id if reranker is not None else None,
            "reranker_model_revision": reranker.revision if reranker is not None else None,
            "preprocessing_version": preprocessor.version,
        }
        historical_embeddings = (
            load_catalogue_embeddings(historical_features_path, **feature_kwargs) if reuse_historical else None
        )
        callback = (
            None
            if reuse_historical
            else lambda features: save_catalogue_embeddings(
                historical_features_path,
                features,
                **feature_kwargs,
            )
        )
        historical_cache_hits = 0
        historical_processed = 0
        upcoming_processed = 0

        def progress(event: dict[str, int | float | str]) -> None:
            nonlocal historical_cache_hits, historical_processed, upcoming_processed
            catalog = str(event["catalog"])
            phase_processed = int(event["processed"])
            if catalog == "historical":
                historical_cache_hits = int(event["cached"])
                historical_processed = phase_processed
                processed = phase_processed
                cache_hits = historical_cache_hits
                stage = "indexing_history"
                label = "Historical features"
            else:
                upcoming_processed = phase_processed
                processed = historical_total + phase_processed
                cache_hits = historical_cache_hits + int(event["cached"])
                stage = "matching_upcoming"
                label = "Upcoming features"
            rate = float(event["rate"])
            remaining = max(total_images - processed, 0)
            eta_seconds = math.ceil(remaining / rate) if rate > 0 else None
            measured_progress = 0.15 + (0.72 * processed / total_images if total_images else 0.72)
            eta_label = f" · ETA {eta_seconds}s" if eta_seconds is not None else ""
            self.store.update_run(
                run_id,
                stage=stage,
                progress=min(measured_progress, 0.87),
                message=(
                    f"{label} {phase_processed}/{int(event['total'])} · "
                    f"{rate:.1f} images/s · {cache_hits} cache hits{eta_label}"
                ),
                processed_count=processed,
                total_count=total_images,
                cache_hits=cache_hits,
                eta_seconds=eta_seconds,
                historical_processed=historical_processed,
                upcoming_processed=upcoming_processed,
            )
            self._check_cancelled(run_id)

        return build_artifact_vision_output(
            source,
            settings=paths,
            encoder=encoder,
            preprocessor=preprocessor,
            reranker=reranker,
            candidate_count=matching.dino_candidate_count,
            reranker_weight_grid=matching.dino_weight_grid,
            require_same_item_type=matching.dino_require_same_item_type,
            require_same_design=matching.dino_require_same_design,
            visual_only_ranking=matching.visual_only_ranking,
            pattern_gate_enabled=matching.pattern_gate_enabled,
            pattern_max_distance=matching.pattern_max_distance,
            colour_gate_enabled=matching.colour_gate_enabled,
            colour_max_distance=matching.colour_max_distance,
            appearance_weights={
                "neural": matching.appearance_neural_weight,
                "colour": matching.appearance_colour_weight,
                "texture": matching.appearance_texture_weight,
            },
            batch_size=matching.batch_size,
            historical_embeddings=historical_embeddings,
            historical_embeddings_callback=callback,
            feature_cache_root=self.store.root / "feature-cache",
            progress_callback=progress,
        )

    def _prune_feature_cache(self, referenced_keys: set[str]) -> None:
        cache_root = self.store.root / "feature-cache"
        if not cache_root.is_dir():
            return
        for path in cache_root.rglob("*.npz"):
            if path.stem not in referenced_keys:
                path.unlink(missing_ok=True)
        for directory in sorted(cache_root.glob("*")):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()

    def _reuse_source(
        self,
        active: dict[str, Any],
        upcoming_rows: list[dict[str, Any]],
        upcoming_image_ids: set[str],
        run_id: str,
    ) -> dict[str, Any]:
        previous = self.store.artifact(active["id"])
        history = copy.deepcopy(previous["historical"])
        historical_types = {str(item["itemType"]) for item in history}
        retained_rows = [row for row in upcoming_rows if normalize_text(row["item_type"]) in historical_types]
        if not retained_rows:
            raise CatalogValidationError(
                "upcoming data has no item types represented by the active historical catalogue"
            )
        upcoming, unseen = build_upcoming_features(retained_rows, historical_types)
        for item in history:
            source_id = str(item["sourceId"])
            item["imageUrl"] = f"/api/builds/{run_id}/images/historical/{source_id}" if item.get("imageUrl") else None
            item["hasVisualFeature"] = False
            item.pop("salesTarget", None)
            item.pop("qualityFlags", None)
            item.pop("weeklyLogRate", None)
            item.pop("demandCensored", None)
        for item in upcoming:
            product_id = str(item["id"])
            item["imageUrl"] = (
                f"/api/builds/{run_id}/images/upcoming/{product_id}"
                if product_id.upper() in upcoming_image_ids
                else None
            )
            item["hasVisualFeature"] = False
        return {
            "meta": {
                "title": "Turtle Season Intelligence AI",
                "dataMode": "uploaded",
                "historicalVersionId": active["historical_version_id"],
                "historicalReused": True,
                "upcomingSeason": normalize_text(retained_rows[0]["season"]),
                "historicalItems": len(history),
                "upcomingItems": len(upcoming),
                "historicalImageCoverage": sum(bool(item["imageUrl"]) for item in history),
                "upcomingImageCoverage": sum(bool(item["imageUrl"]) for item in upcoming),
                "missingUpcomingImages": [item["id"] for item in upcoming if not item["imageUrl"]],
                "dataQuality": {"upcomingWithoutHistoricalItem": unseen},
            },
            "historical": history,
            "upcoming": upcoming,
        }

    def process(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        # Tests and recovery tools may invoke the processor directly instead
        # of going through submit(). Keep timing complete for those paths too.
        if not run["analysis_started_at"]:
            self.store.update_run(run_id, analysis_started_at=ist_now(), completed_at=None)
        staging = self.store.staging_path(run_id)
        issues: list[ValidationIssue] = []
        try:
            self._check_cancelled(run_id)
            self.store.update_run(
                run_id, status="processing", stage="validating", progress=0.05, message="Validating CSVs and images"
            )
            upcoming = validate_catalog(
                self._single_catalogue(staging, "upcoming"),
                staging / "upcoming" / "images",
                "upcoming",
            )
            issues.extend(upcoming.report_records)
            active = self.store.active_build()
            replace_historical = run["mode"] == "full_replace"
            if replace_historical:
                historical = validate_catalog(
                    self._single_catalogue(staging, "historical"),
                    staging / "historical" / "images",
                    "historical",
                )
                issues.extend(historical.report_records)
                source = build_feature_source(
                    historical.rows,
                    upcoming.rows,
                    historical_image_ids=set(historical.image_index),
                    upcoming_image_ids=set(upcoming.image_index),
                    historical_version_id=run_id,
                    build_id=run_id,
                )
                historical_version_id = run_id
                historical_staging = staging / "historical"
                historical_features = historical_staging / "features.npz"
                historical_image_root = historical_staging / "images"
            else:
                if active is None:
                    raise CatalogValidationError("no active historical version is available")
                source = self._reuse_source(active, upcoming.rows, set(upcoming.image_index), run_id)
                historical_version_id = str(active["historical_version_id"])
                historical_staging = Path(active["historical_path"])
                historical_features = historical_staging / "features.npz"
                historical_image_root = historical_staging / "images"
                if self.vision_enabled and not historical_features.is_file():
                    raise CatalogValidationError("active historical version has no reusable visual feature index")
            self._write_issues(run_id, issues)
            self._check_cancelled(run_id)
            self.store.update_run(
                run_id,
                stage="indexing_history" if replace_historical else "matching_upcoming",
                progress=0.15,
                message="Building visual features" if replace_historical else "Reusing historical features",
            )
            vision = self._vision(
                source,
                paths=VisionPaths(historical_image_root, staging / "upcoming" / "images"),
                historical_features_path=historical_features,
                reuse_historical=not replace_historical,
                run_id=run_id,
            )
            self._check_cancelled(run_id)
            self.store.update_run(run_id, stage="building_results", progress=0.9, message="Building recommendations")
            artifact = build_model_artifact(source, vision)
            build_staging = staging / "build"
            build_staging.mkdir(parents=True, exist_ok=True)
            staged_artifact = build_staging / "artifact.json"
            staged_artifact.write_text(
                json.dumps(artifact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
            )
            staged_report = staging / "validation-report.csv"
            if staged_report.is_file():
                shutil.copy2(staged_report, build_staging / "validation-report.csv")

            upcoming_path = self.store.upcoming_path(run_id)
            build_path = self.store.build_path(run_id)
            upcoming_path.parent.mkdir(parents=True, exist_ok=True)
            build_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging / "upcoming"), upcoming_path)
            shutil.move(str(build_staging), build_path)
            artifact_path = build_path / "artifact.json"
            activated_report = build_path / "validation-report.csv"
            if activated_report.is_file():
                self.store.update_run(run_id, issues_path=str(activated_report))
            if replace_historical:
                historical_path = self.store.historical_path(run_id)
                historical_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staging / "historical"), historical_path)
            else:
                historical_path = historical_staging
            cleanup = self.store.activate(
                run_id=run_id,
                historical_version_id=historical_version_id,
                historical_path=historical_path,
                historical_count=len(artifact["historical"]),
                historical_coverage=int(artifact["meta"]["historicalImageCoverage"]),
                model_version=str(artifact["meta"]["model"]["version"]),
                upcoming_version_id=run_id,
                upcoming_path=upcoming_path,
                upcoming_count=len(artifact["upcoming"]),
                upcoming_coverage=int(artifact["meta"]["upcomingImageCoverage"]),
                artifact_path=artifact_path,
                replace_historical=replace_historical,
            )
            self.store.cleanup_paths(cleanup)
            self.store.cleanup_paths(self.store.purge_superseded_run_history())
            if self.vision_enabled:
                self._prune_feature_cache(set(vision.get("featureCacheKeys", [])))
            shutil.rmtree(staging, ignore_errors=True)
        except InterruptedError:
            return
        except Exception as exc:
            if isinstance(exc, CatalogValidationError):
                issues.extend(exc.issues)
            if issues:
                self._write_issues(run_id, issues)
            self.store.update_run(
                run_id,
                status="failed",
                stage="failed",
                error=f"{type(exc).__name__}: {exc}",
                eta_seconds=0,
                completed_at=ist_now(),
                message="The previous active build is unchanged",
            )
