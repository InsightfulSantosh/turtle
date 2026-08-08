"""Generate two-stage visual-retrieval input for the browser decision artifact."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_pipeline.images import build_image_index
from fashion_matching.appearance import (
    COLOUR_DELTA_E_SCALE,
    COLOUR_DESCRIPTOR_DIMENSION,
    COLOUR_PALETTE_SIZE,
    OTTR_TROUSER_ROI,
    TEXTURE_DESCRIPTOR_DIMENSION,
    FashionCandidateRetriever,
    body_pattern_views,
    canonical_retrieval_value,
    cosine_distance,
    dominant_palette_distance,
    extract_pipeline_appearance_features,
    requires_ottr_pattern_gate,
    requires_pattern_gate,
)
from fashion_matching.encoders import FashionEncoder, ImageEncoder
from fashion_matching.preprocessing import ImagePreprocessor, ImageValidationError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisionPaths:
    """The two image directories a build encodes, historical first."""

    historical_image_root: Path
    upcoming_image_root: Path


@dataclass(frozen=True)
class CatalogueVisualEmbeddings:
    identifiers: list[str]
    fashion: np.ndarray
    detail: np.ndarray | None
    pattern: np.ndarray | None
    colour: np.ndarray
    texture: np.ndarray
    colour_available: list[bool]
    texture_available: list[bool]
    cache_keys: list[str]


def save_catalogue_embeddings(
    path: Path,
    embeddings: CatalogueVisualEmbeddings,
    *,
    model_id: str,
    model_revision: str,
    reranker_model_id: str | None,
    reranker_model_revision: str | None,
    preprocessing_version: str,
) -> None:
    """Persist validated historical features for upcoming-only rebuilds."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "modelId": model_id,
        "modelRevision": model_revision,
        "rerankerModelId": reranker_model_id,
        "rerankerModelRevision": reranker_model_revision,
        "preprocessingVersion": preprocessing_version,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            identifiers=np.asarray(embeddings.identifiers, dtype=np.str_),
            fashion=embeddings.fashion,
            detail=embeddings.detail if embeddings.detail is not None else np.empty((0, 0), dtype=np.float32),
            pattern=embeddings.pattern if embeddings.pattern is not None else np.empty((0, 0), dtype=np.float32),
            colour=embeddings.colour,
            texture=embeddings.texture,
            colour_available=np.asarray(embeddings.colour_available, dtype=np.bool_),
            texture_available=np.asarray(embeddings.texture_available, dtype=np.bool_),
            cache_keys=np.asarray(embeddings.cache_keys, dtype=np.str_),
            metadata=np.asarray([json.dumps(metadata, separators=(",", ":"))], dtype=np.str_),
        )
    temporary.replace(path)


def load_catalogue_embeddings(
    path: Path,
    *,
    model_id: str,
    model_revision: str,
    reranker_model_id: str | None,
    reranker_model_revision: str | None,
    preprocessing_version: str,
) -> CatalogueVisualEmbeddings:
    """Load historical features only when model and preprocessing keys match."""

    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"][0]))
        expected = {
            "modelId": model_id,
            "modelRevision": model_revision,
            "rerankerModelId": reranker_model_id,
            "rerankerModelRevision": reranker_model_revision,
            "preprocessingVersion": preprocessing_version,
        }
        if metadata != expected:
            raise ValueError("stored historical features use a different model or preprocessing version")
        detail = np.asarray(stored["detail"], dtype=np.float32)
        pattern = np.asarray(stored["pattern"], dtype=np.float32)
        return CatalogueVisualEmbeddings(
            identifiers=[str(value) for value in stored["identifiers"].tolist()],
            fashion=np.asarray(stored["fashion"], dtype=np.float32),
            detail=detail if detail.size else None,
            pattern=pattern if pattern.size else None,
            colour=np.asarray(stored["colour"], dtype=np.float32),
            texture=np.asarray(stored["texture"], dtype=np.float32),
            colour_available=[bool(value) for value in stored["colour_available"].tolist()],
            texture_available=[bool(value) for value in stored["texture_available"].tolist()],
            cache_keys=[str(value) for value in stored["cache_keys"].tolist()],
        )


def _encode_catalog(
    items: list[dict[str, Any]],
    *,
    image_root: Path,
    identifier_field: str,
    encoder: FashionEncoder,
    reranker: ImageEncoder | None,
    preprocessor: ImagePreprocessor,
    batch_size: int,
    feature_cache_root: Path | None = None,
    catalog_name: str = "catalog",
    progress_callback: Callable[[dict[str, int | float | str]], None] | None = None,
) -> CatalogueVisualEmbeddings:
    """Encode each usable catalogue image once for every enabled visual stage."""

    image_index = build_image_index(image_root)
    path_to_items: dict[Path, list[dict[str, Any]]] = {}
    for item in items:
        identifier = str(item.get(identifier_field, "")).upper()
        image_path = image_index.get(identifier)
        if item.get("imageUrl") and image_path is not None:
            path_to_items.setdefault(image_path, []).append(item)

    paths = list(path_to_items)
    total_paths = len(paths)
    started_at = time.monotonic()
    encoded_count = 0
    cached_count = 0
    skipped_count = 0

    def report_progress(processed_count: int) -> None:
        elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
        processing_rate = processed_count / elapsed_seconds
        eta_seconds = (total_paths - processed_count) / processing_rate if processing_rate else 0.0
        LOGGER.info(
            (
                "Vision progress catalog=%s processed=%s total=%s percent=%.1f%% "
                "encoded=%s cached=%s skipped=%s elapsed=%.1fmin eta=%.1fmin"
            ),
            catalog_name,
            processed_count,
            total_paths,
            (processed_count / total_paths * 100.0) if total_paths else 100.0,
            encoded_count,
            cached_count,
            skipped_count,
            elapsed_seconds / 60.0,
            eta_seconds / 60.0,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "catalog": catalog_name,
                    "processed": processed_count,
                    "total": total_paths,
                    "encoded": encoded_count,
                    "cached": cached_count,
                    "skipped": skipped_count,
                    "rate": processing_rate,
                    "etaSeconds": eta_seconds,
                }
            )
    LOGGER.info(
        "Vision encoding started catalog=%s total=%s batch_size=%s",
        catalog_name,
        total_paths,
        batch_size,
    )
    fashion_vectors_by_path: dict[Path, np.ndarray] = {}
    detail_vectors_by_path: dict[Path, np.ndarray] = {}
    pattern_vectors_by_path: dict[Path, np.ndarray] = {}
    colour_vectors_by_path: dict[Path, np.ndarray] = {}
    texture_vectors_by_path: dict[Path, np.ndarray] = {}
    cache_keys_by_path: dict[Path, str] = {}
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        prepared_paths: list[Path] = []
        prepared_images = []
        cache_paths: list[Path | None] = []
        for image_path in batch_paths:
            try:
                prepared = preprocessor.prepare(image_path)
            except ImageValidationError as exc:
                skipped_count += 1
                LOGGER.warning("Skipping invalid product image %s: %s", image_path, exc)
                continue
            item_types = {
                canonical_retrieval_value(item.get("itemType"))
                for item in path_to_items[image_path]
            }
            item_types.discard("")
            if len(item_types) > 1:
                raise ValueError(
                    f"Image {image_path} is mapped to multiple item types: {sorted(item_types)}"
                )
            item_type = next(iter(item_types), "")
            cache_path: Path | None = None
            if feature_cache_root is not None:
                cache_key = hashlib.sha256(
                    "\n".join(
                        (
                            prepared.checksum,
                            item_type,
                            encoder.model_id,
                            encoder.revision,
                            reranker.model_id if reranker is not None else "",
                            reranker.revision if reranker is not None else "",
                            preprocessor.version,
                        )
                    ).encode("utf-8")
                ).hexdigest()
                cache_path = feature_cache_root / cache_key[:2] / f"{cache_key}.npz"
                cache_keys_by_path[image_path] = cache_key
                if cache_path.is_file():
                    try:
                        with np.load(cache_path, allow_pickle=False) as cached:
                            fashion_vectors_by_path[image_path] = np.asarray(cached["fashion"], dtype=np.float32)
                            colour_vectors_by_path[image_path] = np.asarray(cached["colour"], dtype=np.float32)
                            texture_vectors_by_path[image_path] = np.asarray(cached["texture"], dtype=np.float32)
                            if reranker is not None:
                                detail_vectors_by_path[image_path] = np.asarray(cached["detail"], dtype=np.float32)
                                pattern_vectors_by_path[image_path] = np.asarray(cached["pattern"], dtype=np.float32)
                        cached_count += 1
                        prepared.image.close()
                        continue
                    except (OSError, KeyError, ValueError):
                        cache_path.unlink(missing_ok=True)
            appearance = extract_pipeline_appearance_features(
                prepared.image,
                item_type=item_type,
            )
            prepared_paths.append(image_path)
            prepared_images.append(appearance.image)
            cache_paths.append(cache_path)
            colour_vectors_by_path[image_path] = appearance.colour_vector
            texture_vectors_by_path[image_path] = appearance.texture_vector

        processed_count = min(offset + len(batch_paths), total_paths)
        if not prepared_images:
            report_progress(processed_count)
            continue
        try:
            fashion_vectors = encoder.encode_images(prepared_images)
            detail_vectors = reranker.encode_images(prepared_images) if reranker else None
            pattern_images = (
                [
                    view
                    # ``prepared_images`` already contains the item-type
                    # analysis image. Do not apply the OTTR ROI a second time.
                    for image in prepared_images
                    for view in body_pattern_views(image)
                ]
                if reranker
                else []
            )
            try:
                pattern_vectors = reranker.encode_images(pattern_images) if reranker else None
            finally:
                for image in pattern_images:
                    image.close()
        finally:
            for image in prepared_images:
                image.close()
        if len(fashion_vectors) != len(prepared_paths):
            raise RuntimeError("Fashion candidate encoder returned a different number of vectors")
        if detail_vectors is not None and len(detail_vectors) != len(prepared_paths):
            raise RuntimeError("DINOv2 reranker returned a different number of vectors")
        if pattern_vectors is not None and len(pattern_vectors) != len(prepared_paths) * 3:
            raise RuntimeError("DINOv2 body-pattern encoder returned a different number of vectors")
        for index, image_path in enumerate(prepared_paths):
            fashion_vectors_by_path[image_path] = np.asarray(fashion_vectors[index], dtype=np.float32)
            if detail_vectors is not None:
                detail_vectors_by_path[image_path] = np.asarray(detail_vectors[index], dtype=np.float32)
                combined = np.concatenate(pattern_vectors[index * 3 : (index + 1) * 3]).astype(np.float32)
                pattern_vectors_by_path[image_path] = combined / max(float(np.linalg.norm(combined)), 1e-12)
            cache_path = cache_paths[index]
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        fashion=fashion_vectors_by_path[image_path],
                        detail=(
                            detail_vectors_by_path[image_path]
                            if reranker is not None
                            else np.empty((0,), dtype=np.float32)
                        ),
                        pattern=(
                            pattern_vectors_by_path[image_path]
                            if reranker is not None
                            else np.empty((0,), dtype=np.float32)
                        ),
                        colour=colour_vectors_by_path[image_path],
                        texture=texture_vectors_by_path[image_path],
                    )
                temporary.replace(cache_path)
        encoded_count += len(prepared_paths)
        report_progress(processed_count)

    LOGGER.info(
        "Vision encoding completed catalog=%s total=%s encoded=%s cached=%s skipped=%s elapsed=%.1fmin",
        catalog_name,
        total_paths,
        encoded_count,
        cached_count,
        skipped_count,
        (time.monotonic() - started_at) / 60.0,
    )

    identifiers: list[str] = []
    fashion_vectors: list[np.ndarray] = []
    detail_vectors: list[np.ndarray] = []
    pattern_vectors: list[np.ndarray] = []
    colour_vectors: list[np.ndarray] = []
    texture_vectors: list[np.ndarray] = []
    colour_available: list[bool] = []
    texture_available: list[bool] = []
    cache_keys: list[str] = []
    for image_path, catalog_items in path_to_items.items():
        fashion_vector = fashion_vectors_by_path.get(image_path)
        detail_vector = detail_vectors_by_path.get(image_path) if reranker else None
        pattern_vector = pattern_vectors_by_path.get(image_path) if reranker else None
        colour_vector = colour_vectors_by_path.get(image_path)
        texture_vector = texture_vectors_by_path.get(image_path)
        missing_reranker_vector = reranker is not None and (detail_vector is None or pattern_vector is None)
        if fashion_vector is None or colour_vector is None or texture_vector is None or missing_reranker_vector:
            continue
        for item in catalog_items:
            item["hasVisualFeature"] = True
            identifiers.append(str(item["id"]))
            fashion_vectors.append(fashion_vector)
            colour_available.append(True)
            texture_available.append(True)
            cache_keys.append(cache_keys_by_path.get(image_path, ""))
            colour_vectors.append(colour_vector)
            texture_vectors.append(texture_vector)
            if detail_vector is not None:
                detail_vectors.append(detail_vector)
                pattern_vectors.append(pattern_vector)
    if not fashion_vectors:
        empty_fashion = np.empty((0, int(encoder.dimension)), dtype=np.float32)
        empty_detail = np.empty((0, int(reranker.dimension)), dtype=np.float32) if reranker is not None else None
        return CatalogueVisualEmbeddings(
            identifiers=identifiers,
            fashion=empty_fashion,
            detail=empty_detail,
            pattern=(
                np.empty((0, int(reranker.dimension) * 3), dtype=np.float32)
                if reranker is not None
                else None
            ),
            colour=np.empty((0, COLOUR_DESCRIPTOR_DIMENSION), dtype=np.float32),
            texture=np.empty((0, TEXTURE_DESCRIPTOR_DIMENSION), dtype=np.float32),
            colour_available=[],
            texture_available=[],
            cache_keys=[],
        )
    return CatalogueVisualEmbeddings(
        identifiers=identifiers,
        fashion=np.stack(fashion_vectors),
        detail=np.stack(detail_vectors) if reranker is not None else None,
        pattern=np.stack(pattern_vectors) if reranker is not None else None,
        colour=np.stack(colour_vectors),
        texture=np.stack(texture_vectors),
        colour_available=colour_available,
        texture_available=texture_available,
        cache_keys=cache_keys,
    )


def _distance_rows(
    left_ids: list[str],
    right_ids: list[str],
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
) -> list[dict[str, str | float]]:
    if not left_ids or not right_ids:
        return []
    distances = np.clip(1.0 - left_vectors @ right_vectors.T, 0.0, 2.0)
    return [
        {
            "leftId": left_id,
            "rightId": right_id,
            "distance": round(float(distances[left_index, right_index]), 7),
        }
        for left_index, left_id in enumerate(left_ids)
        for right_index, right_id in enumerate(right_ids)
    ]


def _two_stage_candidate_rows(
    left_items: list[dict[str, Any]],
    right_items: list[dict[str, Any]],
    left_ids: list[str],
    right_ids: list[str],
    left_fashion_vectors: np.ndarray,
    right_fashion_vectors: np.ndarray,
    left_detail_vectors: np.ndarray,
    right_detail_vectors: np.ndarray,
    left_pattern_vectors: np.ndarray,
    right_pattern_vectors: np.ndarray,
    left_colour_vectors: np.ndarray,
    right_colour_vectors: np.ndarray,
    left_texture_vectors: np.ndarray,
    right_texture_vectors: np.ndarray,
    left_colour_available: list[bool],
    right_colour_available: list[bool],
    left_texture_available: list[bool],
    right_texture_available: list[bool],
    *,
    candidate_count: int,
    require_same_item_type: bool,
    require_same_design: bool,
    visual_only_ranking: bool,
    pattern_gate_enabled: bool,
    pattern_max_distance: float,
    colour_gate_enabled: bool,
    colour_max_distance: float,
) -> tuple[list[dict[str, str | float | int]], str, dict[str, Any]]:
    """Use FAISS (when available) to retrieve FashionSigLIP candidates once."""

    rows: list[dict[str, str | float | int]] = []
    retriever = FashionCandidateRetriever(right_items, right_fashion_vectors)
    audit: dict[str, Any] = {
        "queries": len(left_items),
        "retrievedCandidates": 0,
        "colourRejected": 0,
        "patternGateApplied": 0,
        "patternGateBypassed": 0,
        "patternRejected": 0,
        "acceptedCandidates": 0,
        "queriesWithNoAcceptedCandidate": 0,
        "byItemType": {},
    }
    for left_index, left_item in enumerate(left_items):
        item_type = str(left_item.get("itemType") or "UNSPECIFIED")
        item_audit = audit["byItemType"].setdefault(
            item_type,
            {
                "queries": 0,
                "retrievedCandidates": 0,
                "colourRejected": 0,
                "patternGateApplied": 0,
                "patternGateBypassed": 0,
                "patternRejected": 0,
                "acceptedCandidates": 0,
                "queriesWithNoAcceptedCandidate": 0,
            },
        )
        item_audit["queries"] += 1
        candidate_indices = retriever.search(
            left_item,
            left_fashion_vectors[left_index],
            candidate_count,
            require_same_item_type=require_same_item_type,
            require_same_design=require_same_design,
        )
        audit["retrievedCandidates"] += len(candidate_indices)
        item_audit["retrievedCandidates"] += len(candidate_indices)
        accepted_rank = 0
        for retrieval_rank, right_index in enumerate(candidate_indices, start=1):
            fashion_distance = float(
                np.clip(
                    1.0 - left_fashion_vectors[left_index] @ right_fashion_vectors[right_index],
                    0.0,
                    2.0,
                )
            )
            global_dino_distance = float(
                np.clip(
                    1.0 - left_detail_vectors[left_index] @ right_detail_vectors[right_index],
                    0.0,
                    2.0,
                )
            )
            pattern_distance = float(
                np.clip(
                    1.0 - left_pattern_vectors[left_index] @ right_pattern_vectors[right_index],
                    0.0,
                    2.0,
                )
            )
            colour_distance = (
                dominant_palette_distance(
                    left_colour_vectors[left_index],
                    right_colour_vectors[right_index],
                )
                if left_colour_available[left_index] and right_colour_available[right_index]
                else None
            )
            if colour_gate_enabled and (
                colour_distance is None or colour_distance > colour_max_distance
            ):
                audit["colourRejected"] += 1
                item_audit["colourRejected"] += 1
                continue
            is_ottr_query = canonical_retrieval_value(left_item.get("itemType")) == "OTTR"
            pattern_gated = (
                requires_ottr_pattern_gate(left_item.get("design"))
                if is_ottr_query
                else visual_only_ranking
                or (
                    requires_pattern_gate(left_item.get("design"))
                    and requires_pattern_gate(right_items[right_index].get("design"))
                )
            )
            audit["patternGateApplied" if pattern_gated else "patternGateBypassed"] += 1
            item_audit["patternGateApplied" if pattern_gated else "patternGateBypassed"] += 1
            if pattern_gate_enabled and pattern_gated and pattern_distance > pattern_max_distance:
                audit["patternRejected"] += 1
                item_audit["patternRejected"] += 1
                continue
            texture_distance = (
                cosine_distance(
                    left_texture_vectors[left_index],
                    right_texture_vectors[right_index],
                )
                if left_texture_available[left_index] and right_texture_available[right_index]
                else None
            )
            accepted_rank += 1
            audit["acceptedCandidates"] += 1
            item_audit["acceptedCandidates"] += 1
            rows.append(
                {
                    "leftId": left_ids[left_index],
                    "rightId": right_ids[right_index],
                    "fashionDistance": round(fashion_distance, 7),
                    # This is the DINO signal used for scoring: a three-scale
                    # body-only pattern/structure representation.  Retain the
                    # old whole-image DINO distance solely for audit analysis.
                    "dinoDistance": round(pattern_distance, 7),
                    "globalDinoDistance": round(global_dino_distance, 7),
                    "patternDistance": round(pattern_distance, 7),
                    "colourDistance": round(colour_distance, 7) if colour_distance is not None else None,
                    "colourDeltaE": (
                        round(colour_distance * COLOUR_DELTA_E_SCALE, 3)
                        if colour_distance is not None
                        else None
                    ),
                    "textureDistance": round(texture_distance, 7) if texture_distance is not None else None,
                    "candidateRank": accepted_rank,
                    "retrievalRank": retrieval_rank,
                }
            )
        if accepted_rank == 0:
            audit["queriesWithNoAcceptedCandidate"] += 1
            item_audit["queriesWithNoAcceptedCandidate"] += 1
    audit["byItemType"] = {
        key: audit["byItemType"][key] for key in sorted(audit["byItemType"])
    }
    return rows, retriever.backend, audit


def build_artifact_vision_output(
    source: dict[str, Any],
    *,
    settings: VisionPaths,
    encoder: FashionEncoder,
    preprocessor: ImagePreprocessor,
    reranker: ImageEncoder | None = None,
    candidate_count: int = 50,
    reranker_weight_grid: tuple[float, ...] = (0.5833333333,),
    require_same_item_type: bool = True,
    require_same_design: bool = False,
    visual_only_ranking: bool = True,
    pattern_gate_enabled: bool = True,
    pattern_max_distance: float = 0.42,
    colour_gate_enabled: bool = True,
    colour_max_distance: float = 0.20,
    appearance_weights: dict[str, float] | None = None,
    batch_size: int = 16,
    historical_embeddings: CatalogueVisualEmbeddings | None = None,
    historical_embeddings_callback: Callable[[CatalogueVisualEmbeddings], None] | None = None,
    feature_cache_root: Path | None = None,
    progress_callback: Callable[[dict[str, int | float | str]], None] | None = None,
) -> dict[str, Any]:
    """Build either legacy visual distances or two-stage retrieval candidates."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not reranker_weight_grid or any(not 0 <= weight <= 1 for weight in reranker_weight_grid):
        raise ValueError("reranker_weight_grid values must be between 0 and 1")
    if not 0.0 <= pattern_max_distance <= 2.0:
        raise ValueError("pattern_max_distance must be between 0 and 2")
    if not 0.0 <= colour_max_distance <= 2.0:
        raise ValueError("colour_max_distance must be between 0 and 2")
    resolved_appearance_weights = {
        "neural": 0.45,
        "colour": 0.45,
        "texture": 0.10,
        **(appearance_weights or {}),
    }
    if set(resolved_appearance_weights) != {"neural", "colour", "texture"}:
        raise ValueError("appearance_weights must contain neural, colour and texture values")
    if any(weight < 0 for weight in resolved_appearance_weights.values()) or not np.isclose(
        sum(resolved_appearance_weights.values()),
        1.0,
    ):
        raise ValueError("appearance_weights must be non-negative and sum to 1")
    historical = historical_embeddings
    if historical is None:
        historical = _encode_catalog(
            source["historical"],
            image_root=settings.historical_image_root,
            identifier_field="sourceId",
            encoder=encoder,
            reranker=reranker,
            preprocessor=preprocessor,
            batch_size=batch_size,
            feature_cache_root=feature_cache_root,
            catalog_name="historical",
            progress_callback=progress_callback,
        )
        if historical_embeddings_callback is not None:
            historical_embeddings_callback(historical)
    else:
        reusable_ids = set(historical.identifiers)
        for item in source["historical"]:
            item["hasVisualFeature"] = str(item["id"]) in reusable_ids
    upcoming = _encode_catalog(
        source["upcoming"],
        image_root=settings.upcoming_image_root,
        identifier_field="id",
        encoder=encoder,
        reranker=reranker,
        preprocessor=preprocessor,
        batch_size=batch_size,
        feature_cache_root=feature_cache_root,
        catalog_name="upcoming",
        progress_callback=progress_callback,
    )
    feature_cache_keys = sorted({key for key in (*historical.cache_keys, *upcoming.cache_keys) if key})
    if reranker is None:
        distances = _distance_rows(
            historical.identifiers,
            historical.identifiers,
            historical.fashion,
            historical.fashion,
        )
        distances.extend(
            _distance_rows(
                upcoming.identifiers,
                historical.identifiers,
                upcoming.fashion,
                historical.fashion,
            )
        )
        return {
            "engine": "Fashion-domain SigLIP image embeddings with cosine distance",
            "modelId": encoder.model_id,
            "modelRevision": encoder.revision,
            "embeddingDimension": encoder.dimension,
            "device": str(getattr(encoder, "device", "unknown")),
            "historicalCoverage": len(historical.identifiers),
            "upcomingCoverage": len(upcoming.identifiers),
            "calibrationMethod": (
                "Separate robust logistic calibration for historical validation "
                "and upcoming-to-historical serving distances"
            ),
            "distances": distances,
            "featureCacheKeys": feature_cache_keys,
        }

    assert historical.detail is not None
    assert upcoming.detail is not None
    assert historical.pattern is not None
    assert upcoming.pattern is not None
    historical_by_id = {str(item["id"]): item for item in source["historical"]}
    upcoming_by_id = {str(item["id"]): item for item in source["upcoming"]}
    historical_items = [historical_by_id[identifier] for identifier in historical.identifiers]
    upcoming_items = [upcoming_by_id[identifier] for identifier in upcoming.identifiers]
    historical_pairs, historical_index_backend, historical_gate_audit = _two_stage_candidate_rows(
        historical_items,
        historical_items,
        historical.identifiers,
        historical.identifiers,
        historical.fashion,
        historical.fashion,
        historical.detail,
        historical.detail,
        historical.pattern,
        historical.pattern,
        historical.colour,
        historical.colour,
        historical.texture,
        historical.texture,
        historical.colour_available,
        historical.colour_available,
        historical.texture_available,
        historical.texture_available,
        candidate_count=candidate_count,
        require_same_item_type=require_same_item_type,
        require_same_design=require_same_design,
        visual_only_ranking=visual_only_ranking,
        pattern_gate_enabled=pattern_gate_enabled,
        pattern_max_distance=pattern_max_distance,
        colour_gate_enabled=colour_gate_enabled,
        colour_max_distance=colour_max_distance,
    )
    upcoming_pairs, upcoming_index_backend, upcoming_gate_audit = _two_stage_candidate_rows(
        upcoming_items,
        historical_items,
        upcoming.identifiers,
        historical.identifiers,
        upcoming.fashion,
        historical.fashion,
        upcoming.detail,
        historical.detail,
        upcoming.pattern,
        historical.pattern,
        upcoming.colour,
        historical.colour,
        upcoming.texture,
        historical.texture,
        upcoming.colour_available,
        historical.colour_available,
        upcoming.texture_available,
        historical.texture_available,
        candidate_count=candidate_count,
        require_same_item_type=require_same_item_type,
        require_same_design=require_same_design,
        visual_only_ranking=visual_only_ranking,
        pattern_gate_enabled=pattern_gate_enabled,
        pattern_max_distance=pattern_max_distance,
        colour_gate_enabled=colour_gate_enabled,
        colour_max_distance=colour_max_distance,
    )
    candidate_pairs = historical_pairs + upcoming_pairs
    return {
        "engine": (
            "Visual-only FashionSigLIP retrieval with multi-scale DINO, dominant-palette "
            "CIEDE2000 colour gating and texture reranking"
        ),
        "modelId": encoder.model_id,
        "modelRevision": encoder.revision,
        "embeddingDimension": encoder.dimension,
        "device": str(getattr(encoder, "device", "unknown")),
        "historicalCoverage": len(historical.identifiers),
        "upcomingCoverage": len(upcoming.identifiers),
        "visualOnlyRanking": visual_only_ranking,
        "reranker": {
            "modelId": reranker.model_id,
            "modelRevision": reranker.revision,
            "embeddingDimension": reranker.dimension,
            "device": str(getattr(reranker, "device", "unknown")),
            "candidateCount": candidate_count,
            "weightGrid": list(reranker_weight_grid),
            "sameItemTypeConstraint": require_same_item_type,
            "sameDesignConstraint": require_same_design,
            "visualOnlyRanking": visual_only_ranking,
            "patternGate": {
                "enabled": pattern_gate_enabled,
                "method": "three-scale-centre-garment-body-DINOv2",
                "scope": (
                    "all visually retrieved candidates except OTTR; OTTR hard-gates only "
                    "CHECKS, PRINTS, STRIPES and DOBBY/STRUCTURE"
                    if visual_only_ranking
                    else "label-selected patterns"
                ),
                "maximumDistance": pattern_max_distance,
                "policy": (
                    "exclude candidate when the body-pattern distance exceeds the limit"
                    if pattern_gate_enabled
                    else "disabled; pattern distance contributes to visual reranking without excluding candidates"
                ),
            },
            "gateAudit": {
                "historical": historical_gate_audit,
                "upcoming": upcoming_gate_audit,
            },
            "candidateIndex": {
                "engine": (
                    historical_index_backend
                    if historical_index_backend == upcoming_index_backend
                    else "mixed-faiss-and-numpy-exact"
                ),
                "metric": "inner-product on L2-normalized FashionSigLIP embeddings",
                "fallback": "numpy-exact-inner-product",
            },
            "appearance": {
                "colourDescriptor": {
                    "space": "CIELAB",
                    "method": "dominant-palette-CIEDE2000",
                    "paletteSize": COLOUR_PALETTE_SIZE,
                    "distance": "symmetric weighted palette ΔE plus dominant-colour ΔE",
                    "normalisationScaleDeltaE": COLOUR_DELTA_E_SCALE,
                    "region": "central garment body with studio-background suppression",
                    "fullImage": False,
                },
                "itemTypeOverrides": {
                    "OTTR": {
                        "analysisRegion": "waist-to-lower-leg trouser ROI excluding footwear",
                        "relativeBox": list(OTTR_TROUSER_ROI),
                        "usedFor": [
                            "FashionSigLIP retrieval",
                            "global DINO detail",
                            "multi-scale DINO pattern detail",
                            "dominant colour palette",
                            "texture descriptor",
                        ],
                        "displayedImageModified": False,
                        "patternHardGateDesigns": [
                            "CHECKS",
                            "PRINTS",
                            "STRIPES",
                            "DOBBY/STRUCTURE",
                        ],
                    }
                },
                "colourGate": {
                    "enabled": colour_gate_enabled,
                    "maximumDistance": colour_max_distance,
                    "maximumDeltaE": round(colour_max_distance * COLOUR_DELTA_E_SCALE, 3),
                    "policy": (
                        "exclude candidate when garment-palette perceptual ΔE exceeds the limit"
                        if colour_gate_enabled
                        else "disabled"
                    ),
                },
                "textureDescriptor": {
                    "method": "full-image-gradient-orientation-and-local-energy",
                    "dimension": TEXTURE_DESCRIPTOR_DIMENSION,
                },
                "weights": {key: round(value, 4) for key, value in resolved_appearance_weights.items()},
            },
        },
        "calibrationMethod": (
            "Separate robust logistic calibration for FashionSigLIP, multi-scale DINOv2, dominant-palette "
            "CIEDE2000 colour and texture candidate distances; component weights are configured explicitly"
        ),
        "candidatePairs": candidate_pairs,
        "featureCacheKeys": feature_cache_keys,
    }
