"""Generate two-stage visual-retrieval input for the browser decision artifact."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_pipeline.images import build_image_index
from data_pipeline.settings import PipelineSettings
from fashion_matching.appearance import (
    COLOUR_BINS_PER_CHANNEL,
    COLOUR_DESCRIPTOR_DIMENSION,
    TEXTURE_DESCRIPTOR_DIMENSION,
    FashionCandidateRetriever,
    cosine_distance,
    extract_appearance_features,
)
from fashion_matching.encoders import FashionEncoder, ImageEncoder
from fashion_matching.models import ManifestRecord
from fashion_matching.preprocessing import ImagePreprocessor, ImageValidationError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogueVisualEmbeddings:
    identifiers: list[str]
    fashion: np.ndarray
    detail: np.ndarray | None
    colour: np.ndarray
    texture: np.ndarray
    mask_coverages: list[float]
    mask_confidences: list[float]
    segmentation_methods: list[str]
    fallback_reasons: list[str | None]


def _encode_catalog(
    items: list[dict[str, Any]],
    *,
    image_root: Path,
    identifier_field: str,
    encoder: FashionEncoder,
    reranker: ImageEncoder | None,
    preprocessor: ImagePreprocessor,
    appearance_mask_enabled: bool,
    batch_size: int,
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
    fashion_vectors_by_path: dict[Path, np.ndarray] = {}
    detail_vectors_by_path: dict[Path, np.ndarray] = {}
    colour_vectors_by_path: dict[Path, np.ndarray] = {}
    texture_vectors_by_path: dict[Path, np.ndarray] = {}
    mask_metadata_by_path: dict[Path, tuple[float, float, str, str | None]] = {}
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        prepared_paths: list[Path] = []
        prepared_images = []
        source_images = []
        for image_path in batch_paths:
            try:
                prepared = preprocessor.prepare(
                    ManifestRecord(
                        product_id=image_path.stem,
                        image_id=image_path.stem,
                        image_path=image_path,
                    )
                )
            except ImageValidationError as exc:
                LOGGER.warning("Skipping invalid product image %s: %s", image_path, exc)
                continue
            appearance = extract_appearance_features(
                prepared.image,
                mask_enabled=appearance_mask_enabled,
            )
            prepared_paths.append(image_path)
            prepared_images.append(appearance.image)
            source_images.append(prepared.image)
            colour_vectors_by_path[image_path] = appearance.colour_vector
            texture_vectors_by_path[image_path] = appearance.texture_vector
            mask_metadata_by_path[image_path] = (
                appearance.mask_coverage,
                appearance.mask_confidence,
                appearance.segmentation_method,
                appearance.fallback_reason,
            )

        if not prepared_images:
            continue
        try:
            fashion_vectors = encoder.encode_images(prepared_images)
            detail_vectors = reranker.encode_images(prepared_images) if reranker else None
        finally:
            for image in prepared_images:
                image.close()
            for image in source_images:
                image.close()
        if len(fashion_vectors) != len(prepared_paths):
            raise RuntimeError("Fashion candidate encoder returned a different number of vectors")
        if detail_vectors is not None and len(detail_vectors) != len(prepared_paths):
            raise RuntimeError("DINOv2 reranker returned a different number of vectors")
        for index, image_path in enumerate(prepared_paths):
            fashion_vectors_by_path[image_path] = np.asarray(fashion_vectors[index], dtype=np.float32)
            if detail_vectors is not None:
                detail_vectors_by_path[image_path] = np.asarray(detail_vectors[index], dtype=np.float32)
        LOGGER.info(
            "Encoded %s/%s unique images from %s",
            min(offset + batch_size, len(paths)),
            len(paths),
            image_root.name,
        )

    identifiers: list[str] = []
    fashion_vectors: list[np.ndarray] = []
    detail_vectors: list[np.ndarray] = []
    colour_vectors: list[np.ndarray] = []
    texture_vectors: list[np.ndarray] = []
    mask_coverages: list[float] = []
    mask_confidences: list[float] = []
    segmentation_methods: list[str] = []
    fallback_reasons: list[str | None] = []
    for image_path, catalog_items in path_to_items.items():
        fashion_vector = fashion_vectors_by_path.get(image_path)
        detail_vector = detail_vectors_by_path.get(image_path) if reranker else None
        colour_vector = colour_vectors_by_path.get(image_path)
        texture_vector = texture_vectors_by_path.get(image_path)
        metadata = mask_metadata_by_path.get(image_path)
        if (
            fashion_vector is None
            or colour_vector is None
            or texture_vector is None
            or metadata is None
            or (reranker is not None and detail_vector is None)
        ):
            continue
        for item in catalog_items:
            item["hasVisualFeature"] = True
            identifiers.append(str(item["id"]))
            fashion_vectors.append(fashion_vector)
            colour_vectors.append(colour_vector)
            texture_vectors.append(texture_vector)
            mask_coverages.append(metadata[0])
            mask_confidences.append(metadata[1])
            segmentation_methods.append(metadata[2])
            fallback_reasons.append(metadata[3])
            if detail_vector is not None:
                detail_vectors.append(detail_vector)
    if not fashion_vectors:
        empty_fashion = np.empty((0, int(encoder.dimension)), dtype=np.float32)
        empty_detail = np.empty((0, int(reranker.dimension)), dtype=np.float32) if reranker is not None else None
        return CatalogueVisualEmbeddings(
            identifiers=identifiers,
            fashion=empty_fashion,
            detail=empty_detail,
            colour=np.empty((0, COLOUR_DESCRIPTOR_DIMENSION), dtype=np.float32),
            texture=np.empty((0, TEXTURE_DESCRIPTOR_DIMENSION), dtype=np.float32),
            mask_coverages=[],
            mask_confidences=[],
            segmentation_methods=[],
            fallback_reasons=[],
        )
    return CatalogueVisualEmbeddings(
        identifiers=identifiers,
        fashion=np.stack(fashion_vectors),
        detail=np.stack(detail_vectors) if reranker is not None else None,
        colour=np.stack(colour_vectors),
        texture=np.stack(texture_vectors),
        mask_coverages=mask_coverages,
        mask_confidences=mask_confidences,
        segmentation_methods=segmentation_methods,
        fallback_reasons=fallback_reasons,
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
    left_colour_vectors: np.ndarray,
    right_colour_vectors: np.ndarray,
    left_texture_vectors: np.ndarray,
    right_texture_vectors: np.ndarray,
    *,
    candidate_count: int,
    require_same_item_type: bool,
) -> tuple[list[dict[str, str | float | int]], str]:
    """Use FAISS (when available) to retrieve FashionSigLIP candidates once."""

    rows: list[dict[str, str | float | int]] = []
    retriever = FashionCandidateRetriever(right_items, right_fashion_vectors)
    for left_index, left_item in enumerate(left_items):
        candidate_indices = retriever.search(
            left_item,
            left_fashion_vectors[left_index],
            candidate_count,
            require_same_item_type=require_same_item_type,
        )
        for rank, right_index in enumerate(candidate_indices, start=1):
            fashion_distance = float(
                np.clip(
                    1.0 - left_fashion_vectors[left_index] @ right_fashion_vectors[right_index],
                    0.0,
                    2.0,
                )
            )
            detail_distance = float(
                np.clip(
                    1.0 - left_detail_vectors[left_index] @ right_detail_vectors[right_index],
                    0.0,
                    2.0,
                )
            )
            colour_distance = cosine_distance(
                left_colour_vectors[left_index],
                right_colour_vectors[right_index],
            )
            texture_distance = cosine_distance(
                left_texture_vectors[left_index],
                right_texture_vectors[right_index],
            )
            rows.append(
                {
                    "leftId": left_ids[left_index],
                    "rightId": right_ids[right_index],
                    "fashionDistance": round(fashion_distance, 7),
                    "dinoDistance": round(detail_distance, 7),
                    "colourDistance": round(colour_distance, 7),
                    "textureDistance": round(texture_distance, 7),
                    "candidateRank": rank,
                }
            )
    return rows, retriever.backend


def build_artifact_vision_output(
    source: dict[str, Any],
    *,
    settings: PipelineSettings,
    encoder: FashionEncoder,
    preprocessor: ImagePreprocessor,
    reranker: ImageEncoder | None = None,
    candidate_count: int = 50,
    reranker_weight_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    require_same_item_type: bool = True,
    appearance_mask_enabled: bool = True,
    appearance_weights: dict[str, float] | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Build either legacy visual distances or two-stage retrieval candidates."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not reranker_weight_grid or any(not 0 <= weight <= 1 for weight in reranker_weight_grid):
        raise ValueError("reranker_weight_grid values must be between 0 and 1")
    resolved_appearance_weights = {
        "neural": 0.70,
        "colour": 0.20,
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
    historical = _encode_catalog(
        source["historical"],
        image_root=settings.historical_image_root,
        identifier_field="sourceId",
        encoder=encoder,
        reranker=reranker,
        preprocessor=preprocessor,
        appearance_mask_enabled=appearance_mask_enabled,
        batch_size=batch_size,
    )
    upcoming = _encode_catalog(
        source["upcoming"],
        image_root=settings.upcoming_image_root,
        identifier_field="id",
        encoder=encoder,
        reranker=reranker,
        preprocessor=preprocessor,
        appearance_mask_enabled=appearance_mask_enabled,
        batch_size=batch_size,
    )
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
        }

    assert historical.detail is not None
    assert upcoming.detail is not None
    historical_by_id = {str(item["id"]): item for item in source["historical"]}
    upcoming_by_id = {str(item["id"]): item for item in source["upcoming"]}
    historical_items = [historical_by_id[identifier] for identifier in historical.identifiers]
    upcoming_items = [upcoming_by_id[identifier] for identifier in upcoming.identifiers]
    historical_pairs, historical_index_backend = _two_stage_candidate_rows(
        historical_items,
        historical_items,
        historical.identifiers,
        historical.identifiers,
        historical.fashion,
        historical.fashion,
        historical.detail,
        historical.detail,
        historical.colour,
        historical.colour,
        historical.texture,
        historical.texture,
        candidate_count=candidate_count,
        require_same_item_type=require_same_item_type,
    )
    upcoming_pairs, upcoming_index_backend = _two_stage_candidate_rows(
        upcoming_items,
        historical_items,
        upcoming.identifiers,
        historical.identifiers,
        upcoming.fashion,
        historical.fashion,
        upcoming.detail,
        historical.detail,
        upcoming.colour,
        historical.colour,
        upcoming.texture,
        historical.texture,
        candidate_count=candidate_count,
        require_same_item_type=require_same_item_type,
    )
    candidate_pairs = historical_pairs + upcoming_pairs
    mask_coverages = historical.mask_coverages + upcoming.mask_coverages
    mask_confidences = historical.mask_confidences + upcoming.mask_confidences
    segmentation_methods = historical.segmentation_methods + upcoming.segmentation_methods
    accepted_mask_coverages = [
        coverage
        for coverage, method in zip(mask_coverages, segmentation_methods, strict=True)
        if method == "adaptive-lab-border-foreground-mask"
    ]
    accepted_mask_confidences = [
        confidence
        for confidence, method in zip(mask_confidences, segmentation_methods, strict=True)
        if method == "adaptive-lab-border-foreground-mask"
    ]
    fallback_counts = Counter(
        reason for reason in historical.fallback_reasons + upcoming.fallback_reasons if reason is not None
    )
    return {
        "engine": (
            "Two-stage FashionSigLIP candidate retrieval with DINOv2, masked CIELAB colour "
            "and texture visual-detail reranking"
        ),
        "modelId": encoder.model_id,
        "modelRevision": encoder.revision,
        "embeddingDimension": encoder.dimension,
        "device": str(getattr(encoder, "device", "unknown")),
        "historicalCoverage": len(historical.identifiers),
        "upcomingCoverage": len(upcoming.identifiers),
        "reranker": {
            "modelId": reranker.model_id,
            "modelRevision": reranker.revision,
            "embeddingDimension": reranker.dimension,
            "device": str(getattr(reranker, "device", "unknown")),
            "candidateCount": candidate_count,
            "weightGrid": list(reranker_weight_grid),
            "sameItemTypeConstraint": require_same_item_type,
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
                "segmentation": {
                    "enabled": appearance_mask_enabled,
                    "method": "adaptive-lab-border-foreground-mask",
                    "maskedImages": sum(
                        method == "adaptive-lab-border-foreground-mask" for method in segmentation_methods
                    ),
                    "fallbackImages": sum(fallback_counts.values()),
                    "fallbackReasons": dict(sorted(fallback_counts.items())),
                    "meanForegroundCoverage": (
                        round(float(np.mean(accepted_mask_coverages)), 4) if accepted_mask_coverages else 0.0
                    ),
                    "meanMaskConfidence": (
                        round(float(np.mean(accepted_mask_confidences)), 4) if accepted_mask_confidences else 0.0
                    ),
                },
                "colourDescriptor": {
                    "space": "CIELAB",
                    "binsPerChannel": COLOUR_BINS_PER_CHANNEL,
                    "maskOnly": True,
                },
                "textureDescriptor": {
                    "method": "masked-gradient-orientation-and-local-energy",
                    "dimension": TEXTURE_DESCRIPTOR_DIMENSION,
                },
                "weights": {key: round(value, 4) for key, value in resolved_appearance_weights.items()},
            },
        },
        "calibrationMethod": (
            "Separate robust logistic calibration for FashionSigLIP, DINOv2, masked CIELAB "
            "colour and texture candidate distances; component weights are configured explicitly"
        ),
        "candidatePairs": candidate_pairs,
    }
