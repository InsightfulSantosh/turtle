"""Generate two-stage visual-retrieval input for the browser decision artifact."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from data_pipeline.images import build_image_index
from data_pipeline.settings import PipelineSettings
from fashion_matching.encoders import FashionEncoder, ImageEncoder
from fashion_matching.models import ManifestRecord
from fashion_matching.preprocessing import ImagePreprocessor, ImageValidationError

LOGGER = logging.getLogger(__name__)


def _encode_catalog(
    items: list[dict[str, Any]],
    *,
    image_root: Path,
    identifier_field: str,
    encoder: FashionEncoder,
    reranker: ImageEncoder | None,
    preprocessor: ImagePreprocessor,
    batch_size: int,
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
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
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        prepared_paths: list[Path] = []
        prepared_images = []
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
            prepared_paths.append(image_path)
            prepared_images.append(prepared.image)

        if not prepared_images:
            continue
        try:
            fashion_vectors = encoder.encode_images(prepared_images)
            detail_vectors = reranker.encode_images(prepared_images) if reranker else None
        finally:
            for image in prepared_images:
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
    for image_path, catalog_items in path_to_items.items():
        fashion_vector = fashion_vectors_by_path.get(image_path)
        detail_vector = detail_vectors_by_path.get(image_path) if reranker else None
        if fashion_vector is None or (reranker is not None and detail_vector is None):
            continue
        for item in catalog_items:
            item["hasVisualFeature"] = True
            identifiers.append(str(item["id"]))
            fashion_vectors.append(fashion_vector)
            if detail_vector is not None:
                detail_vectors.append(detail_vector)
    if not fashion_vectors:
        empty_fashion = np.empty((0, int(encoder.dimension)), dtype=np.float32)
        empty_detail = (
            np.empty((0, int(reranker.dimension)), dtype=np.float32)
            if reranker is not None
            else None
        )
        return identifiers, empty_fashion, empty_detail
    return (
        identifiers,
        np.stack(fashion_vectors),
        np.stack(detail_vectors) if reranker is not None else None,
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


def _item_type(item: dict[str, Any]) -> str:
    return " ".join(str(item.get("itemType") or "").upper().split())


def _candidate_indices(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    candidate_count: int,
    require_same_item_type: bool,
) -> list[int]:
    """Return a FashionSigLIP shortlist before the DINOv2 reranking stage.

    The constraint is relaxed only when it would leave a query with fewer than
    two choices. This keeps retrieval useful for sparse categories while
    protecting the normal shirt-versus-trouser case from visual false matches.
    """

    eligible = list(range(len(candidates)))
    query_type = _item_type(query)
    if require_same_item_type and query_type:
        same_type = [index for index, item in enumerate(candidates) if _item_type(item) == query_type]
        if len(same_type) >= 2:
            eligible = same_type
    scores = candidate_vectors[np.asarray(eligible)] @ query_vector
    ranked = np.argsort(-scores, kind="stable")[:candidate_count]
    return [eligible[int(index)] for index in ranked]


def _two_stage_candidate_rows(
    left_items: list[dict[str, Any]],
    right_items: list[dict[str, Any]],
    left_ids: list[str],
    right_ids: list[str],
    left_fashion_vectors: np.ndarray,
    right_fashion_vectors: np.ndarray,
    left_detail_vectors: np.ndarray,
    right_detail_vectors: np.ndarray,
    *,
    candidate_count: int,
    require_same_item_type: bool,
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for left_index, left_item in enumerate(left_items):
        candidate_indices = _candidate_indices(
            left_item,
            right_items,
            left_fashion_vectors[left_index],
            right_fashion_vectors,
            candidate_count=candidate_count,
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
            rows.append(
                {
                    "leftId": left_ids[left_index],
                    "rightId": right_ids[right_index],
                    "fashionDistance": round(fashion_distance, 7),
                    "dinoDistance": round(detail_distance, 7),
                    "candidateRank": rank,
                }
            )
    return rows


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
    batch_size: int = 16,
) -> dict[str, Any]:
    """Build either legacy visual distances or two-stage retrieval candidates."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not reranker_weight_grid or any(not 0 <= weight <= 1 for weight in reranker_weight_grid):
        raise ValueError("reranker_weight_grid values must be between 0 and 1")
    historical_ids, historical_fashion, historical_detail = _encode_catalog(
        source["historical"],
        image_root=settings.historical_image_root,
        identifier_field="sourceId",
        encoder=encoder,
        reranker=reranker,
        preprocessor=preprocessor,
        batch_size=batch_size,
    )
    upcoming_ids, upcoming_fashion, upcoming_detail = _encode_catalog(
        source["upcoming"],
        image_root=settings.upcoming_image_root,
        identifier_field="id",
        encoder=encoder,
        reranker=reranker,
        preprocessor=preprocessor,
        batch_size=batch_size,
    )
    if reranker is None:
        distances = _distance_rows(
            historical_ids,
            historical_ids,
            historical_fashion,
            historical_fashion,
        )
        distances.extend(
            _distance_rows(
                upcoming_ids,
                historical_ids,
                upcoming_fashion,
                historical_fashion,
            )
        )
        return {
            "engine": "Fashion-domain SigLIP image embeddings with cosine distance",
            "modelId": encoder.model_id,
            "modelRevision": encoder.revision,
            "embeddingDimension": encoder.dimension,
            "device": str(getattr(encoder, "device", "unknown")),
            "historicalCoverage": len(historical_ids),
            "upcomingCoverage": len(upcoming_ids),
            "calibrationMethod": (
                "Separate robust logistic calibration for historical validation "
                "and upcoming-to-historical serving distances"
            ),
            "distances": distances,
        }

    assert historical_detail is not None
    assert upcoming_detail is not None
    historical_by_id = {str(item["id"]): item for item in source["historical"]}
    upcoming_by_id = {str(item["id"]): item for item in source["upcoming"]}
    historical_items = [historical_by_id[identifier] for identifier in historical_ids]
    upcoming_items = [upcoming_by_id[identifier] for identifier in upcoming_ids]
    candidate_pairs = _two_stage_candidate_rows(
        historical_items,
        historical_items,
        historical_ids,
        historical_ids,
        historical_fashion,
        historical_fashion,
        historical_detail,
        historical_detail,
        candidate_count=candidate_count,
        require_same_item_type=require_same_item_type,
    )
    candidate_pairs.extend(
        _two_stage_candidate_rows(
            upcoming_items,
            historical_items,
            upcoming_ids,
            historical_ids,
            upcoming_fashion,
            historical_fashion,
            upcoming_detail,
            historical_detail,
            candidate_count=candidate_count,
            require_same_item_type=require_same_item_type,
        )
    )
    return {
        "engine": "Two-stage FashionSigLIP candidate retrieval with DINOv2 visual-detail reranking",
        "modelId": encoder.model_id,
        "modelRevision": encoder.revision,
        "embeddingDimension": encoder.dimension,
        "device": str(getattr(encoder, "device", "unknown")),
        "historicalCoverage": len(historical_ids),
        "upcomingCoverage": len(upcoming_ids),
        "reranker": {
            "modelId": reranker.model_id,
            "modelRevision": reranker.revision,
            "embeddingDimension": reranker.dimension,
            "device": str(getattr(reranker, "device", "unknown")),
            "candidateCount": candidate_count,
            "weightGrid": list(reranker_weight_grid),
            "sameItemTypeConstraint": require_same_item_type,
        },
        "calibrationMethod": (
            "Separate robust logistic calibration for FashionSigLIP and DINOv2 "
            "candidate distances; reranker weight selected on temporal holdout"
        ),
        "candidatePairs": candidate_pairs,
    }
