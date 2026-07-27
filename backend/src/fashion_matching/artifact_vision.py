"""Generate visual-distance input for the browser decision artifact."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from data_pipeline.images import build_image_index
from data_pipeline.settings import PipelineSettings
from fashion_matching.encoders import FashionEncoder
from fashion_matching.models import ManifestRecord
from fashion_matching.preprocessing import ImagePreprocessor, ImageValidationError

LOGGER = logging.getLogger(__name__)


def _encode_catalog(
    items: list[dict[str, Any]],
    *,
    image_root: Path,
    identifier_field: str,
    encoder: FashionEncoder,
    preprocessor: ImagePreprocessor,
    batch_size: int,
) -> tuple[list[str], np.ndarray]:
    image_index = build_image_index(image_root)
    path_to_items: dict[Path, list[dict[str, Any]]] = {}
    for item in items:
        identifier = str(item.get(identifier_field, "")).upper()
        image_path = image_index.get(identifier)
        if item.get("imageUrl") and image_path is not None:
            path_to_items.setdefault(image_path, []).append(item)

    paths = list(path_to_items)
    path_vectors: dict[Path, np.ndarray] = {}
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
            vectors = encoder.encode_images(prepared_images)
        finally:
            for image in prepared_images:
                image.close()
        if len(vectors) != len(prepared_paths):
            raise RuntimeError("Image encoder returned a different number of vectors")
        for image_path, vector in zip(prepared_paths, vectors, strict=True):
            path_vectors[image_path] = np.asarray(vector, dtype=np.float32)
        LOGGER.info(
            "Encoded %s/%s unique images from %s",
            min(offset + batch_size, len(paths)),
            len(paths),
            image_root.name,
        )

    identifiers: list[str] = []
    vectors: list[np.ndarray] = []
    for image_path, catalog_items in path_to_items.items():
        vector = path_vectors.get(image_path)
        if vector is None:
            continue
        for item in catalog_items:
            item["hasVisualFeature"] = True
            identifiers.append(str(item["id"]))
            vectors.append(vector)
    if not vectors:
        return identifiers, np.empty((0, int(encoder.dimension)), dtype=np.float32)
    return identifiers, np.stack(vectors)


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


def build_artifact_vision_output(
    source: dict[str, Any],
    *,
    settings: PipelineSettings,
    encoder: FashionEncoder,
    preprocessor: ImagePreprocessor,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Embed mapped images and produce cosine distances for model calibration."""

    historical_ids, historical_vectors = _encode_catalog(
        source["historical"],
        image_root=settings.historical_image_root,
        identifier_field="sourceId",
        encoder=encoder,
        preprocessor=preprocessor,
        batch_size=batch_size,
    )
    upcoming_ids, upcoming_vectors = _encode_catalog(
        source["upcoming"],
        image_root=settings.upcoming_image_root,
        identifier_field="id",
        encoder=encoder,
        preprocessor=preprocessor,
        batch_size=batch_size,
    )
    distances = _distance_rows(
        historical_ids,
        historical_ids,
        historical_vectors,
        historical_vectors,
    )
    distances.extend(
        _distance_rows(
            upcoming_ids,
            historical_ids,
            upcoming_vectors,
            historical_vectors,
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
