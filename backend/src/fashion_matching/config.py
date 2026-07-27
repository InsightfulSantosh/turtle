from __future__ import annotations

import os
from dataclasses import dataclass

from fashion_matching.scoring import SignalWeights


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MatchingSettings:
    model_id: str = "Marqo/marqo-fashionSigLIP"
    model_revision: str = "main"
    device: str = "auto"
    batch_size: int = 16
    candidate_count: int = 100
    top_k: int = 5
    minimum_score: float | None = 0.50
    preprocessing_version: str = "rgb-exif-pad-v1"
    max_image_bytes: int = 8 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    min_image_dimension: int = 32
    pad_to_square: bool = True
    crop_uniform_background: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_prefix: str = "turtle-fashion"
    collection_alias: str = "turtle-fashion-active"
    allowed_image_domains: tuple[str, ...] = ()
    weights: SignalWeights = SignalWeights(
        image=0.70,
        attributes=0.20,
        text=0.10,
    )

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")
        if not 1 <= self.candidate_count <= 10_000:
            raise ValueError("candidate_count must be between 1 and 10000")
        if not 1 <= self.top_k <= self.candidate_count:
            raise ValueError("top_k must be positive and no larger than candidate_count")
        if self.minimum_score is not None and not 0 <= self.minimum_score <= 1:
            raise ValueError("minimum_score must be between 0 and 1")

    @classmethod
    def from_environment(cls) -> MatchingSettings:
        threshold = os.getenv("FASHION_MINIMUM_SCORE")
        domains = tuple(
            part.strip().lower() for part in os.getenv("ALLOWED_IMAGE_DOMAINS", "").split(",") if part.strip()
        )
        return cls(
            model_id=os.getenv(
                "FASHION_MATCHING_MODEL_ID",
                "Marqo/marqo-fashionSigLIP",
            ),
            model_revision=os.getenv("FASHION_MATCHING_MODEL_REVISION", "main"),
            device=os.getenv("FASHION_MATCHING_DEVICE", "auto"),
            batch_size=int(os.getenv("FASHION_MATCHING_BATCH_SIZE", "16")),
            candidate_count=int(os.getenv("FASHION_MATCHING_CANDIDATE_COUNT", "100")),
            top_k=int(os.getenv("FASHION_MATCHING_TOP_K", "5")),
            minimum_score=float(threshold) if threshold else 0.62,
            preprocessing_version=os.getenv(
                "FASHION_PREPROCESSING_VERSION",
                "rgb-exif-pad-v1",
            ),
            max_image_bytes=int(os.getenv("FASHION_MAX_IMAGE_BYTES", str(8 * 1024 * 1024))),
            max_image_pixels=int(os.getenv("FASHION_MAX_IMAGE_PIXELS", "40000000")),
            min_image_dimension=int(os.getenv("FASHION_MIN_IMAGE_DIMENSION", "32")),
            pad_to_square=_as_bool("FASHION_PAD_TO_SQUARE", True),
            crop_uniform_background=_as_bool(
                "FASHION_CROP_UNIFORM_BACKGROUND",
                False,
            ),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            collection_prefix=os.getenv(
                "FASHION_COLLECTION_PREFIX",
                "turtle-fashion",
            ),
            collection_alias=os.getenv(
                "FASHION_COLLECTION_ALIAS",
                "turtle-fashion-active",
            ),
            allowed_image_domains=domains,
            weights=SignalWeights(
                image=float(os.getenv("FASHION_IMAGE_WEIGHT", "0.70")),
                attributes=float(os.getenv("FASHION_ATTRIBUTE_WEIGHT", "0.20")),
                text=float(os.getenv("FASHION_TEXT_WEIGHT", "0.10")),
            ),
        )
