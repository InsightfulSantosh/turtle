from __future__ import annotations

import math
import os
from dataclasses import dataclass

from fashion_matching.scoring import SignalWeights


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_weight_grid(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one numeric weight")
    return values


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
    dino_reranker_enabled: bool = True
    dino_model_id: str = "facebook/dinov2-base"
    dino_model_revision: str = "main"
    dino_candidate_count: int = 50
    dino_weight_grid: tuple[float, ...] = (0.5833333333,)
    dino_require_same_item_type: bool = True
    dino_require_same_design: bool = True
    dino_require_same_colour_family: bool = True
    pattern_gate_enabled: bool = True
    pattern_max_distance: float = 0.42
    appearance_mask_enabled: bool = True
    garment_segmentation_enabled: bool = True
    garment_segmentation_border_fallback: bool = True
    garment_detector_model_id: str = "IDEA-Research/grounding-dino-tiny"
    garment_detector_revision: str = "main"
    garment_sam2_model_id: str = "facebook/sam2.1-hiera-small"
    garment_sam2_revision: str = "main"
    garment_detector_threshold: float = 0.35
    garment_text_threshold: float = 0.25
    garment_minimum_coverage: float = 0.04
    garment_maximum_coverage: float = 0.88
    appearance_neural_weight: float = 0.60
    appearance_colour_weight: float = 0.30
    appearance_texture_weight: float = 0.10
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
        if not 1 <= self.dino_candidate_count <= self.candidate_count:
            raise ValueError("dino_candidate_count must be between 1 and candidate_count")
        if not self.dino_weight_grid or any(not 0 <= weight <= 1 for weight in self.dino_weight_grid):
            raise ValueError("dino_weight_grid values must be between 0 and 1")
        if not 0 <= self.pattern_max_distance <= 2:
            raise ValueError("pattern_max_distance must be between 0 and 2")
        appearance_weight_total = (
            self.appearance_neural_weight + self.appearance_colour_weight + self.appearance_texture_weight
        )
        if any(
            weight < 0
            for weight in (
                self.appearance_neural_weight,
                self.appearance_colour_weight,
                self.appearance_texture_weight,
            )
        ) or not math.isclose(appearance_weight_total, 1.0, abs_tol=1e-6):
            raise ValueError("appearance reranker weights must be non-negative and sum to 1")
        if any(
            not 0 <= value <= 1
            for value in (
                self.garment_detector_threshold,
                self.garment_text_threshold,
            )
        ):
            raise ValueError("garment detector thresholds must be between 0 and 1")
        if not 0 < self.garment_minimum_coverage < self.garment_maximum_coverage < 1:
            raise ValueError("garment mask coverage bounds must be inside (0, 1)")

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
            dino_reranker_enabled=_as_bool("FASHION_DINO_RERANK_ENABLED", True),
            dino_model_id=os.getenv("FASHION_DINO_MODEL_ID", "facebook/dinov2-base"),
            dino_model_revision=os.getenv("FASHION_DINO_MODEL_REVISION", "main"),
            dino_candidate_count=int(os.getenv("FASHION_DINO_CANDIDATE_COUNT", "50")),
            dino_weight_grid=_as_weight_grid(
                "FASHION_DINO_WEIGHT_GRID",
                (0.5833333333,),
            ),
            dino_require_same_item_type=_as_bool(
                "FASHION_DINO_REQUIRE_SAME_ITEM_TYPE",
                True,
            ),
            dino_require_same_design=_as_bool(
                "FASHION_DINO_REQUIRE_SAME_DESIGN",
                True,
            ),
            dino_require_same_colour_family=_as_bool(
                "FASHION_DINO_REQUIRE_SAME_COLOUR_FAMILY",
                True,
            ),
            pattern_gate_enabled=_as_bool("FASHION_PATTERN_GATE_ENABLED", True),
            pattern_max_distance=float(os.getenv("FASHION_PATTERN_MAX_DISTANCE", "0.42")),
            appearance_mask_enabled=_as_bool("FASHION_APPEARANCE_MASK_ENABLED", True),
            garment_segmentation_enabled=_as_bool("FASHION_GARMENT_SEGMENTATION_ENABLED", True),
            garment_segmentation_border_fallback=_as_bool(
                "FASHION_GARMENT_SEGMENTATION_BORDER_FALLBACK",
                True,
            ),
            garment_detector_model_id=os.getenv(
                "FASHION_GARMENT_DETECTOR_MODEL_ID",
                "IDEA-Research/grounding-dino-tiny",
            ),
            garment_detector_revision=os.getenv("FASHION_GARMENT_DETECTOR_REVISION", "main"),
            garment_sam2_model_id=os.getenv("FASHION_GARMENT_SAM2_MODEL_ID", "facebook/sam2.1-hiera-small"),
            garment_sam2_revision=os.getenv("FASHION_GARMENT_SAM2_REVISION", "main"),
            garment_detector_threshold=float(os.getenv("FASHION_GARMENT_DETECTOR_THRESHOLD", "0.35")),
            garment_text_threshold=float(os.getenv("FASHION_GARMENT_TEXT_THRESHOLD", "0.25")),
            garment_minimum_coverage=float(os.getenv("FASHION_GARMENT_MINIMUM_COVERAGE", "0.04")),
            garment_maximum_coverage=float(os.getenv("FASHION_GARMENT_MAXIMUM_COVERAGE", "0.88")),
            appearance_neural_weight=float(os.getenv("FASHION_APPEARANCE_NEURAL_WEIGHT", "0.60")),
            appearance_colour_weight=float(os.getenv("FASHION_APPEARANCE_COLOUR_WEIGHT", "0.30")),
            appearance_texture_weight=float(os.getenv("FASHION_APPEARANCE_TEXTURE_WEIGHT", "0.10")),
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
