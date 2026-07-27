from __future__ import annotations

from fashion_matching.config import MatchingSettings
from fashion_matching.encoders import FashionEncoder, create_encoder
from fashion_matching.preprocessing import ImagePreprocessor
from fashion_matching.storage import QdrantVectorStore, VectorStore


def build_encoder(settings: MatchingSettings) -> FashionEncoder:
    return create_encoder(
        settings.model_id,
        revision=settings.model_revision,
        device=settings.device,
    )


def build_preprocessor(settings: MatchingSettings) -> ImagePreprocessor:
    return ImagePreprocessor(
        version=settings.preprocessing_version,
        max_bytes=settings.max_image_bytes,
        max_pixels=settings.max_image_pixels,
        minimum_dimension=settings.min_image_dimension,
        pad_to_square=settings.pad_to_square,
        crop_uniform_background=settings.crop_uniform_background,
        allowed_image_domains=settings.allowed_image_domains,
    )


def build_store(settings: MatchingSettings) -> VectorStore:
    return QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
