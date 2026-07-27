from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ATTRIBUTE_FIELDS = (
    "item_type",
    "category",
    "category_type",
    "colour",
    "pattern",
    "material",
    "fabric",
    "design",
    "fit",
    "gender",
    "brand",
)


def _clean_optional(value: Any) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    return text or None


@dataclass(frozen=True)
class ManifestRecord:
    product_id: str
    image_id: str
    image_path: Path | None = None
    image_url: str | None = None
    view: str | None = None
    text: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    @property
    def image_source(self) -> str:
        if self.image_path is not None:
            return str(self.image_path)
        if self.image_url is not None:
            return self.image_url
        raise ValueError("manifest record has no image source")

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        base_directory: Path,
    ) -> ManifestRecord:
        product_id = _clean_optional(row.get("product_id"))
        image_id = _clean_optional(row.get("image_id"))
        if not product_id:
            raise ValueError("product_id is required")
        if not image_id:
            raise ValueError("image_id is required")

        raw_path = _clean_optional(row.get("image_path"))
        image_url = _clean_optional(row.get("image_url"))
        if bool(raw_path) == bool(image_url):
            raise ValueError("exactly one of image_path or image_url is required")
        image_path = None
        if raw_path:
            candidate = Path(raw_path).expanduser()
            image_path = candidate if candidate.is_absolute() else (base_directory / candidate).resolve()

        text_parts = [_clean_optional(row.get(name)) for name in ("title", "description", "text")]
        text = " | ".join(part for part in text_parts if part) or None
        attributes = {name: value for name in ATTRIBUTE_FIELDS if (value := _clean_optional(row.get(name))) is not None}
        return cls(
            product_id=product_id,
            image_id=image_id,
            image_path=image_path,
            image_url=image_url,
            view=_clean_optional(row.get("view")),
            text=text,
            attributes=attributes,
        )


@dataclass(frozen=True)
class PreparedImage:
    image: Any
    checksum: str
    width: int
    height: int
    source_bytes: int


@dataclass(frozen=True)
class VectorPoint:
    point_id: str
    vectors: Mapping[str, list[float]]
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class SearchHit:
    point_id: str
    score: float
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class RankedMatch:
    product_id: str
    image_id: str
    view: str | None
    rank: int
    final_score: float
    image_score: float | None
    text_score: float | None
    attribute_score: float | None
    applied_weights: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "image_id": self.image_id,
            "view": self.view,
            "rank": self.rank,
            "final_score": self.final_score,
            "image_score": self.image_score,
            "text_score": self.text_score,
            "attribute_score": self.attribute_score,
            "applied_weights": dict(self.applied_weights),
        }


@dataclass(frozen=True)
class MatchResult:
    query_product_id: str
    query_image_id: str
    model_version: str
    matches: tuple[RankedMatch, ...]
    no_suitable_match: bool
    processing_time_ms: float
    error: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_product_id": self.query_product_id,
            "query_image_id": self.query_image_id,
            "model_version": self.model_version,
            "matches": [match.to_dict() for match in self.matches],
            "no_suitable_match": self.no_suitable_match,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "error": self.error,
            "warnings": list(self.warnings),
        }


@dataclass
class IndexSummary:
    total_images: int = 0
    successfully_indexed: int = 0
    skipped_images: int = 0
    updated_images: int = 0
    failed_images: int = 0
    processing_time_seconds: float = 0.0
    failures: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        indexed_or_updated = self.successfully_indexed + self.updated_images
        return {
            "total_images": self.total_images,
            "successfully_indexed": self.successfully_indexed,
            "skipped_images": self.skipped_images,
            "updated_images": self.updated_images,
            "failed_images": self.failed_images,
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "indexing_throughput_images_per_second": (
                round(
                    indexed_or_updated / self.processing_time_seconds,
                    3,
                )
                if self.processing_time_seconds > 0
                else 0.0
            ),
            "failures": list(self.failures),
        }
