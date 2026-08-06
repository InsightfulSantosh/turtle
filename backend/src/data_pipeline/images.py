"""Map catalogue identifiers to the product images available on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def build_image_index(directory: Path) -> dict[str, Path]:
    """Return a case-insensitive product-id-to-image lookup."""

    if not directory.is_dir():
        return {}
    return {
        path.stem.upper(): path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }


def attach_catalog_images(
    items: list[dict[str, Any]],
    *,
    image_directory: Path,
    catalog: str,
    identifier_field: str,
) -> int:
    """Attach backend image URLs to items whose identifiers have local images."""

    image_index = build_image_index(image_directory)
    matched = 0
    for item in items:
        identifier = str(item.get(identifier_field, "")).strip()
        image_path = image_index.get(identifier.upper())
        item["imageUrl"] = f"/product-images/{catalog}/{identifier}" if image_path is not None else None
        item["hasVisualFeature"] = False
        matched += image_path is not None
    return matched
