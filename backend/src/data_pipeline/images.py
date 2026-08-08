"""Map catalogue identifiers to the product images available on disk."""

from __future__ import annotations

from pathlib import Path

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
