from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fashion_matching.manifests import ManifestError, read_manifest
from fashion_matching.models import ManifestRecord
from fashion_matching.preprocessing import (
    ImagePreprocessor,
    ImageValidationError,
)


def _record(path: Path) -> ManifestRecord:
    return ManifestRecord(
        product_id="PRODUCT-1",
        image_id="IMAGE-1",
        image_path=path,
    )


def test_preprocessing_is_deterministic_and_preserves_aspect_ratio(
    tmp_path: Path,
) -> None:
    path = tmp_path / "item.png"
    Image.new("RGB", (80, 40), (10, 20, 30)).save(path)
    preprocessor = ImagePreprocessor()
    first = preprocessor.prepare(_record(path))
    second = preprocessor.prepare(_record(path))
    assert first.checksum == second.checksum
    assert first.image.tobytes() == second.image.tobytes()
    assert first.image.size == (80, 80)
    assert first.image.getpixel((40, 5)) == (255, 255, 255)
    assert first.image.getpixel((40, 40)) == (10, 20, 30)


def test_preprocessing_applies_exif_orientation(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (40, 80), "navy")
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    prepared = ImagePreprocessor(pad_to_square=False).prepare(_record(path))
    assert prepared.image.size == (80, 40)


@pytest.mark.parametrize("name", ["missing.jpg", "corrupt.png"])
def test_preprocessing_rejects_missing_or_corrupt_images(
    tmp_path: Path,
    name: str,
) -> None:
    path = tmp_path / name
    if name.startswith("corrupt"):
        path.write_bytes(b"not-an-image")
    with pytest.raises(ImageValidationError):
        ImagePreprocessor().prepare(_record(path))


def test_preprocessing_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "large.png"
    Image.new("RGB", (40, 40), "red").save(path)
    with pytest.raises(ImageValidationError, match="byte limit"):
        ImagePreprocessor(max_bytes=10).prepare(_record(path))


def test_manifest_requires_unique_image_ids_and_does_not_use_id_as_text(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "item.png"
    Image.new("RGB", (40, 40), "red").save(image_path)
    manifest = tmp_path / "catalog.csv"
    manifest.write_text(
        "product_id,image_id,image_path,description\nSKU-SECRET,IMAGE-1,item.png,Red cotton shirt\n",
        encoding="utf-8",
    )
    records = read_manifest(manifest)
    assert records[0].text == "Red cotton shirt"
    assert "SKU-SECRET" not in records[0].text

    manifest.write_text(
        "product_id,image_id,image_path\nP1,DUP,item.png\nP2,DUP,item.png\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="duplicate image_id"):
        read_manifest(manifest)
