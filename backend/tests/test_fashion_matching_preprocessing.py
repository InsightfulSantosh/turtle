from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fashion_matching.appearance import (
    OTTR_TROUSER_ROI,
    delta_e_ciede2000,
    dominant_palette_distance,
    extract_appearance_features,
    requires_ottr_pattern_gate,
)
from fashion_matching.preprocessing import (
    ImagePreprocessor,
    ImageValidationError,
)


def test_preprocessing_is_deterministic_and_preserves_aspect_ratio(
    tmp_path: Path,
) -> None:
    path = tmp_path / "item.png"
    Image.new("RGB", (80, 40), (10, 20, 30)).save(path)
    preprocessor = ImagePreprocessor()
    first = preprocessor.prepare(path)
    second = preprocessor.prepare(path)
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
    prepared = ImagePreprocessor(pad_to_square=False).prepare(path)
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
        ImagePreprocessor().prepare(path)


def test_preprocessing_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "large.png"
    Image.new("RGB", (40, 40), "red").save(path)
    with pytest.raises(ImageValidationError, match="byte limit"):
        ImagePreprocessor(max_bytes=10).prepare(path)


def test_appearance_features_compare_dominant_garment_palettes() -> None:
    red = Image.new("RGB", (120, 160), "white")
    lighter_red = Image.new("RGB", (120, 160), "white")
    blue = Image.new("RGB", (120, 160), "white")
    for top in range(30, 130):
        for left in range(25, 95):
            red.putpixel((left, top), (210, 35, 35))
            lighter_red.putpixel((left, top), (225, 50, 48))
            blue.putpixel((left, top), (35, 60, 210))

    red_features = extract_appearance_features(red)
    lighter_red_features = extract_appearance_features(lighter_red)
    blue_features = extract_appearance_features(blue)

    assert red_features.image.getpixel((0, 0)) == (255, 255, 255)
    assert dominant_palette_distance(
        red_features.colour_vector,
        lighter_red_features.colour_vector,
    ) < dominant_palette_distance(
        red_features.colour_vector,
        blue_features.colour_vector,
    )
    assert dominant_palette_distance(red_features.colour_vector, blue_features.colour_vector) > 0.2
    assert len(red_features.texture_vector) == 16


def test_ottr_colour_uses_lower_centre_trouser_region_only() -> None:
    composed = Image.new("RGB", (160, 240), (35, 60, 210))
    for top in range(0, 70):
        for left in range(160):
            composed.putpixel((left, top), (210, 35, 35))
    blue = Image.new("RGB", composed.size, (35, 60, 210))

    generic = extract_appearance_features(composed, item_type="OTSH")
    trousers = extract_appearance_features(composed, item_type="OTTR")
    blue_trousers = extract_appearance_features(blue, item_type="OTTR")

    assert dominant_palette_distance(
        trousers.colour_vector,
        blue_trousers.colour_vector,
    ) < 0.01
    assert dominant_palette_distance(
        generic.colour_vector,
        blue_trousers.colour_vector,
    ) > 0.05


def test_ottr_analysis_image_excludes_shoes_for_every_visual_encoder() -> None:
    image = Image.new("RGB", (100, 100), (245, 245, 245))
    for top in range(28, 81):
        for left in range(16, 85):
            image.putpixel((left, top), (35, 60, 210))
    for top in range(82, 100):
        for left in range(100):
            image.putpixel((left, top), (10, 10, 10))

    trousers = extract_appearance_features(image, item_type="OTTR")
    generic = extract_appearance_features(image, item_type="OTSH")
    blue_reference = extract_appearance_features(
        Image.new("RGB", trousers.image.size, (35, 60, 210))
    )

    assert OTTR_TROUSER_ROI == (0.16, 0.28, 0.84, 0.80)
    assert trousers.image.size == (68, 52)
    assert trousers.image.getpixel((34, 51)) == (35, 60, 210)
    assert generic.image.size == image.size
    assert generic.image.getpixel((50, 90)) == (10, 10, 10)
    assert dominant_palette_distance(
        trousers.colour_vector,
        blue_reference.colour_vector,
    ) < 0.01


@pytest.mark.parametrize(
    ("design", "expected"),
    [
        ("CHECKS", True),
        ("PRINTS", True),
        ("STRIPES", True),
        ("DOBBY/STRUCTURE", True),
        ("PLAINS", False),
        ("SOLID", False),
    ],
)
def test_ottr_pattern_gate_is_limited_to_visible_pattern_designs(
    design: str,
    expected: bool,
) -> None:
    assert requires_ottr_pattern_gate(design) is expected


def test_ciede2000_matches_published_reference_pair() -> None:
    left = [50.0, 2.6772, -79.7751]
    right = [50.0, 0.0, -82.7485]

    assert float(delta_e_ciede2000(left, right)) == pytest.approx(2.0425, abs=1e-4)
