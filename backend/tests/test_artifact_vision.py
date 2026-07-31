from __future__ import annotations

from pathlib import Path

from PIL import Image

from data_pipeline.settings import PipelineSettings
from fashion_matching.artifact_vision import build_artifact_vision_output
from fashion_matching.preprocessing import ImagePreprocessor


class FakeEncoder:
    model_id = "local/test-fashion-encoder"
    revision = "test-revision"
    dimension = 3
    supports_text = False
    device = "cpu"

    def encode_images(self, images):
        colours = [image.getpixel((0, 0)) for image in images]
        vectors = []
        for red, green, blue in colours:
            total = max(red + green + blue, 1)
            vectors.append([red / total, green / total, blue / total])
        return vectors


class FakeDetailEncoder(FakeEncoder):
    model_id = "local/test-dino-reranker"
    revision = "test-dino-revision"


def _save_image(path: Path, colour: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), colour).save(path)


def test_artifact_vision_embeds_only_mapped_product_images(
    tmp_path: Path,
) -> None:
    settings = PipelineSettings(
        data_root=tmp_path / "DATA",
        temporary_root=tmp_path / "tmp",
        output_path=tmp_path / "artifact.json",
    )
    _save_image(
        settings.historical_image_root / "OTSH-100-1001.jpg",
        (255, 0, 0),
    )
    _save_image(
        settings.historical_image_root / "OTSH-101-1001.jpg",
        (0, 255, 0),
    )
    _save_image(
        settings.upcoming_image_root / "OTSH-200-1001.jpg",
        (255, 0, 0),
    )
    source = {
        "historical": [
            {
                "id": "AW25-OTSH-100-1001",
                "sourceId": "OTSH-100-1001",
                "imageUrl": "/product-images/historical/OTSH-100-1001",
                "hasVisualFeature": False,
            },
            {
                "id": "AW25-OTSH-101-1001",
                "sourceId": "OTSH-101-1001",
                "imageUrl": "/product-images/historical/OTSH-101-1001",
                "hasVisualFeature": False,
            },
        ],
        "upcoming": [
            {
                "id": "OTSH-200-1001",
                "imageUrl": "/product-images/upcoming/OTSH-200-1001",
                "hasVisualFeature": False,
            },
            {
                "id": "OTSH-201-1001",
                "imageUrl": None,
                "hasVisualFeature": False,
            },
        ],
    }

    output = build_artifact_vision_output(
        source,
        settings=settings,
        encoder=FakeEncoder(),
        preprocessor=ImagePreprocessor(pad_to_square=False),
        batch_size=2,
    )

    assert output["historicalCoverage"] == 2
    assert output["upcomingCoverage"] == 1
    assert output["embeddingDimension"] == 3
    assert len(output["distances"]) == 6
    assert source["historical"][0]["hasVisualFeature"] is True
    assert source["upcoming"][0]["hasVisualFeature"] is True
    assert source["upcoming"][1]["hasVisualFeature"] is False
    matching_distance = next(
        row["distance"]
        for row in output["distances"]
        if row["leftId"] == "OTSH-200-1001" and row["rightId"] == "AW25-OTSH-100-1001"
    )
    assert matching_distance == 0.0


def test_two_stage_visual_artifact_reranks_only_same_item_type_candidates(
    tmp_path: Path,
) -> None:
    settings = PipelineSettings(
        data_root=tmp_path / "DATA",
        temporary_root=tmp_path / "tmp",
        output_path=tmp_path / "artifact.json",
    )
    for identifier, colour in (
        ("OTSH-100-1001", (255, 0, 0)),
        ("OTSH-101-1001", (0, 255, 0)),
        ("OTSH-102-1001", (255, 0, 0)),
        ("OTTR-102-1001", (255, 0, 0)),
    ):
        _save_image(settings.historical_image_root / f"{identifier}.jpg", colour)
    _save_image(settings.upcoming_image_root / "OTSH-200-1001.jpg", (255, 0, 0))
    source = {
        "historical": [
            {
                "id": "AW25-OTSH-100-1001",
                "sourceId": "OTSH-100-1001",
                "itemType": "OTSH",
                "design": "CHECKS",
                "colour": "BLUE",
                "imageUrl": "/product-images/historical/OTSH-100-1001",
                "hasVisualFeature": False,
            },
            {
                "id": "AW25-OTSH-101-1001",
                "sourceId": "OTSH-101-1001",
                "itemType": "OTSH",
                "design": "CHECKS",
                "colour": "NAVY BLUE",
                "imageUrl": "/product-images/historical/OTSH-101-1001",
                "hasVisualFeature": False,
            },
            {
                "id": "AW25-OTSH-102-1001",
                "sourceId": "OTSH-102-1001",
                "itemType": "OTSH",
                "design": "STRIPES",
                "colour": "BLUE",
                "imageUrl": "/product-images/historical/OTSH-102-1001",
                "hasVisualFeature": False,
            },
            {
                "id": "AW25-OTTR-102-1001",
                "sourceId": "OTTR-102-1001",
                "itemType": "OTTR",
                "design": "CHECKS",
                "colour": "BLUE",
                "imageUrl": "/product-images/historical/OTTR-102-1001",
                "hasVisualFeature": False,
            },
        ],
        "upcoming": [
            {
                "id": "OTSH-200-1001",
                "itemType": "OTSH",
                "design": "CHECKS",
                "colour": "ROYAL BLUE",
                "imageUrl": "/product-images/upcoming/OTSH-200-1001",
                "hasVisualFeature": False,
            },
        ],
    }

    output = build_artifact_vision_output(
        source,
        settings=settings,
        encoder=FakeEncoder(),
        reranker=FakeDetailEncoder(),
        preprocessor=ImagePreprocessor(pad_to_square=False),
        candidate_count=2,
        pattern_gate_enabled=False,
    )

    assert output["reranker"]["modelId"] == "local/test-dino-reranker"
    assert output["reranker"]["candidateIndex"]["metric"].startswith("inner-product")
    assert output["reranker"]["sameItemTypeConstraint"] is True
    assert output["reranker"]["sameDesignConstraint"] is True
    assert output["reranker"]["appearance"]["colourDescriptor"]["space"] == "CIELAB"
    assert output["reranker"]["appearance"]["weights"] == {
        "neural": 0.6,
        "colour": 0.3,
        "texture": 0.1,
    }
    upcoming_candidates = [row for row in output["candidatePairs"] if row["leftId"] == "OTSH-200-1001"]
    assert [row["candidateRank"] for row in upcoming_candidates] == [1, 2]
    assert {row["rightId"] for row in upcoming_candidates} == {
        "AW25-OTSH-100-1001",
        "AW25-OTSH-101-1001",
    }
    assert all("colourDistance" in row and "textureDistance" in row for row in upcoming_candidates)
    assert all(row["colourDistance"] is None and row["textureDistance"] is None for row in upcoming_candidates)
    segmentation = output["reranker"]["appearance"]["segmentation"]
    assert segmentation["maskRequiredForColourAndTexture"] is True
    assert segmentation["unavailableAppearanceImages"] == 5


def test_pattern_gate_excludes_visibly_different_candidates_with_the_same_design_label(
    tmp_path: Path,
) -> None:
    settings = PipelineSettings(
        data_root=tmp_path / "DATA",
        temporary_root=tmp_path / "tmp",
        output_path=tmp_path / "artifact.json",
    )
    _save_image(settings.historical_image_root / "OTSH-100-1001.jpg", (255, 0, 0))
    _save_image(settings.historical_image_root / "OTSH-101-1001.jpg", (0, 255, 0))
    _save_image(settings.upcoming_image_root / "OTSH-200-1001.jpg", (255, 0, 0))
    source = {
        "historical": [
            {
                "id": "AW25-OTSH-100-1001",
                "sourceId": "OTSH-100-1001",
                "itemType": "OTSH",
                "design": "CHECKS",
                "colour": "BLUE",
                "imageUrl": "/product-images/historical/OTSH-100-1001",
            },
            {
                "id": "AW25-OTSH-101-1001",
                "sourceId": "OTSH-101-1001",
                "itemType": "OTSH",
                "design": "CHECKS",
                "colour": "NAVY BLUE",
                "imageUrl": "/product-images/historical/OTSH-101-1001",
            },
        ],
        "upcoming": [
            {
                "id": "OTSH-200-1001",
                "itemType": "OTSH",
                "design": "CHECKS",
                "colour": "ROYAL BLUE",
                "imageUrl": "/product-images/upcoming/OTSH-200-1001",
            }
        ],
    }

    output = build_artifact_vision_output(
        source,
        settings=settings,
        encoder=FakeEncoder(),
        reranker=FakeDetailEncoder(),
        preprocessor=ImagePreprocessor(pad_to_square=False),
        candidate_count=2,
        pattern_max_distance=0.1,
    )

    upcoming_candidates = [row for row in output["candidatePairs"] if row["leftId"] == "OTSH-200-1001"]
    assert [row["rightId"] for row in upcoming_candidates] == ["AW25-OTSH-100-1001"]
    assert upcoming_candidates[0]["patternDistance"] < 1e-6
    assert output["reranker"]["patternGate"] == {
        "enabled": True,
        "method": "three-scale-centre-garment-body-DINOv2",
        "designs": "checks, stripes, prints, structured fabrics",
        "maximumDistance": 0.1,
        "policy": "exclude candidate when the body-pattern distance exceeds the limit",
    }
