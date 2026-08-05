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
    assert output["reranker"]["sameDesignConstraint"] is False
    assert "sameColourFamilyConstraint" not in output["reranker"]
    assert output["reranker"]["visualOnlyRanking"] is True
    assert output["reranker"]["appearance"]["colourDescriptor"]["space"] == "CIELAB"
    assert output["reranker"]["appearance"]["colourDescriptor"]["method"] == (
        "dominant-palette-CIEDE2000"
    )
    assert output["reranker"]["appearance"]["weights"] == {
        "neural": 0.45,
        "colour": 0.45,
        "texture": 0.1,
    }
    upcoming_candidates = [row for row in output["candidatePairs"] if row["leftId"] == "OTSH-200-1001"]
    assert [row["candidateRank"] for row in upcoming_candidates] == [1, 2]
    assert {row["rightId"] for row in upcoming_candidates} == {
        "AW25-OTSH-100-1001",
        "AW25-OTSH-102-1001",
    }
    assert all("colourDistance" in row and "textureDistance" in row for row in upcoming_candidates)
    assert all(row["colourDistance"] is not None and row["textureDistance"] is not None for row in upcoming_candidates)
    appearance = output["reranker"]["appearance"]
    assert "segmentation" not in appearance
    assert appearance["colourDescriptor"]["fullImage"] is False
    assert appearance["colourDescriptor"]["paletteSize"] == 4
    assert appearance["colourGate"] == {
        "enabled": True,
        "maximumDistance": 0.2,
        "maximumDeltaE": 10.0,
        "policy": "exclude candidate when garment-palette perceptual ΔE exceeds the limit",
    }
    assert output["reranker"]["gateAudit"]["upcoming"]["retrievedCandidates"] == 2
    assert output["reranker"]["gateAudit"]["upcoming"]["acceptedCandidates"] == 2


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
        colour_gate_enabled=False,
        pattern_gate_enabled=True,
        pattern_max_distance=0.1,
    )

    upcoming_candidates = [row for row in output["candidatePairs"] if row["leftId"] == "OTSH-200-1001"]
    assert [row["rightId"] for row in upcoming_candidates] == ["AW25-OTSH-100-1001"]
    assert upcoming_candidates[0]["patternDistance"] < 1e-6
    assert output["reranker"]["patternGate"] == {
        "enabled": True,
        "method": "three-scale-centre-garment-body-DINOv2",
        "scope": (
            "all visually retrieved candidates except OTTR; OTTR hard-gates only "
            "CHECKS, PRINTS, STRIPES and DOBBY/STRUCTURE"
        ),
        "maximumDistance": 0.1,
        "policy": "exclude candidate when the body-pattern distance exceeds the limit",
    }


def test_plain_ottr_bypasses_hard_pattern_gate_but_retains_dino_score(
    tmp_path: Path,
) -> None:
    settings = PipelineSettings(
        data_root=tmp_path / "DATA",
        temporary_root=tmp_path / "tmp",
        output_path=tmp_path / "artifact.json",
    )
    _save_image(settings.historical_image_root / "OTTR-100-1001.jpg", (255, 0, 0))
    _save_image(settings.upcoming_image_root / "OTTR-200-1001.jpg", (0, 0, 255))
    source = {
        "historical": [
            {
                "id": "AW25-OTTR-100-1001",
                "sourceId": "OTTR-100-1001",
                "itemType": "OTTR",
                "design": "PLAINS",
                "imageUrl": "/product-images/historical/OTTR-100-1001",
            }
        ],
        "upcoming": [
            {
                "id": "OTTR-200-1001",
                "itemType": "OTTR",
                "design": "PLAINS",
                "imageUrl": "/product-images/upcoming/OTTR-200-1001",
            }
        ],
    }

    output = build_artifact_vision_output(
        source,
        settings=settings,
        encoder=FakeEncoder(),
        reranker=FakeDetailEncoder(),
        preprocessor=ImagePreprocessor(pad_to_square=False),
        candidate_count=1,
        colour_gate_enabled=False,
        pattern_gate_enabled=True,
        pattern_max_distance=0.1,
    )

    candidate = next(
        row for row in output["candidatePairs"] if row["leftId"] == "OTTR-200-1001"
    )
    assert candidate["patternDistance"] > 0.1
    assert candidate["dinoDistance"] == candidate["patternDistance"]
    audit = output["reranker"]["gateAudit"]["upcoming"]
    assert audit["patternGateApplied"] == 0
    assert audit["patternGateBypassed"] == 1
    assert audit["patternRejected"] == 0
    assert output["reranker"]["appearance"]["itemTypeOverrides"]["OTTR"] == {
        "analysisRegion": "waist-to-lower-leg trouser ROI excluding footwear",
        "relativeBox": [0.16, 0.28, 0.84, 0.8],
        "usedFor": [
            "FashionSigLIP retrieval",
            "global DINO detail",
            "multi-scale DINO pattern detail",
            "dominant colour palette",
            "texture descriptor",
        ],
        "displayedImageModified": False,
        "patternHardGateDesigns": ["CHECKS", "PRINTS", "STRIPES", "DOBBY/STRUCTURE"],
    }


def test_colour_gate_uses_image_colour_and_ignores_colour_labels(tmp_path: Path) -> None:
    settings = PipelineSettings(
        data_root=tmp_path / "DATA",
        temporary_root=tmp_path / "tmp",
        output_path=tmp_path / "artifact.json",
    )
    _save_image(settings.historical_image_root / "OTSH-100-1001.jpg", (255, 0, 0))
    _save_image(settings.historical_image_root / "OTSH-101-1001.jpg", (0, 0, 255))
    _save_image(settings.upcoming_image_root / "OTSH-200-1001.jpg", (255, 0, 0))
    source = {
        "historical": [
            {
                "id": "AW25-OTSH-100-1001",
                "sourceId": "OTSH-100-1001",
                "itemType": "OTSH",
                "design": "PLAINS",
                "colour": "BLUE",
                "imageUrl": "/product-images/historical/OTSH-100-1001",
            },
            {
                "id": "AW25-OTSH-101-1001",
                "sourceId": "OTSH-101-1001",
                "itemType": "OTSH",
                "design": "PLAINS",
                "colour": "RED",
                "imageUrl": "/product-images/historical/OTSH-101-1001",
            },
        ],
        "upcoming": [
            {
                "id": "OTSH-200-1001",
                "itemType": "OTSH",
                "design": "PLAINS",
                "colour": "GREEN",
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
        pattern_gate_enabled=False,
        colour_gate_enabled=True,
        colour_max_distance=0.2,
    )

    candidates = [row for row in output["candidatePairs"] if row["leftId"] == "OTSH-200-1001"]
    assert [row["rightId"] for row in candidates] == ["AW25-OTSH-100-1001"]
    assert candidates[0]["colourDistance"] < 1e-6
    assert candidates[0]["colourDeltaE"] < 1e-6
    assert output["reranker"]["gateAudit"]["upcoming"]["colourRejected"] == 1
