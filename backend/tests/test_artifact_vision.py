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
