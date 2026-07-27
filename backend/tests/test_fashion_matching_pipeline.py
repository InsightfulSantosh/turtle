from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from fashion_matching.evaluation import evaluate_results
from fashion_matching.indexing import CatalogueIndexer
from fashion_matching.matching import FashionMatcher
from fashion_matching.models import ManifestRecord
from fashion_matching.preprocessing import ImagePreprocessor
from fashion_matching.scoring import SignalWeights, l2_normalize
from fashion_matching.storage import InMemoryVectorStore, QdrantVectorStore, VectorStoreError


class LightweightEncoder:
    model_id = "test/lightweight-fashion"
    revision = "0123456789abcdef"
    dimension = 4
    supports_text = True

    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def encode_images(
        self,
        images: Sequence[Image.Image],
    ) -> list[list[float]]:
        vectors = []
        for image in images:
            red, green, blue = image.resize((1, 1)).getpixel((0, 0))
            vectors.append(
                l2_normalize(
                    [
                        red / 255,
                        green / 255,
                        blue / 255,
                        0.2,
                    ]
                )
            )
        return vectors

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.seen_texts.extend(texts)
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append(
                l2_normalize(
                    [
                        digest[0] + 1,
                        digest[1] + 1,
                        digest[2] + 1,
                        digest[3] + 1,
                    ]
                )
            )
        return vectors


def _image(tmp_path: Path, name: str, colour: str) -> Path:
    path = tmp_path / name
    Image.new("RGB", (64, 48), colour).save(path)
    return path


def _record(
    product_id: str,
    image_id: str,
    path: Path,
    *,
    text: str | None = None,
    colour: str | None = None,
) -> ManifestRecord:
    return ManifestRecord(
        product_id=product_id,
        image_id=image_id,
        image_path=path,
        text=text,
        attributes={"colour": colour} if colour else {},
    )


def test_idempotent_index_and_product_level_matching(tmp_path: Path) -> None:
    encoder = LightweightEncoder()
    store = InMemoryVectorStore()
    preprocessor = ImagePreprocessor()
    records = [
        _record(
            "RED-PRODUCT",
            "RED-FRONT",
            _image(tmp_path, "red-front.png", "red"),
            text="Cotton red shirt",
            colour="red",
        ),
        _record(
            "RED-PRODUCT",
            "RED-BACK",
            _image(tmp_path, "red-back.png", "#dd0000"),
            text="Cotton red shirt back view",
            colour="red",
        ),
        _record(
            "BLUE-PRODUCT",
            "BLUE-FRONT",
            _image(tmp_path, "blue.png", "blue"),
            text="Blue formal shirt",
            colour="blue",
        ),
    ]
    indexer = CatalogueIndexer(
        encoder=encoder,
        preprocessor=preprocessor,
        store=store,
        collection_prefix="test",
        batch_size=2,
    )
    first = indexer.index(records, activate_alias="active")
    assert first.successfully_indexed == 3
    assert first.failed_images == 0
    second = indexer.index(records)
    assert second.skipped_images == 3
    assert second.successfully_indexed == 0

    query = _record(
        "QUERY-ID-MUST-NOT-BE-TEXT",
        "QUERY-IMAGE",
        _image(tmp_path, "query.png", "#f00000"),
        text="Red casual top",
        colour="red",
    )
    result = FashionMatcher(
        encoder=encoder,
        preprocessor=preprocessor,
        store=store,
        collection="active",
        weights=SignalWeights(image=0.8, attributes=0.2, text=0.0),
        top_k=5,
    ).match(query)
    assert result.error is None
    assert result.matches[0].product_id == "RED-PRODUCT"
    assert len([match for match in result.matches if match.product_id == "RED-PRODUCT"]) == 1
    assert "QUERY-ID-MUST-NOT-BE-TEXT" not in encoder.seen_texts
    assert "Red casual top" in encoder.seen_texts


def test_minimum_score_and_incompatible_model_protection(
    tmp_path: Path,
) -> None:
    encoder = LightweightEncoder()
    store = InMemoryVectorStore()
    preprocessor = ImagePreprocessor()
    record = _record(
        "P1",
        "I1",
        _image(tmp_path, "one.png", "red"),
    )
    indexer = CatalogueIndexer(
        encoder=encoder,
        preprocessor=preprocessor,
        store=store,
        collection_prefix="test",
    )
    indexer.index([record], activate_alias="active")
    no_match = FashionMatcher(
        encoder=encoder,
        preprocessor=preprocessor,
        store=store,
        collection="active",
        minimum_score=1.0,
    ).match(_record("Q", "QI", _image(tmp_path, "different.png", "green")))
    assert no_match.no_suitable_match

    point = next(iter(store.collections[indexer.collection].points.values()))
    store.collections[indexer.collection].points[point.point_id] = type(point)(
        point_id=point.point_id,
        vectors=point.vectors,
        payload={**point.payload, "model_version": "wrong-version"},
    )
    incompatible = FashionMatcher(
        encoder=encoder,
        preprocessor=preprocessor,
        store=store,
        collection="active",
    ).match(record)
    assert "incompatible model" in (incompatible.error or "")


def test_dimension_validation() -> None:
    store = InMemoryVectorStore()
    store.ensure_collection("catalog", {"image": 4})
    with pytest.raises(VectorStoreError, match="incompatible"):
        store.search("catalog", "image", [1.0, 0.0], 10)


def test_qdrant_named_vectors_and_alias() -> None:
    store = QdrantVectorStore(url=":memory:")
    store.ensure_collection("catalog-v1", {"image": 4, "text": 4})
    from fashion_matching.models import VectorPoint

    point = VectorPoint(
        point_id="c53d30a7-f09f-4e75-82aa-c2b870bf5589",
        vectors={
            "image": [1.0, 0.0, 0.0, 0.0],
            "text": [0.0, 1.0, 0.0, 0.0],
        },
        payload={"product_id": "P1", "image_id": "I1"},
    )
    store.upsert("catalog-v1", [point])
    store.activate_alias("catalog-v1", "active")
    hits = store.search("active", "image", [1.0, 0.0, 0.0, 0.0], 5)
    assert hits[0].point_id == point.point_id
    assert hits[0].score == pytest.approx(1)
    restored = store.get_point("active", point.point_id)
    assert restored is not None
    assert restored.payload["product_id"] == "P1"


def test_evaluation_metrics() -> None:
    metrics = evaluate_results(
        [
            {
                "query_image_id": "Q1",
                "matches": [
                    {"product_id": "WRONG"},
                    {"product_id": "RIGHT"},
                ],
                "no_suitable_match": False,
                "processing_time_ms": 10,
            },
            {
                "query_image_id": "Q2",
                "matches": [],
                "no_suitable_match": True,
                "processing_time_ms": 20,
            },
        ],
        [
            {
                "query_image_id": "Q1",
                "relevant_product_id": "RIGHT",
                "relevance": "2",
                "no_match": "false",
            },
            {
                "query_image_id": "Q2",
                "relevant_product_id": "",
                "relevance": "",
                "no_match": "true",
            },
        ],
    )
    assert metrics["recall_at_1"] == 0
    assert metrics["recall_at_3"] == 1
    assert metrics["mrr"] == 0.5
    assert metrics["no_match_accuracy"] == 1
    assert metrics["p50_latency_ms"] == 15
