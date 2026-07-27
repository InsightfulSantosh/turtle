from __future__ import annotations

import math

import pytest

from domain.contracts import product_text
from fashion_matching.scoring import (
    SignalWeights,
    attribute_similarity,
    cosine_to_unit_interval,
    fuse_signals,
    l2_normalize,
)


def test_l2_embedding_normalization() -> None:
    vector = l2_normalize([3.0, 4.0])
    assert vector == pytest.approx([0.6, 0.8])
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1)
    with pytest.raises(ValueError, match="non-zero"):
        l2_normalize([0.0, 0.0])


def test_score_fusion_renormalizes_only_available_signals() -> None:
    score, applied = fuse_signals(
        {"image": 0.8, "attributes": 0.5, "text": None},
        SignalWeights(image=0.7, attributes=0.2, text=0.1),
    )
    assert applied == pytest.approx(
        {
            "image": 7 / 9,
            "attributes": 2 / 9,
        }
    )
    assert score == pytest.approx(0.8 * 7 / 9 + 0.5 * 2 / 9)
    assert "text" not in applied


def test_score_validation_and_attribute_matching() -> None:
    assert cosine_to_unit_interval(-1) == 0
    assert cosine_to_unit_interval(0) == 0.5
    assert cosine_to_unit_interval(1) == 1
    assert (
        attribute_similarity(
            {"colour": "navy blue", "fabric": "Cotton – 100%"},
            {"colour": "NAVY BLUE", "fabric": "Cotton - 100%"},
        )
        == 1
    )
    assert attribute_similarity({}, {"colour": "blue"}) is None
    with pytest.raises(ValueError, match="between 0 and 1"):
        fuse_signals(
            {"image": 1.1},
            SignalWeights(),
        )


def test_product_text_contains_only_descriptive_values() -> None:
    assert product_text({"id": "SECRET-SKU"}) == ""
    assert (
        product_text(
            {
                "id": "SECRET-SKU",
                "colour": "navy",
                "fabric": "cotton",
            }
        )
        == "cotton | navy"
    )
