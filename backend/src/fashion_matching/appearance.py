"""Foreground-aware appearance signals for hybrid fashion retrieval.

The catalogue does not yet carry pixel masks, so this module deliberately uses
a conservative border-background segmenter rather than pretending every image
has a perfect garment cut-out.  It only replaces the image background when the
mask is credible; otherwise all encoders and descriptors fall back to the
original image.  That makes the pipeline safe for editorial, modelled and
non-studio catalogue photos while improving plain-background product shots.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

COLOUR_BINS_PER_CHANNEL = 8
COLOUR_DESCRIPTOR_DIMENSION = COLOUR_BINS_PER_CHANNEL**3
TEXTURE_ORIENTATION_BINS = 8
TEXTURE_ENERGY_BINS = 8
TEXTURE_DESCRIPTOR_DIMENSION = TEXTURE_ORIENTATION_BINS + TEXTURE_ENERGY_BINS
_MASK_WORKING_SIDE = 384
_WHITE_BACKGROUND = np.asarray([245, 245, 245], dtype=np.uint8)


@dataclass(frozen=True)
class AppearanceFeatures:
    """Masked appearance features plus audit information for one image."""

    image: Image.Image
    colour_vector: np.ndarray
    texture_vector: np.ndarray
    mask_coverage: float
    mask_confidence: float
    segmentation_method: str
    fallback_reason: str | None = None

    @property
    def masked(self) -> bool:
        return self.fallback_reason is None


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1e-12 else values


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine distance with an explicit, deterministic zero-vector policy."""

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 and right_norm <= 1e-12:
        return 0.0
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 1.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 255] to CIE Lab (D65) without OpenCV."""

    srgb = np.asarray(rgb, dtype=np.float32) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    xyz = (
        linear
        @ np.asarray(
            (
                (0.4124564, 0.3575761, 0.1804375),
                (0.2126729, 0.7151522, 0.0721750),
                (0.0193339, 0.1191920, 0.9503041),
            ),
            dtype=np.float32,
        ).T
    )
    xyz /= np.asarray((0.95047, 1.0, 1.08883), dtype=np.float32)
    epsilon = 216 / 24389
    kappa = 24389 / 27
    transformed = np.where(
        xyz > epsilon,
        np.cbrt(xyz),
        (kappa * xyz + 16) / 116,
    )
    lightness = 116 * transformed[..., 1] - 16
    a_channel = 500 * (transformed[..., 0] - transformed[..., 1])
    b_channel = 200 * (transformed[..., 1] - transformed[..., 2])
    return np.stack((lightness, a_channel, b_channel), axis=-1)


def _border_pixels(values: np.ndarray) -> np.ndarray:
    height, width = values.shape[:2]
    thickness = max(1, min(height, width) // 40)
    return np.concatenate(
        (
            values[:thickness].reshape(-1, values.shape[-1]),
            values[-thickness:].reshape(-1, values.shape[-1]),
            values[thickness:-thickness, :thickness].reshape(-1, values.shape[-1]),
            values[thickness:-thickness, -thickness:].reshape(-1, values.shape[-1]),
        )
    )


def _majority_mask(mask: np.ndarray) -> np.ndarray:
    """Remove isolated pixels and close one-pixel holes without scipy."""

    padded = np.pad(mask.astype(np.uint8), 1)
    neighbours = sum(
        padded[row : row + mask.shape[0], column : column + mask.shape[1]] for row in range(3) for column in range(3)
    )
    return neighbours >= 5


def _working_image(image: Image.Image) -> Image.Image:
    longest_side = max(image.size)
    if longest_side <= _MASK_WORKING_SIDE:
        return image
    scale = _MASK_WORKING_SIDE / longest_side
    size = (max(round(image.width * scale), 1), max(round(image.height * scale), 1))
    return image.resize(size, Image.Resampling.BILINEAR)


def _foreground_mask(image: Image.Image) -> tuple[np.ndarray | None, float, str | None]:
    """Return a reliable foreground mask or an explicit fallback reason."""

    working = _working_image(image)
    rgb = np.asarray(working, dtype=np.uint8)
    lab = _rgb_to_lab(rgb)
    border = _border_pixels(lab)
    background = np.median(border, axis=0)
    border_distances = np.linalg.norm(border - background, axis=1)
    border_spread = float(np.percentile(border_distances, 75))
    if border_spread > 12.0:
        return None, 0.0, "non_uniform_border"

    threshold = max(14.0, float(np.percentile(border_distances, 95)) + 8.0)
    raw_mask = np.linalg.norm(lab - background, axis=2) > threshold
    mask = _majority_mask(raw_mask)
    coverage = float(mask.mean())
    if not 0.05 <= coverage <= 0.90:
        return None, coverage, "implausible_foreground_coverage"

    height, width = mask.shape
    centre = mask[height // 4 : max(3 * height // 4, height // 4 + 1), width // 4 : max(3 * width // 4, width // 4 + 1)]
    centre_coverage = float(centre.mean()) if centre.size else 0.0
    if centre_coverage < 0.08:
        return None, coverage, "foreground_not_central"

    background_confidence = float(np.clip(1.0 - border_spread / 12.0, 0.0, 1.0))
    coverage_confidence = float(np.clip(1.0 - abs(coverage - 0.42) / 0.55, 0.0, 1.0))
    centre_confidence = float(np.clip(centre_coverage / 0.40, 0.0, 1.0))
    confidence = 0.55 * background_confidence + 0.30 * coverage_confidence + 0.15 * centre_confidence
    if confidence < 0.60:
        return None, confidence, "low_mask_confidence"

    if working.size != image.size:
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.asarray(mask_image.resize(image.size, Image.Resampling.NEAREST), dtype=bool)
    return mask, confidence, None


def _masked_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    source = np.asarray(image, dtype=np.uint8)
    composite = np.where(mask[..., np.newaxis], source, _WHITE_BACKGROUND)
    return Image.fromarray(composite.astype(np.uint8))


def _colour_descriptor(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = _rgb_to_lab(rgb)[mask]
    if pixels.size == 0:
        return np.zeros(COLOUR_DESCRIPTOR_DIMENSION, dtype=np.float32)
    lightness = np.clip(pixels[:, 0] / 100 * COLOUR_BINS_PER_CHANNEL, 0, COLOUR_BINS_PER_CHANNEL - 1).astype(int)
    a_channel = np.clip(
        (pixels[:, 1] + 128) / 255 * COLOUR_BINS_PER_CHANNEL,
        0,
        COLOUR_BINS_PER_CHANNEL - 1,
    ).astype(int)
    b_channel = np.clip(
        (pixels[:, 2] + 128) / 255 * COLOUR_BINS_PER_CHANNEL,
        0,
        COLOUR_BINS_PER_CHANNEL - 1,
    ).astype(int)
    flat_indices = lightness * COLOUR_BINS_PER_CHANNEL**2 + a_channel * COLOUR_BINS_PER_CHANNEL + b_channel
    histogram = np.bincount(flat_indices, minlength=COLOUR_DESCRIPTOR_DIMENSION)
    return _l2_normalize(histogram.astype(np.float32))


def _texture_descriptor(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = values[..., 0] * 0.299 + values[..., 1] * 0.587 + values[..., 2] * 0.114
    padded = np.pad(gray, 1, mode="edge")
    gradient_x = padded[1:-1, 2:] - padded[1:-1, :-2]
    gradient_y = padded[2:, 1:-1] - padded[:-2, 1:-1]
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = (np.arctan2(gradient_y, gradient_x) + np.pi) % np.pi
    valid = mask & (magnitude > 1e-4)
    orientation_bins = np.minimum(
        (orientation[valid] / np.pi * TEXTURE_ORIENTATION_BINS).astype(int),
        TEXTURE_ORIENTATION_BINS - 1,
    )
    orientation_histogram = np.bincount(
        orientation_bins,
        weights=magnitude[valid],
        minlength=TEXTURE_ORIENTATION_BINS,
    ).astype(np.float32)

    local_mean = (
        sum(
            padded[row : row + gray.shape[0], column : column + gray.shape[1]]
            for row in range(3)
            for column in range(3)
        )
        / 9.0
    )
    local_energy = np.abs(gray - local_mean)
    energy_bins = np.minimum(
        (local_energy[mask] / 0.25 * TEXTURE_ENERGY_BINS).astype(int),
        TEXTURE_ENERGY_BINS - 1,
    )
    energy_histogram = np.bincount(energy_bins, minlength=TEXTURE_ENERGY_BINS).astype(np.float32)
    return _l2_normalize(np.concatenate((orientation_histogram, energy_histogram)))


def extract_appearance_features(image: Image.Image, *, mask_enabled: bool = True) -> AppearanceFeatures:
    """Mask a garment conservatively, then derive colour and texture signals."""

    source = image.convert("RGB")
    mask, confidence, reason = _foreground_mask(source) if mask_enabled else (None, 0.0, "masking_disabled")
    if mask is None:
        mask = np.ones((source.height, source.width), dtype=bool)
        prepared_image = source.copy()
        segmentation_method = "fallback-original-image" if mask_enabled else "masking-disabled"
        mask_coverage = 1.0
    else:
        prepared_image = _masked_image(source, mask)
        segmentation_method = "adaptive-lab-border-foreground-mask"
        mask_coverage = float(mask.mean())
    descriptor_image = _working_image(source)
    if descriptor_image.size == source.size:
        descriptor_mask = mask
    else:
        descriptor_mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                descriptor_image.size,
                Image.Resampling.NEAREST,
            ),
            dtype=bool,
        )
    rgb = np.asarray(descriptor_image, dtype=np.uint8)
    return AppearanceFeatures(
        image=prepared_image,
        colour_vector=_colour_descriptor(rgb, descriptor_mask),
        texture_vector=_texture_descriptor(rgb, descriptor_mask),
        mask_coverage=round(mask_coverage, 4),
        mask_confidence=round(confidence, 4),
        segmentation_method=segmentation_method,
        fallback_reason=reason,
    )


class InnerProductIndex:
    """Exact inner-product retrieval backed by FAISS when the runtime provides it."""

    def __init__(self, vectors: np.ndarray) -> None:
        normalised = np.asarray(vectors, dtype=np.float32)
        if normalised.ndim != 2:
            raise ValueError("candidate vectors must be a two-dimensional array")
        self.vectors = normalised
        self.backend = "numpy-exact-inner-product"
        self._index: Any | None = None
        if normalised.size == 0:
            return
        try:
            import faiss
        except ImportError:
            return
        self._index = faiss.IndexFlatIP(normalised.shape[1])
        self._index.add(normalised)
        self.backend = "faiss-indexflatip"

    def search(self, query: np.ndarray, limit: int) -> list[int]:
        if limit <= 0 or not len(self.vectors):
            return []
        count = min(limit, len(self.vectors))
        vector = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self._index is not None:
            _, indices = self._index.search(vector, count)
            return [int(index) for index in indices[0] if index >= 0]
        scores = self.vectors @ vector[0]
        return [int(index) for index in np.argsort(-scores, kind="stable")[:count]]


class FashionCandidateRetriever:
    """Metadata-aware FashionSigLIP retrieval with global sparse-type fallback."""

    def __init__(self, candidates: list[dict[str, Any]], vectors: np.ndarray) -> None:
        if len(candidates) != len(vectors):
            raise ValueError("candidates and vectors must have the same length")
        self._global = InnerProductIndex(vectors)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(candidates):
            item_type = " ".join(str(item.get("itemType") or "").upper().split())
            if item_type:
                groups[item_type].append(index)
        self._by_type = {
            item_type: (indices, InnerProductIndex(vectors[np.asarray(indices)]))
            for item_type, indices in groups.items()
        }
        self.backend = self._global.backend

    def search(
        self,
        query: dict[str, Any],
        vector: np.ndarray,
        limit: int,
        *,
        require_same_item_type: bool,
    ) -> list[int]:
        item_type = " ".join(str(query.get("itemType") or "").upper().split())
        selected = self._by_type.get(item_type) if require_same_item_type and item_type else None
        if selected is not None and len(selected[0]) >= 2:
            source_indices, index = selected
            return [source_indices[position] for position in index.search(vector, limit)]
        return self._global.search(vector, limit)
