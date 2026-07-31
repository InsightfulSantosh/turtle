"""Foreground-aware appearance signals for hybrid fashion retrieval.

The preferred production path is a text-guided garment box followed by SAM 2
segmentation. A conservative border-background mask remains as a secondary
fallback for clean studio imagery. When neither mask passes its quality gate,
colour and texture are deliberately unavailable rather than being calculated
from the original background.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from fashion_matching.garment_segmentation import GarmentSegmenter
from fashion_matching.taxonomy import colour_family

COLOUR_BINS_PER_CHANNEL = 8
COLOUR_DESCRIPTOR_DIMENSION = COLOUR_BINS_PER_CHANNEL**3
TEXTURE_ORIENTATION_BINS = 8
TEXTURE_ENERGY_BINS = 8
TEXTURE_DESCRIPTOR_DIMENSION = TEXTURE_ORIENTATION_BINS + TEXTURE_ENERGY_BINS
_MASK_WORKING_SIDE = 384
_WHITE_BACKGROUND = np.asarray([245, 245, 245], dtype=np.uint8)
_GARMENT_CROP_MARGIN = 0.08

# Preserve the source taxonomy while treating only clear spelling and plural
# variants as equivalent. A generic print is deliberately not treated as a
# digital, pigment, or discharge print.
_RETRIEVAL_DESIGN_ALIASES = {
    "PLAIN": "SOLID",
    "PLAINS": "SOLID",
    "SOLID": "SOLID",
    "SOLIDS": "SOLID",
    "PRINT": "PRINTS",
    "PRINTS": "PRINTS",
    "CHECK": "CHECKS",
    "CHECKS": "CHECKS",
    "STRIPE": "STRIPES",
    "STRIPES": "STRIPES",
    "PRINTS (PIGMENT)": "PIGMENT PRINT",
}

# These design families describe a visible repeat or surface construction where
# a broad label alone is too coarse.  For example, both a micro-check shirt and
# a large-windowpane shirt are `CHECKS` in the source data, but they must not
# be treated as visual equivalents.  The gate below compares the garment body
# at multiple scales before either item can become a final candidate.
_PATTERN_GATED_DESIGNS = frozenset(
    {
        "CHECKS",
        "STRIPES",
        "PRINTS",
        "DIGITAL PRINT",
        "PIGMENT PRINT",
        "DISCHARGE PRINT",
        "PRINTS (DISCHARGE & PIGMENT)",
        "DOBBY/STRUCTURE",
        "WBC DOBBY",
        "JACQUARD",
        "EMBOSSED",
        "HOUNDSTOOTH",
    }
)


def canonical_retrieval_value(value: Any) -> str:
    """Return a stable, non-empty value for strict retrieval constraints."""

    return " ".join(str(value or "").upper().split())


def canonical_retrieval_design(value: Any) -> str:
    """Return the strict source-design key used before visual retrieval."""

    design = canonical_retrieval_value(value)
    return _RETRIEVAL_DESIGN_ALIASES.get(design, design)


def requires_pattern_gate(value: Any) -> bool:
    """Whether a source design needs fine body-pattern verification."""

    return canonical_retrieval_design(value) in _PATTERN_GATED_DESIGNS


@dataclass(frozen=True)
class AppearanceFeatures:
    """Masked appearance features plus audit information for one image."""

    image: Image.Image
    colour_vector: np.ndarray | None
    texture_vector: np.ndarray | None
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
    row_indices, column_indices = np.where(mask)
    if not len(row_indices) or not len(column_indices):
        return image.copy()
    top, bottom = int(row_indices.min()), int(row_indices.max()) + 1
    left, right = int(column_indices.min()), int(column_indices.max()) + 1
    margin = max(round(max(bottom - top, right - left) * _GARMENT_CROP_MARGIN), 2)
    top = max(top - margin, 0)
    bottom = min(bottom + margin, image.height)
    left = max(left - margin, 0)
    right = min(right + margin, image.width)
    source = source[top:bottom, left:right]
    mask = mask[top:bottom, left:right]
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
    """Mask a garment and derive only foreground-backed appearance signals."""

    return _extract_appearance_features(
        image,
        mask_enabled=mask_enabled,
    )


def _extract_appearance_features(
    image: Image.Image,
    *,
    mask_enabled: bool,
    item_type: str = "",
    segmenter: GarmentSegmenter | None = None,
    allow_border_fallback: bool = True,
) -> AppearanceFeatures:
    """Implementation shared by the simple API and the pipeline integration."""

    source = image.convert("RGB")
    mask: np.ndarray | None = None
    confidence = 0.0
    reason: str | None = "masking_disabled" if not mask_enabled else None
    segmentation_method = "masking-disabled" if not mask_enabled else "fallback-original-image"
    if mask_enabled and segmenter is not None:
        semantic = segmenter.segment(source, item_type=item_type)
        mask = semantic.mask
        confidence = semantic.confidence
        reason = semantic.fallback_reason
        segmentation_method = semantic.method if mask is not None else "fallback-original-image"
    if mask_enabled and mask is None and allow_border_fallback:
        border_mask, border_confidence, border_reason = _foreground_mask(source)
        if border_mask is not None:
            mask = border_mask
            confidence = border_confidence
            reason = None
            segmentation_method = "adaptive-lab-border-foreground-mask"
        elif reason is None:
            reason = border_reason
    if mask is None:
        prepared_image = source.copy()
        mask_coverage = 0.0
        descriptor_mask = np.ones((source.height, source.width), dtype=bool) if not mask_enabled else None
    else:
        prepared_image = _masked_image(source, mask)
        mask_coverage = float(mask.mean())
        descriptor_mask = mask
    descriptor_image = _working_image(source)
    if descriptor_mask is not None and descriptor_image.size != source.size:
        descriptor_mask = np.asarray(
            Image.fromarray(descriptor_mask.astype(np.uint8) * 255).resize(
                descriptor_image.size,
                Image.Resampling.NEAREST,
            ),
            dtype=bool,
        )
    rgb = np.asarray(descriptor_image, dtype=np.uint8)
    colour_vector = _colour_descriptor(rgb, descriptor_mask) if descriptor_mask is not None else None
    texture_vector = _texture_descriptor(rgb, descriptor_mask) if descriptor_mask is not None else None
    return AppearanceFeatures(
        image=prepared_image,
        colour_vector=colour_vector,
        texture_vector=texture_vector,
        mask_coverage=round(mask_coverage, 4),
        mask_confidence=round(confidence, 4),
        segmentation_method=segmentation_method,
        fallback_reason=reason,
    )


def extract_pipeline_appearance_features(
    image: Image.Image,
    *,
    item_type: str,
    mask_enabled: bool,
    segmenter: GarmentSegmenter | None,
    allow_border_fallback: bool,
) -> AppearanceFeatures:
    """Pipeline entry point with an optional production garment segmenter."""

    return _extract_appearance_features(
        image,
        mask_enabled=mask_enabled,
        item_type=item_type,
        segmenter=segmenter,
        allow_border_fallback=allow_border_fallback,
    )


def body_pattern_views(image: Image.Image) -> list[Image.Image]:
    """Return large/medium/fine centre-body crops for pattern comparison."""

    source = image.convert("RGB")
    try:
        views: list[Image.Image] = []
        for fraction in (0.82, 0.66, 0.50):
            width, height = max(2, round(source.width * fraction)), max(2, round(source.height * fraction))
            left, top = (source.width - width) // 2, (source.height - height) // 2
            views.append(source.crop((left, top, left + width, top + height)))
        return views
    finally:
        source.close()


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
    """Metadata-aware FashionSigLIP retrieval with strict eligibility cohorts."""

    def __init__(self, candidates: list[dict[str, Any]], vectors: np.ndarray) -> None:
        if len(candidates) != len(vectors):
            raise ValueError("candidates and vectors must have the same length")
        self._global = InnerProductIndex(vectors)
        type_groups: dict[str, list[int]] = defaultdict(list)
        design_groups: dict[str, list[int]] = defaultdict(list)
        colour_groups: dict[str, list[int]] = defaultdict(list)
        type_design_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        type_colour_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        design_colour_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        type_design_colour_groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for index, item in enumerate(candidates):
            item_type = canonical_retrieval_value(item.get("itemType"))
            design = canonical_retrieval_design(item.get("design"))
            colour = colour_family(item.get("colour"))
            if item_type:
                type_groups[item_type].append(index)
            if design:
                design_groups[design].append(index)
            if colour:
                colour_groups[colour].append(index)
            if item_type and design:
                type_design_groups[(item_type, design)].append(index)
            if item_type and colour:
                type_colour_groups[(item_type, colour)].append(index)
            if design and colour:
                design_colour_groups[(design, colour)].append(index)
            if item_type and design and colour:
                type_design_colour_groups[(item_type, design, colour)].append(index)
        self._by_type = {
            item_type: (indices, InnerProductIndex(vectors[np.asarray(indices)]))
            for item_type, indices in type_groups.items()
        }
        self._by_design = {
            design: (indices, InnerProductIndex(vectors[np.asarray(indices)]))
            for design, indices in design_groups.items()
        }
        self._by_type_and_design = {
            key: (indices, InnerProductIndex(vectors[np.asarray(indices)]))
            for key, indices in type_design_groups.items()
        }
        self._by_type_and_colour = {
            key: (indices, InnerProductIndex(vectors[np.asarray(indices)]))
            for key, indices in type_colour_groups.items()
        }
        self._by_design_and_colour = {
            key: (indices, InnerProductIndex(vectors[np.asarray(indices)]))
            for key, indices in design_colour_groups.items()
        }
        self._by_colour = {
            colour: (indices, InnerProductIndex(vectors[np.asarray(indices)]))
            for colour, indices in colour_groups.items()
        }
        self._by_type_and_design_and_colour = {
            key: (indices, InnerProductIndex(vectors[np.asarray(indices)]))
            for key, indices in type_design_colour_groups.items()
        }
        self.backend = self._global.backend

    def search(
        self,
        query: dict[str, Any],
        vector: np.ndarray,
        limit: int,
        *,
        require_same_item_type: bool,
        require_same_design: bool,
        require_same_colour_family: bool,
    ) -> list[int]:
        item_type = canonical_retrieval_value(query.get("itemType"))
        design = canonical_retrieval_design(query.get("design"))
        colour = colour_family(query.get("colour"))
        selected: tuple[list[int], InnerProductIndex] | None
        if require_same_item_type and require_same_design and require_same_colour_family:
            selected = (
                self._by_type_and_design_and_colour.get((item_type, design, colour))
                if item_type and design and colour
                else None
            )
        elif require_same_item_type and require_same_design:
            selected = self._by_type_and_design.get((item_type, design)) if item_type and design else None
        elif require_same_item_type and require_same_colour_family:
            selected = self._by_type_and_colour.get((item_type, colour)) if item_type and colour else None
        elif require_same_design and require_same_colour_family:
            selected = self._by_design_and_colour.get((design, colour)) if design and colour else None
        elif require_same_item_type:
            selected = self._by_type.get(item_type) if item_type else None
        elif require_same_design:
            selected = self._by_design.get(design) if design else None
        elif require_same_colour_family:
            selected = self._by_colour.get(colour) if colour else None
        else:
            return self._global.search(vector, limit)
        if selected is not None:
            source_indices, index = selected
            return [source_indices[position] for position in index.search(vector, limit)]
        # A required attribute with no eligible history is a true no-match,
        # never a reason to fall back to another item type or design.
        return []
