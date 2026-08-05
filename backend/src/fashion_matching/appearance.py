"""Garment-focused appearance signals for visual-only fashion retrieval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

COLOUR_PALETTE_SIZE = 4
COLOUR_PALETTE_COMPONENTS = 4  # Lab centre + cluster weight
COLOUR_DESCRIPTOR_DIMENSION = COLOUR_PALETTE_SIZE * COLOUR_PALETTE_COMPONENTS
COLOUR_DELTA_E_SCALE = 50.0
TEXTURE_ORIENTATION_BINS = 8
TEXTURE_ENERGY_BINS = 8
TEXTURE_DESCRIPTOR_DIMENSION = TEXTURE_ORIENTATION_BINS + TEXTURE_ENERGY_BINS
_DESCRIPTOR_WORKING_SIDE = 384
# Stable waist-to-lower-leg crop for OTTR catalogue photography.  The SS27
# images usually place footwear in the bottom ~18% of the frame, while the
# historical catalogue uses trouser-only composites.  Ending at 80% keeps the
# trouser silhouette and fabric but removes the footwear mismatch.
OTTR_TROUSER_ROI = (0.16, 0.28, 0.84, 0.80)

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


def requires_ottr_pattern_gate(value: Any) -> bool:
    """Limit OTTR hard rejection to designs with visible repeat/texture."""

    return canonical_retrieval_design(value) in {
        "CHECKS",
        "STRIPES",
        "PRINTS",
        "DOBBY/STRUCTURE",
    }


@dataclass(frozen=True)
class AppearanceFeatures:
    """Appearance features calculated without changing the displayed image."""

    image: Image.Image
    colour_vector: np.ndarray
    texture_vector: np.ndarray


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


def _working_image(image: Image.Image) -> Image.Image:
    longest_side = max(image.size)
    if longest_side <= _DESCRIPTOR_WORKING_SIDE:
        return image
    scale = _DESCRIPTOR_WORKING_SIDE / longest_side
    size = (max(round(image.width * scale), 1), max(round(image.height * scale), 1))
    return image.resize(size, Image.Resampling.BILINEAR)


def visual_analysis_region(image: Image.Image, item_type: Any = None) -> Image.Image:
    """Return the non-destructive region used for detailed visual analysis.

    OTTR catalogue layouts frequently place swatches, logos, shirts, hands and
    shoes around the trouser. A stable waist-to-lower-leg crop retains the
    waistband, seat and trouser silhouette while excluding footwear and most
    other distractors. Other item types deliberately keep the established
    full-image analysis path.
    """

    source = image.convert("RGB")
    if canonical_retrieval_value(item_type) != "OTTR":
        return source
    left, top, right, bottom = OTTR_TROUSER_ROI
    box = (
        round(source.width * left),
        round(source.height * top),
        max(round(source.width * right), round(source.width * left) + 2),
        max(round(source.height * bottom), round(source.height * top) + 2),
    )
    try:
        return source.crop(box)
    finally:
        source.close()


def _garment_body_rgb(image: Image.Image, item_type: Any = None) -> np.ndarray:
    """Return foreground pixels from the central garment body.

    Catalogue photography in this project is predominantly centred on white or
    near-white backgrounds.  The crop removes logos, fabric swatches, faces and
    most lower-body styling; border-colour suppression then removes the studio
    background.  This is an internal measurement only and never changes the
    product image served by the application.
    """

    analysis_region = visual_analysis_region(image, item_type)
    working = _working_image(analysis_region)
    rgb = np.asarray(working, dtype=np.uint8)
    height, width = rgb.shape[:2]
    left, right = round(width * 0.14), round(width * 0.86)
    top, bottom = round(height * 0.16), round(height * 0.90)
    roi = rgb[top:max(bottom, top + 1), left:max(right, left + 1)]
    roi_lab = _rgb_to_lab(roi)

    border_width = max(1, round(min(height, width) * 0.035))
    border = np.concatenate(
        (
            rgb[:border_width].reshape(-1, 3),
            rgb[-border_width:].reshape(-1, 3),
            rgb[:, :border_width].reshape(-1, 3),
            rgb[:, -border_width:].reshape(-1, 3),
        ),
        axis=0,
    )
    background_lab = np.median(_rgb_to_lab(border), axis=0)
    background_distance = np.linalg.norm(roi_lab - background_lab, axis=-1)
    chroma = np.linalg.norm(roi_lab[..., 1:], axis=-1)
    near_white = (roi_lab[..., 0] >= 94.0) & (chroma <= 10.0)
    foreground = (background_distance >= 9.0) & ~near_white

    # Avoid an unstable descriptor when the source has a non-standard studio
    # background.  The central crop is still safer than the full image.
    if int(foreground.sum()) < max(round(foreground.size * 0.04), 32):
        foreground = ~near_white
    pixels = roi[foreground]
    try:
        return pixels if len(pixels) else roi.reshape(-1, 3)
    finally:
        if working is not analysis_region:
            working.close()
        analysis_region.close()


def _dominant_colour_palette(rgb: np.ndarray) -> np.ndarray:
    """Return four deterministic dominant Lab colours and their proportions."""

    pixels = _rgb_to_lab(rgb).reshape(-1, 3).astype(np.float32)
    if len(pixels) > 6_000:
        indices = np.linspace(0, len(pixels) - 1, 6_000, dtype=int)
        pixels = pixels[indices]
    if not len(pixels):
        return np.zeros(COLOUR_DESCRIPTOR_DIMENSION, dtype=np.float32)

    # Seed Lloyd clustering from frequent quantised colours, then add centres
    # that maximise perceptual separation. This is deterministic across runs.
    quantisation = np.asarray((4.0, 6.0, 6.0), dtype=np.float32)
    quantised = np.round(pixels / quantisation).astype(np.int16)
    cells, inverse, counts = np.unique(
        quantised,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    cell_centres = np.stack(
        [pixels[inverse == index].mean(axis=0) for index in range(len(cells))]
    )
    chosen = [int(np.argmax(counts))]
    while len(chosen) < min(COLOUR_PALETTE_SIZE, len(cell_centres)):
        minimum_distance = np.min(
            np.linalg.norm(
                cell_centres[:, None, :] - cell_centres[np.asarray(chosen)][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        score = minimum_distance**2 * np.sqrt(counts)
        score[np.asarray(chosen)] = -1.0
        chosen.append(int(np.argmax(score)))
    centres = cell_centres[np.asarray(chosen)].astype(np.float32)

    for _ in range(12):
        distances = np.linalg.norm(pixels[:, None, :] - centres[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        updated = centres.copy()
        for index in range(len(centres)):
            members = pixels[assignments == index]
            if len(members):
                updated[index] = members.mean(axis=0)
        if np.allclose(updated, centres, atol=0.05):
            centres = updated
            break
        centres = updated

    distances = np.linalg.norm(pixels[:, None, :] - centres[None, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    weights = np.asarray(
        [(assignments == index).mean() for index in range(len(centres))],
        dtype=np.float32,
    )
    order = np.argsort(-weights, kind="stable")
    centres = centres[order]
    weights = weights[order]
    descriptor = np.zeros((COLOUR_PALETTE_SIZE, COLOUR_PALETTE_COMPONENTS), dtype=np.float32)
    descriptor[: len(centres), :3] = centres
    descriptor[: len(weights), 3] = weights
    return descriptor.reshape(-1)


def delta_e_ciede2000(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 perceptual colour difference."""

    left_lab = np.asarray(left, dtype=np.float64)
    right_lab = np.asarray(right, dtype=np.float64)
    left_l, left_a, left_b = np.moveaxis(left_lab, -1, 0)
    right_l, right_a, right_b = np.moveaxis(right_lab, -1, 0)
    left_c = np.hypot(left_a, left_b)
    right_c = np.hypot(right_a, right_b)
    mean_c = (left_c + right_c) / 2.0
    adjustment = 0.5 * (
        1.0 - np.sqrt(mean_c**7 / np.maximum(mean_c**7 + 25.0**7, 1e-12))
    )
    left_a_prime = (1.0 + adjustment) * left_a
    right_a_prime = (1.0 + adjustment) * right_a
    left_c_prime = np.hypot(left_a_prime, left_b)
    right_c_prime = np.hypot(right_a_prime, right_b)
    left_h_prime = np.mod(np.degrees(np.arctan2(left_b, left_a_prime)), 360.0)
    right_h_prime = np.mod(np.degrees(np.arctan2(right_b, right_a_prime)), 360.0)

    delta_l = right_l - left_l
    delta_c = right_c_prime - left_c_prime
    hue_difference = right_h_prime - left_h_prime
    hue_difference = np.where(hue_difference > 180.0, hue_difference - 360.0, hue_difference)
    hue_difference = np.where(hue_difference < -180.0, hue_difference + 360.0, hue_difference)
    hue_difference = np.where(left_c_prime * right_c_prime == 0.0, 0.0, hue_difference)
    delta_h = 2.0 * np.sqrt(left_c_prime * right_c_prime) * np.sin(
        np.radians(hue_difference) / 2.0
    )

    mean_l = (left_l + right_l) / 2.0
    mean_c_prime = (left_c_prime + right_c_prime) / 2.0
    hue_sum = left_h_prime + right_h_prime
    mean_h = np.where(
        left_c_prime * right_c_prime == 0.0,
        hue_sum,
        np.where(
            np.abs(left_h_prime - right_h_prime) <= 180.0,
            hue_sum / 2.0,
            np.where(hue_sum < 360.0, (hue_sum + 360.0) / 2.0, (hue_sum - 360.0) / 2.0),
        ),
    )
    t_value = (
        1.0
        - 0.17 * np.cos(np.radians(mean_h - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * mean_h))
        + 0.32 * np.cos(np.radians(3.0 * mean_h + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * mean_h - 63.0))
    )
    sl = 1.0 + 0.015 * (mean_l - 50.0) ** 2 / np.sqrt(20.0 + (mean_l - 50.0) ** 2)
    sc = 1.0 + 0.045 * mean_c_prime
    sh = 1.0 + 0.015 * mean_c_prime * t_value
    rotation = 30.0 * np.exp(-((mean_h - 275.0) / 25.0) ** 2)
    rc = 2.0 * np.sqrt(
        mean_c_prime**7 / np.maximum(mean_c_prime**7 + 25.0**7, 1e-12)
    )
    rt = -rc * np.sin(np.radians(2.0 * rotation))
    return np.sqrt(
        (delta_l / sl) ** 2
        + (delta_c / sc) ** 2
        + (delta_h / sh) ** 2
        + rt * (delta_c / sc) * (delta_h / sh)
    )


def dominant_palette_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Compare weighted Lab palettes and return a normalised [0, 2] distance."""

    left_palette = np.asarray(left, dtype=np.float32).reshape(
        COLOUR_PALETTE_SIZE, COLOUR_PALETTE_COMPONENTS
    )
    right_palette = np.asarray(right, dtype=np.float32).reshape(
        COLOUR_PALETTE_SIZE, COLOUR_PALETTE_COMPONENTS
    )
    left_active = left_palette[:, 3] > 1e-6
    right_active = right_palette[:, 3] > 1e-6
    if not left_active.any() and not right_active.any():
        return 0.0
    if not left_active.any() or not right_active.any():
        return 2.0
    left_colours = left_palette[left_active, :3]
    right_colours = right_palette[right_active, :3]
    left_weights = left_palette[left_active, 3]
    right_weights = right_palette[right_active, 3]
    pairwise = delta_e_ciede2000(
        left_colours[:, None, :],
        right_colours[None, :, :],
    )
    directed_left = float(np.average(np.min(pairwise, axis=1), weights=left_weights))
    directed_right = float(np.average(np.min(pairwise, axis=0), weights=right_weights))
    palette_delta_e = (directed_left + directed_right) / 2.0
    dominant_delta_e = float(delta_e_ciede2000(left_colours[0], right_colours[0]))
    combined_delta_e = 0.70 * palette_delta_e + 0.30 * dominant_delta_e
    return float(np.clip(combined_delta_e / COLOUR_DELTA_E_SCALE, 0.0, 2.0))


def _texture_descriptor(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = values[..., 0] * 0.299 + values[..., 1] * 0.587 + values[..., 2] * 0.114
    padded = np.pad(gray, 1, mode="edge")
    gradient_x = padded[1:-1, 2:] - padded[1:-1, :-2]
    gradient_y = padded[2:, 1:-1] - padded[:-2, 1:-1]
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = (np.arctan2(gradient_y, gradient_x) + np.pi) % np.pi
    valid = magnitude > 1e-4
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
        (local_energy.reshape(-1) / 0.25 * TEXTURE_ENERGY_BINS).astype(int),
        TEXTURE_ENERGY_BINS - 1,
    )
    energy_histogram = np.bincount(energy_bins, minlength=TEXTURE_ENERGY_BINS).astype(np.float32)
    return _l2_normalize(np.concatenate((orientation_histogram, energy_histogram)))


def extract_appearance_features(
    image: Image.Image,
    item_type: Any = None,
) -> AppearanceFeatures:
    """Derive all retrieval signals from the configured analysis region."""

    source = image.convert("RGB")
    analysis_image = visual_analysis_region(source, item_type)
    descriptor_image = _working_image(analysis_image)
    try:
        rgb = np.asarray(descriptor_image, dtype=np.uint8)
        # The item-type crop has already been applied to ``analysis_image``;
        # passing no item type prevents an accidental second OTTR crop.
        garment_rgb = _garment_body_rgb(analysis_image)
        return AppearanceFeatures(
            image=analysis_image,
            colour_vector=_dominant_colour_palette(garment_rgb),
            texture_vector=_texture_descriptor(rgb),
        )
    except Exception:
        analysis_image.close()
        raise
    finally:
        if descriptor_image is not analysis_image:
            descriptor_image.close()
        source.close()


def extract_pipeline_appearance_features(
    image: Image.Image,
    item_type: Any = None,
) -> AppearanceFeatures:
    """Pipeline entry point for non-destructive garment appearance features."""

    return extract_appearance_features(image, item_type)


def body_pattern_views(
    image: Image.Image,
    item_type: Any = None,
) -> list[Image.Image]:
    """Return large/medium/fine centre-body crops for pattern comparison."""

    source = visual_analysis_region(image, item_type)
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
        type_design_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, item in enumerate(candidates):
            item_type = canonical_retrieval_value(item.get("itemType"))
            design = canonical_retrieval_design(item.get("design"))
            if item_type:
                type_groups[item_type].append(index)
            if design:
                design_groups[design].append(index)
            if item_type and design:
                type_design_groups[(item_type, design)].append(index)
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
        self.backend = self._global.backend

    def search(
        self,
        query: dict[str, Any],
        vector: np.ndarray,
        limit: int,
        *,
        require_same_item_type: bool,
        require_same_design: bool,
    ) -> list[int]:
        item_type = canonical_retrieval_value(query.get("itemType"))
        design = canonical_retrieval_design(query.get("design"))
        selected: tuple[list[int], InnerProductIndex] | None
        if require_same_item_type and require_same_design:
            selected = self._by_type_and_design.get((item_type, design)) if item_type and design else None
        elif require_same_item_type:
            selected = self._by_type.get(item_type) if item_type else None
        elif require_same_design:
            selected = self._by_design.get(design) if design else None
        else:
            return self._global.search(vector, limit)
        if selected is not None:
            source_indices, index = selected
            return [source_indices[position] for position in index.search(vector, limit)]
        # A required attribute with no eligible history is a true no-match,
        # never a reason to fall back to another item type or design.
        return []
