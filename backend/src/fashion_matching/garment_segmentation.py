"""Text-guided, quality-gated garment segmentation for catalogue images.

Grounding DINO locates the garment described by the catalogue item type and
SAM 2 turns that box into a pixel mask.  The module deliberately exposes a
small, explicit contract so that no caller has to infer success from a masked
image: every result carries its confidence, method and rejection reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from fashion_matching.encoders import choose_device


class SegmentationError(RuntimeError):
    """Raised when an enabled production segmentation model cannot start."""


@dataclass(frozen=True)
class GarmentMask:
    """A candidate garment mask and its audit metadata."""

    mask: np.ndarray | None
    coverage: float
    confidence: float
    method: str
    fallback_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.mask is not None and self.fallback_reason is None


class GarmentSegmenter(Protocol):
    """Interface shared by real and test segmenters."""

    method: str

    def segment(self, image: Image.Image, *, item_type: str = "") -> GarmentMask: ...

    def metadata(self) -> dict[str, str | float]: ...


# Item types are internal catalogue codes.  The detector must receive a human
# garment noun, never the opaque code itself. Unknown types intentionally use a
# broad garment prompt rather than a guessed, potentially wrong class.
ITEM_TYPE_PROMPTS = {
    "OTSH": "a shirt.",
    "OTTS": "a t-shirt.",
    "OTTR": "a pair of trousers.",
    "OTJK": "a jacket.",
    "OTSU": "a suit.",
}


def garment_prompt(item_type: str) -> str:
    """Return a Grounding DINO prompt, including its required class delimiter."""

    normalized = " ".join(str(item_type or "").upper().split())
    return ITEM_TYPE_PROMPTS.get(normalized, "a garment.")


def _coverage_confidence(coverage: float) -> float:
    # A garment normally occupies a meaningful, but not near-total, portion of
    # a product photo. This is only a quality signal, not a shape assumption.
    return float(np.clip(1.0 - abs(coverage - 0.38) / 0.55, 0.0, 1.0))


def validate_garment_mask(
    mask: np.ndarray,
    *,
    box: np.ndarray,
    detection_score: float,
    minimum_detection_score: float,
    minimum_coverage: float,
    maximum_coverage: float,
) -> GarmentMask:
    """Reject masks that cannot credibly be the detected garment.

    SAM 2 is promptable, not a catalogue-specific classifier.  We therefore
    require sufficient detector evidence, a plausible garment area and that
    almost all selected pixels stay inside the detector's garment box.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.size == 0:
        return GarmentMask(None, 0.0, 0.0, "grounding-dino-sam2", "invalid_mask_shape")
    if not np.isfinite(detection_score) or detection_score < minimum_detection_score:
        return GarmentMask(
            None,
            float(binary.mean()),
            float(detection_score),
            "grounding-dino-sam2",
            "low_detector_score",
        )

    height, width = binary.shape
    coverage = float(binary.mean())
    if not minimum_coverage <= coverage <= maximum_coverage:
        return GarmentMask(None, coverage, float(detection_score), "grounding-dino-sam2", "implausible_mask_coverage")

    left, top, right, bottom = np.asarray(box, dtype=np.float32).tolist()
    left = int(np.clip(np.floor(left), 0, max(width - 1, 0)))
    top = int(np.clip(np.floor(top), 0, max(height - 1, 0)))
    right = int(np.clip(np.ceil(right), left + 1, width))
    bottom = int(np.clip(np.ceil(bottom), top + 1, height))
    detector_area = max((right - left) * (bottom - top), 1)
    inside_detector_box = np.zeros_like(binary, dtype=bool)
    inside_detector_box[top:bottom, left:right] = True
    mask_pixels = max(int(binary.sum()), 1)
    containment = float((binary & inside_detector_box).sum() / mask_pixels)
    box_fill = float((binary & inside_detector_box).sum() / detector_area)
    if containment < 0.92:
        return GarmentMask(None, coverage, float(detection_score), "grounding-dino-sam2", "mask_outside_detector_box")
    if box_fill < 0.10:
        return GarmentMask(
            None,
            coverage,
            float(detection_score),
            "grounding-dino-sam2",
            "mask_too_small_for_detector_box",
        )

    confidence = 0.65 * float(detection_score) + 0.20 * _coverage_confidence(coverage) + 0.15 * containment
    if confidence < 0.52:
        return GarmentMask(None, coverage, confidence, "grounding-dino-sam2", "low_mask_confidence")
    return GarmentMask(binary, coverage, confidence, "grounding-dino-sam2")


@dataclass
class GroundedSam2GarmentSegmenter:
    """Grounding DINO box detection followed by SAM 2.1 mask extraction."""

    detector_model_id: str = "IDEA-Research/grounding-dino-tiny"
    detector_revision: str = "main"
    sam2_model_id: str = "facebook/sam2.1-hiera-small"
    sam2_revision: str = "main"
    configured_device: str = "auto"
    detector_threshold: float = 0.35
    text_threshold: float = 0.25
    minimum_coverage: float = 0.04
    maximum_coverage: float = 0.88

    method = "grounding-dino-sam2"

    def __post_init__(self) -> None:
        if not 0 <= self.detector_threshold <= 1 or not 0 <= self.text_threshold <= 1:
            raise ValueError("segmentation detector thresholds must be between 0 and 1")
        if not 0 < self.minimum_coverage < self.maximum_coverage < 1:
            raise ValueError("segmentation mask coverage range must be inside (0, 1)")
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise SegmentationError(
                "Garment segmentation requires sam2, huggingface_hub, torch and transformers. "
                "Install backend/requirements-fashion-matching.txt."
            ) from exc

        self.device = choose_device(self.configured_device)
        self._torch = torch
        self.detector_processor = AutoProcessor.from_pretrained(
            self.detector_model_id,
            revision=self.detector_revision,
        )
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.detector_model_id,
            revision=self.detector_revision,
        ).to(self.device).eval()
        self.resolved_detector_revision = getattr(self.detector.config, "_commit_hash", None) or self.detector_revision

        try:
            config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[self.sam2_model_id]
        except KeyError as exc:
            raise SegmentationError(f"Unsupported SAM 2 model: {self.sam2_model_id}") from exc
        checkpoint_path = Path(
            hf_hub_download(
                repo_id=self.sam2_model_id,
                filename=checkpoint_name,
                revision=self.sam2_revision,
            )
        )
        self.predictor = SAM2ImagePredictor(build_sam2(config_name, checkpoint_path, device=self.device))
        self.resolved_sam2_revision = next(
            (parent.name for parent in checkpoint_path.parents if parent.parent.name == "snapshots"),
            self.sam2_revision,
        )

    def metadata(self) -> dict[str, str | float]:
        return {
            "method": self.method,
            "detectorModelId": self.detector_model_id,
            "detectorModelRevision": self.resolved_detector_revision,
            "sam2ModelId": self.sam2_model_id,
            "sam2ModelRevision": self.resolved_sam2_revision,
            "device": self.device,
            "detectorThreshold": self.detector_threshold,
            "textThreshold": self.text_threshold,
            "minimumCoverage": self.minimum_coverage,
            "maximumCoverage": self.maximum_coverage,
        }

    def segment(self, image: Image.Image, *, item_type: str = "") -> GarmentMask:
        """Return the highest-confidence valid SAM 2 mask for the item type."""

        source = image.convert("RGB")
        prompt = garment_prompt(item_type)
        try:
            # Current Transformers accepts one text prompt per image as a flat
            # list. Earlier releases accepted a nested list, so keep the input
            # shape explicit here instead of relying on implicit broadcasting.
            inputs = self.detector_processor(images=source, text=[prompt], return_tensors="pt").to(self.device)
            with self._torch.inference_mode():
                outputs = self.detector(**inputs)
            result = self.detector_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.detector_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(source.height, source.width)],
            )[0]
            scores = result.get("scores")
            boxes = result.get("boxes")
            if scores is None or boxes is None or len(scores) == 0:
                return GarmentMask(None, 0.0, 0.0, self.method, "garment_not_detected")
            best_index = int(scores.argmax().item())
            detection_score = float(scores[best_index].item())
            box = boxes[best_index].detach().cpu().numpy().astype(np.float32)
            self.predictor.set_image(np.asarray(source, dtype=np.uint8).copy())
            masks, mask_scores, _ = self.predictor.predict(box=box, multimask_output=True)
            if masks is None or len(masks) == 0:
                return GarmentMask(None, 0.0, detection_score, self.method, "sam2_returned_no_mask")
            mask_index = int(np.asarray(mask_scores).argmax())
            return validate_garment_mask(
                masks[mask_index],
                box=box,
                detection_score=detection_score,
                minimum_detection_score=self.detector_threshold,
                minimum_coverage=self.minimum_coverage,
                maximum_coverage=self.maximum_coverage,
            )
        except Exception as exc:  # pragma: no cover - model-runtime dependent
            # The pipeline continues deterministically: appearance signals are
            # unavailable and the caller can retain neural-only fallback.
            return GarmentMask(None, 0.0, 0.0, self.method, f"segmentation_runtime_error:{type(exc).__name__}")
        finally:
            reset = getattr(self.predictor, "reset_predictor", None)
            if callable(reset):
                reset()
