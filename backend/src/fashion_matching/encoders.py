from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image


class EncoderError(RuntimeError):
    """Raised when a configured embedding model cannot produce valid vectors."""


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("embedding must have a finite, non-zero L2 norm")
    return [float(value / norm) for value in vector]


class ImageEncoder(Protocol):
    model_id: str
    revision: str
    dimension: int

    def encode_images(self, images: Sequence[Image.Image]) -> list[list[float]]: ...


class FashionEncoder(ImageEncoder, Protocol):
    supports_text: bool

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


def choose_device(configured: str) -> str:
    if configured != "auto":
        return configured
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise EncoderError("PyTorch is required for model inference") from exc
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_version_identifier(
    encoder: FashionEncoder,
    preprocessing_version: str,
) -> str:
    source = "\n".join(
        (
            encoder.model_id,
            encoder.revision,
            str(encoder.dimension),
            preprocessing_version,
        )
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"{encoder.model_id.rsplit('/', 1)[-1].lower()}-{digest}"


def _tensor_vectors(value: Any, expected: int | None = None) -> list[list[float]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise EncoderError("PyTorch is required for model inference") from exc
    if not isinstance(value, torch.Tensor):
        for field in ("image_embeds", "text_embeds", "pooler_output"):
            candidate = getattr(value, field, None)
            if isinstance(candidate, torch.Tensor):
                value = candidate
                break
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise EncoderError("model returned an invalid embedding tensor")
    rows = [l2_normalize(row.detach().cpu().float().tolist()) for row in value]
    if expected is not None and any(len(row) != expected for row in rows):
        raise EncoderError(f"model returned an unexpected embedding dimension; expected {expected}")
    return rows


@dataclass
class HuggingFaceFashionEncoder:
    model_id: str
    requested_revision: str = "main"
    configured_device: str = "auto"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise EncoderError("torch and transformers are required for Hugging Face encoders") from exc
        self.device = choose_device(self.configured_device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=self.requested_revision,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            revision=self.requested_revision,
            trust_remote_code=True,
        )
        self.model.to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or self.requested_revision
        projection = getattr(self.model.config, "projection_dim", None)
        self.dimension = int(projection or self._detect_dimension(torch))
        self.supports_text = hasattr(self.model, "get_text_features")

    def _detect_dimension(self, torch_module: Any) -> int:
        sample = Image.new("RGB", (224, 224), "white")
        with torch_module.inference_mode():
            return len(self.encode_images([sample])[0])

    def _image_features(self, inputs: dict[str, Any]) -> Any:
        pixel_values = inputs["pixel_values"]
        try:
            return self.model.get_image_features(
                pixel_values=pixel_values,
                normalize=True,
            )
        except TypeError:
            try:
                return self.model.get_image_features(pixel_values)
            except TypeError:
                return self.model.get_image_features(**inputs)

    def encode_images(
        self,
        images: Sequence[Image.Image],
    ) -> list[list[float]]:
        if not images:
            return []
        import torch

        inputs = self.processor(
            images=list(images),
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items() if hasattr(tensor, "to")}
        with torch.inference_mode():
            features = self._image_features(inputs)
        expected = getattr(self, "dimension", None)
        return _tensor_vectors(features, expected)

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.supports_text:
            raise EncoderError(f"{self.model_id} does not support text embeddings")
        if not texts or any(not text.strip() for text in texts):
            raise EncoderError("text embeddings require non-empty descriptions")
        import torch

        inputs = self.processor(
            text=list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items() if hasattr(tensor, "to")}
        with torch.inference_mode():
            try:
                features = self.model.get_text_features(
                    **inputs,
                    normalize=True,
                )
            except TypeError:
                try:
                    features = self.model.get_text_features(
                        inputs["input_ids"],
                        normalize=True,
                    )
                except TypeError:
                    features = self.model.get_text_features(**inputs)
        return _tensor_vectors(features, self.dimension)


@dataclass
class DINOv2VisualEncoder:
    """Image-only DINOv2 encoder used to rerank fashion-retrieval candidates.

    DINOv2 is deliberately kept separate from the FashionSigLIP encoder. It
    captures fine visual detail without replacing the fashion model's semantic
    candidate-retrieval role.
    """

    model_id: str = "facebook/dinov2-base"
    requested_revision: str = "main"
    configured_device: str = "auto"

    def __post_init__(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise EncoderError("torch and transformers are required for DINOv2 inference") from exc
        self.device = choose_device(self.configured_device)
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id,
            revision=self.requested_revision,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            revision=self.requested_revision,
            trust_remote_code=False,
        )
        self.model.to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or self.requested_revision
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise EncoderError(f"{self.model_id} does not expose a visual embedding dimension")
        self.dimension = int(hidden_size)

    def encode_images(
        self,
        images: Sequence[Image.Image],
    ) -> list[list[float]]:
        if not images:
            return []
        import torch

        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {
            name: tensor.to(self.device)
            for name, tensor in inputs.items()
            if hasattr(tensor, "to")
        }
        with torch.inference_mode():
            output = self.model(**inputs)
        return _tensor_vectors(output, self.dimension)


@dataclass
class OpenClipFashionEncoder:
    model_id: str
    requested_revision: str = "main"
    configured_device: str = "auto"

    def __post_init__(self) -> None:
        try:
            import open_clip
            from huggingface_hub import hf_hub_download, model_info, snapshot_download
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise EncoderError("open_clip_torch is required for this encoder") from exc
        self.device = choose_device(self.configured_device)
        if self.model_id.lower().startswith("marqo/"):
            snapshot = snapshot_download(
                repo_id=self.model_id,
                revision=self.requested_revision,
                allow_patterns=[
                    "open_clip_config.json",
                    "open_clip_model.safetensors",
                    "spiece.model",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ],
            )
            local_model = f"local-dir:{snapshot}"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                local_model,
                device=self.device,
            )
            self.tokenizer = open_clip.get_tokenizer(local_model)
            self.revision = snapshot.rsplit("/", 1)[-1]
        else:
            checkpoint = hf_hub_download(
                repo_id=self.model_id,
                filename="open_clip_model.safetensors",
                revision=self.requested_revision,
            )
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16-SigLIP",
                pretrained=checkpoint,
                device=self.device,
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
            self.revision = str(
                model_info(
                    self.model_id,
                    revision=self.requested_revision,
                ).sha
            )
        self.model.eval()
        self.dimension = len(self.encode_images([Image.new("RGB", (224, 224), "white")])[0])
        self.supports_text = hasattr(self.model, "encode_text")

    def encode_images(
        self,
        images: Sequence[Image.Image],
    ) -> list[list[float]]:
        if not images:
            return []
        import torch

        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(batch, normalize=True)
        return _tensor_vectors(features, getattr(self, "dimension", None))

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.supports_text:
            raise EncoderError(f"{self.model_id} does not support text embeddings")
        if not texts or any(not text.strip() for text in texts):
            raise EncoderError("text embeddings require non-empty descriptions")
        import torch

        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens, normalize=True)
        return _tensor_vectors(features, self.dimension)


def create_encoder(
    model_id: str,
    *,
    revision: str = "main",
    device: str = "auto",
) -> FashionEncoder:
    if model_id.lower().startswith(("hopitai/", "marqo/")):
        return OpenClipFashionEncoder(model_id, revision, device)
    return HuggingFaceFashionEncoder(model_id, revision, device)


def create_dino_encoder(
    model_id: str = "facebook/dinov2-base",
    *,
    revision: str = "main",
    device: str = "auto",
) -> ImageEncoder:
    """Create the isolated visual-detail encoder for second-stage reranking."""

    return DINOv2VisualEncoder(model_id, revision, device)
