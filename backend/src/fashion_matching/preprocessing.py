from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class ImageValidationError(ValueError):
    """Raised when an image cannot be safely and deterministically prepared."""


@dataclass(frozen=True)
class PreparedImage:
    image: Any
    checksum: str
    width: int
    height: int


@dataclass(frozen=True)
class ImagePreprocessor:
    version: str = "rgb-exif-pad-v1"
    max_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 40_000_000
    minimum_dimension: int = 32
    pad_to_square: bool = True

    def prepare(self, path: Path) -> PreparedImage:
        data = self._read_local(path)
        checksum = hashlib.sha256(data).hexdigest()
        image = self._decode(data)
        if self.pad_to_square:
            image = self._pad_square(image)
        return PreparedImage(
            image=image,
            checksum=checksum,
            width=image.width,
            height=image.height,
        )

    def _read_local(self, path: Path) -> bytes:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ImageValidationError(f"unsupported image extension: {path.suffix or '<none>'}")
        if not path.is_file():
            raise ImageValidationError(f"image does not exist: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise ImageValidationError("image file is empty")
        if size > self.max_bytes:
            raise ImageValidationError(f"image exceeds {self.max_bytes} byte limit")
        return path.read_bytes()

    def _decode(self, data: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(data)) as source:
                source.verify()
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in SUPPORTED_FORMATS:
                    raise ImageValidationError(f"unsupported image format: {source.format}")
                width, height = source.size
                if min(width, height) < self.minimum_dimension:
                    raise ImageValidationError(
                        f"image dimensions must be at least {self.minimum_dimension}x{self.minimum_dimension}"
                    )
                if width * height > self.max_pixels:
                    raise ImageValidationError(f"image exceeds {self.max_pixels} pixel limit")
                transposed = ImageOps.exif_transpose(source)
                if transposed.mode in {"RGBA", "LA"}:
                    background = Image.new("RGB", transposed.size, "white")
                    alpha = transposed.getchannel("A")
                    background.paste(transposed.convert("RGB"), mask=alpha)
                    return background
                return transposed.convert("RGB")
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ImageValidationError("image is invalid or corrupted") from exc

    @staticmethod
    def _pad_square(image: Image.Image) -> Image.Image:
        side = max(image.size)
        left = (side - image.width) // 2
        top = (side - image.height) // 2
        canvas = Image.new("RGB", (side, side), "white")
        canvas.paste(image, (left, top))
        return canvas
