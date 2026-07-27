from __future__ import annotations

import hashlib
import io
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from fashion_matching.models import ManifestRecord, PreparedImage

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class ImageValidationError(ValueError):
    """Raised when an image cannot be safely and deterministically prepared."""


@dataclass(frozen=True)
class ImagePreprocessor:
    version: str = "rgb-exif-pad-v1"
    max_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 40_000_000
    minimum_dimension: int = 32
    pad_to_square: bool = True
    crop_uniform_background: bool = False
    allowed_image_domains: tuple[str, ...] = ()

    def prepare(self, record: ManifestRecord) -> PreparedImage:
        data = self._load_bytes(record)
        checksum = hashlib.sha256(data).hexdigest()
        image = self._decode(data)
        if self.crop_uniform_background:
            image = self._crop_background(image)
        if self.pad_to_square:
            image = self._pad_square(image)
        return PreparedImage(
            image=image,
            checksum=checksum,
            width=image.width,
            height=image.height,
            source_bytes=len(data),
        )

    def _load_bytes(self, record: ManifestRecord) -> bytes:
        if record.image_path is not None:
            return self._read_local(record.image_path)
        if record.image_url is not None:
            return self._download(record.image_url)
        raise ImageValidationError("record has no image source")

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

    def _validate_remote_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ImageValidationError("remote image URL must use HTTPS")
        if parsed.username or parsed.password:
            raise ImageValidationError("remote image URL cannot contain credentials")
        host = parsed.hostname.lower()
        if not self.allowed_image_domains or not any(
            host == domain or host.endswith(f".{domain}") for domain in self.allowed_image_domains
        ):
            raise ImageValidationError("remote image host is not allowlisted")
        try:
            addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ImageValidationError("remote image host cannot be resolved") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise ImageValidationError("remote image host resolved to a blocked network")
        return url

    def _download(self, url: str) -> bytes:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("httpx is required to download remote images") from exc
        self._validate_remote_url(url)
        timeout = httpx.Timeout(12.0, connect=4.0)
        total = 0
        chunks: list[bytes] = []
        try:
            with (
                httpx.Client(
                    timeout=timeout,
                    follow_redirects=False,
                ) as client,
                client.stream(
                    "GET",
                    url,
                    headers={"User-Agent": "TurtleFashionMatching/1.0"},
                ) as response,
            ):
                response.raise_for_status()
                content_type = response.headers.get(
                    "content-type",
                    "",
                ).split(";", 1)[0]
                if content_type not in {
                    "image/jpeg",
                    "image/png",
                    "image/webp",
                }:
                    raise ImageValidationError(f"unsupported remote content type: {content_type}")
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise ImageValidationError(f"image exceeds {self.max_bytes} byte limit")
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise ImageValidationError(f"image download failed: {exc}") from exc
        if not chunks:
            raise ImageValidationError("downloaded image is empty")
        return b"".join(chunks)

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

    @staticmethod
    def _crop_background(image: Image.Image) -> Image.Image:
        array = np.asarray(image, dtype=np.int16)
        corners = np.stack(
            [
                array[0, 0],
                array[0, -1],
                array[-1, 0],
                array[-1, -1],
            ]
        )
        background = np.median(corners, axis=0)
        distance = np.max(np.abs(array - background), axis=2)
        foreground = distance > 18
        y, x = np.where(foreground)
        if not len(x):
            return image
        left, right = int(x.min()), int(x.max()) + 1
        top, bottom = int(y.min()), int(y.max()) + 1
        foreground_area = (right - left) * (bottom - top)
        if foreground_area < image.width * image.height * 0.02:
            return image
        margin = max(round(max(image.size) * 0.03), 2)
        box = (
            max(left - margin, 0),
            max(top - margin, 0),
            min(right + margin, image.width),
            min(bottom + margin, image.height),
        )
        return image.crop(box)
