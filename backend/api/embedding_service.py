from __future__ import annotations

import io
import ipaddress
import os
import socket
from functools import lru_cache
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from fashion_matching.encoders import (
    EncoderError,
    FashionEncoder,
    create_encoder,
)


class ProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    text: str = Field(default="", max_length=2_000)
    imageUrl: str | None = Field(default=None, max_length=2_048)


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    products: list[ProductInput] = Field(min_length=1, max_length=32)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("EMBEDDING_SERVICE_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _validate_image_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("image URL must use HTTPS")
    allowed = [domain.strip().lower() for domain in os.getenv("ALLOWED_IMAGE_DOMAINS", "").split(",") if domain.strip()]
    host = parsed.hostname.lower()
    if not allowed or not any(host == domain or host.endswith("." + domain) for domain in allowed):
        raise ValueError("image host is not allowlisted")
    for address in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("image host resolved to a blocked network")


def fetch_image(url: str) -> Image.Image:
    _validate_image_host(url)
    max_bytes = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
    timeout = httpx.Timeout(10.0, connect=4.0)
    with (
        httpx.Client(
            timeout=timeout,
            follow_redirects=False,
        ) as client,
        client.stream(
            "GET",
            url,
            headers={"User-Agent": "TurtleSeasonIntelligence/4.0"},
        ) as response,
    ):
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        if content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("unsupported image content type")
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("image exceeds configured size limit")
            chunks.append(chunk)
    data = b"".join(chunks)
    image = Image.open(io.BytesIO(data))
    image.verify()
    decoded = Image.open(io.BytesIO(data))
    if decoded.width * decoded.height > int(os.getenv("MAX_IMAGE_PIXELS", "40000000")):
        raise ValueError("image exceeds configured pixel limit")
    return ImageOps.exif_transpose(decoded).convert("RGB")


class FashionEmbeddingRuntime:
    def __init__(self) -> None:
        self.encoder: FashionEncoder = create_encoder(
            os.getenv(
                "FASHION_MODEL_ID",
                "Marqo/marqo-fashionSigLIP",
            ),
            revision=os.getenv("FASHION_MODEL_REVISION", "main"),
            device=os.getenv("FASHION_MODEL_DEVICE", "auto"),
        )
        self.model_id = self.encoder.model_id

    def embed_batch(self, products: list[ProductInput]) -> list[dict]:
        results = [
            {
                "id": product.id,
                "imageVector": None,
                "textVector": None,
            }
            for product in products
        ]
        image_positions = [index for index, product in enumerate(products) if product.imageUrl]
        if image_positions:
            images = [fetch_image(products[index].imageUrl or "") for index in image_positions]
            vectors = self.encoder.encode_images(images)
            for index, vector in zip(
                image_positions,
                vectors,
                strict=True,
            ):
                results[index]["imageVector"] = vector

        text_positions = [index for index, product in enumerate(products) if product.text.strip()]
        if text_positions and self.encoder.supports_text:
            vectors = self.encoder.encode_texts([products[index].text for index in text_positions])
            for index, vector in zip(
                text_positions,
                vectors,
                strict=True,
            ):
                results[index]["textVector"] = vector
        for result in results:
            result["availableSignals"] = [
                signal
                for signal, field in (
                    ("image", "imageVector"),
                    ("text", "textVector"),
                )
                if result[field] is not None
            ]
            # Backward-compatible alias: this is image-only and never falls
            # back to the text vector.
            result["vector"] = result["imageVector"]
        return results


@lru_cache(maxsize=1)
def runtime() -> FashionEmbeddingRuntime:
    return FashionEmbeddingRuntime()


app = FastAPI(
    title="Turtle Fashion Embedding Service",
    version="4.0.0",
    redoc_url=None,
)


@app.get("/healthz", include_in_schema=False)
def health() -> dict:
    return {"status": "ok", "modelLoaded": runtime.cache_info().currsize > 0}


@app.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
def create_embeddings(payload: EmbeddingRequest) -> dict:
    try:
        model = runtime()
        embeddings = model.embed_batch(payload.products)
    except (EncoderError, ValueError, httpx.HTTPError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "model": model.model_id,
        "revision": model.encoder.revision,
        "dimension": model.encoder.dimension,
        "embeddings": embeddings,
    }
