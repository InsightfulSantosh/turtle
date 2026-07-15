from __future__ import annotations

import io
import ipaddress
import os
import socket
from functools import lru_cache
from urllib.parse import urlparse

import httpx
import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoProcessor, CLIPModel


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
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        with client.stream("GET", url, headers={"User-Agent": "TurtleSeasonIntelligence/3.0"}) as response:
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
    return Image.open(io.BytesIO(data)).convert("RGB")


class FashionEmbeddingRuntime:
    def __init__(self) -> None:
        model_id = os.getenv("FASHION_MODEL_ID", "patrickjohncyh/fashion-clip")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.eval()
        self.model_id = model_id
        self.image_weight = min(max(float(os.getenv("IMAGE_WEIGHT", "0.70")), 0.0), 1.0)

    @torch.inference_mode()
    def embed(self, product: ProductInput) -> list[float]:
        text_inputs = self.processor(
            text=[product.text or product.id],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_vector = torch.nn.functional.normalize(self.model.get_text_features(**text_inputs), dim=-1)
        if product.imageUrl:
            image_inputs = self.processor(images=[fetch_image(product.imageUrl)], return_tensors="pt")
            image_vector = torch.nn.functional.normalize(self.model.get_image_features(**image_inputs), dim=-1)
            vector = image_vector * self.image_weight + text_vector * (1 - self.image_weight)
        else:
            vector = text_vector
        return torch.nn.functional.normalize(vector, dim=-1)[0].cpu().float().tolist()


@lru_cache(maxsize=1)
def runtime() -> FashionEmbeddingRuntime:
    return FashionEmbeddingRuntime()


app = FastAPI(title="Turtle Fashion Embedding Service", version="3.0.0", redoc_url=None)


@app.get("/healthz", include_in_schema=False)
def health() -> dict:
    return {"status": "ok", "modelLoaded": runtime.cache_info().currsize > 0}


@app.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
def create_embeddings(payload: EmbeddingRequest) -> dict:
    try:
        model = runtime()
        embeddings = [{"id": product.id, "vector": model.embed(product)} for product in payload.products]
    except (ValueError, httpx.HTTPError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"model": model.model_id, "dimension": len(embeddings[0]["vector"]), "embeddings": embeddings}
