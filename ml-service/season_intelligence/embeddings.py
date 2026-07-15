from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Mapping, Protocol

from season_intelligence.contracts import product_text


class EmbeddingProvider(Protocol):
    def embed(self, product: Mapping[str, Any]) -> list[float]: ...


class HTTPEmbeddingProvider:
    """Calls the separately deployable FashionCLIP/SigLIP service."""

    def __init__(self, url: str, api_key: str | None = None, dimension: int = 512, timeout: float = 12.0):
        self.url = url.rstrip("/") + "/v1/embeddings"
        self.api_key = api_key
        self.dimension = dimension
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> "HTTPEmbeddingProvider":
        url = os.getenv("EMBEDDING_SERVICE_URL")
        if not url:
            raise RuntimeError("EMBEDDING_SERVICE_URL is required in scale mode")
        return cls(
            url=url,
            api_key=os.getenv("EMBEDDING_SERVICE_API_KEY"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "512")),
            timeout=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "12")),
        )

    def embed(self, product: Mapping[str, Any]) -> list[float]:
        body = json.dumps({
            "products": [{
                "id": str(product["id"]),
                "text": product_text(product),
                "imageUrl": product.get("imageUrl"),
            }]
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("embedding service unavailable") from exc
        try:
            vector = payload["embeddings"][0]["vector"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("embedding service returned an invalid response") from exc
        if len(vector) != self.dimension or not all(isinstance(value, (int, float)) for value in vector):
            raise RuntimeError("embedding service returned an invalid vector")
        return [float(value) for value in vector]


class SuppliedEmbeddingProvider:
    """Test/offline provider; production callers cannot enable this accidentally."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def embed(self, product: Mapping[str, Any]) -> list[float]:
        vector = product.get("embedding")
        if not isinstance(vector, list) or len(vector) != self.dimension:
            raise ValueError(f"a {self.dimension}-dimension embedding is required")
        return [float(value) for value in vector]
