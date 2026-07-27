from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from domain.contracts import product_text


@dataclass(frozen=True)
class EmbeddingSignals:
    image: list[float] | None = None
    text: list[float] | None = None


class EmbeddingProvider(Protocol):
    def embed(self, product: Mapping[str, Any]) -> list[float]: ...

    def embed_signals(self, product: Mapping[str, Any]) -> EmbeddingSignals: ...


class HTTPEmbeddingProvider:
    """Calls the separately deployable FashionCLIP/SigLIP service."""

    def __init__(self, url: str, api_key: str | None = None, dimension: int = 768, timeout: float = 12.0):
        self.url = url.rstrip("/") + "/v1/embeddings"
        self.api_key = api_key
        self.dimension = dimension
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> HTTPEmbeddingProvider:
        url = os.getenv("EMBEDDING_SERVICE_URL")
        if not url:
            raise RuntimeError("EMBEDDING_SERVICE_URL is required in scale mode")
        return cls(
            url=url,
            api_key=os.getenv("EMBEDDING_SERVICE_API_KEY"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            timeout=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "12")),
        )

    def embed(self, product: Mapping[str, Any]) -> list[float]:
        signals = self.embed_signals(product)
        if signals.image is None:
            raise RuntimeError("image embedding is unavailable; text is not substituted for an image signal")
        return signals.image

    def embed_signals(
        self,
        product: Mapping[str, Any],
    ) -> EmbeddingSignals:
        body = json.dumps(
            {
                "products": [
                    {
                        "id": str(product["id"]),
                        "text": product_text(product),
                        "imageUrl": product.get("imageUrl"),
                    }
                ]
            }
        ).encode("utf-8")
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
            item = payload["embeddings"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("embedding service returned an invalid response") from exc
        image = self._optional_vector(item.get("imageVector"))
        text = self._optional_vector(item.get("textVector"))
        if image is None and text is None:
            raise RuntimeError("embedding service returned no usable signals")
        return EmbeddingSignals(image=image, text=text)

    def _optional_vector(self, vector: Any) -> list[float] | None:
        if vector is None:
            return None
        if (
            not isinstance(vector, list)
            or len(vector) != self.dimension
            or not all(isinstance(value, (int, float)) for value in vector)
        ):
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

    def embed_signals(
        self,
        product: Mapping[str, Any],
    ) -> EmbeddingSignals:
        return EmbeddingSignals(image=self.embed(product))
