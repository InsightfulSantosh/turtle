from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from fashion_matching.models import SearchHit, VectorPoint


class VectorStoreError(RuntimeError):
    """Raised when vector storage is unavailable or incompatible."""


class VectorStore(Protocol):
    def ensure_collection(
        self,
        name: str,
        vector_dimensions: Mapping[str, int],
    ) -> None: ...

    def get_point(self, collection: str, point_id: str) -> VectorPoint | None: ...

    def upsert(self, collection: str, points: Sequence[VectorPoint]) -> None: ...

    def search(
        self,
        collection: str,
        vector_name: str,
        query: Sequence[float],
        limit: int,
    ) -> list[SearchHit]: ...

    def activate_alias(self, collection: str, alias: str) -> None: ...


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        raise VectorStoreError("cannot compare a zero vector")
    return numerator / (left_norm * right_norm)


@dataclass
class _MemoryCollection:
    dimensions: dict[str, int]
    points: dict[str, VectorPoint] = field(default_factory=dict)


class InMemoryVectorStore:
    """Deterministic test and small-development store."""

    def __init__(self) -> None:
        self.collections: dict[str, _MemoryCollection] = {}
        self.aliases: dict[str, str] = {}

    def _resolve(self, name: str) -> str:
        return self.aliases.get(name, name)

    def ensure_collection(
        self,
        name: str,
        vector_dimensions: Mapping[str, int],
    ) -> None:
        expected = dict(vector_dimensions)
        existing = self.collections.get(name)
        if existing is None:
            self.collections[name] = _MemoryCollection(expected)
            return
        if existing.dimensions != expected:
            raise VectorStoreError(f"collection {name!r} has incompatible vector dimensions")

    def get_point(self, collection: str, point_id: str) -> VectorPoint | None:
        name = self._resolve(collection)
        if name not in self.collections:
            raise VectorStoreError(f"collection does not exist: {name}")
        return self.collections[name].points.get(point_id)

    def upsert(self, collection: str, points: Sequence[VectorPoint]) -> None:
        name = self._resolve(collection)
        if name not in self.collections:
            raise VectorStoreError(f"collection does not exist: {name}")
        target = self.collections[name]
        for point in points:
            for vector_name, vector in point.vectors.items():
                expected = target.dimensions.get(vector_name)
                if expected is None or len(vector) != expected:
                    raise VectorStoreError(f"point {point.point_id!r} has incompatible {vector_name!r} vector")
            if "image" not in point.vectors:
                raise VectorStoreError("every point must contain an image vector")
            target.points[point.point_id] = point

    def search(
        self,
        collection: str,
        vector_name: str,
        query: Sequence[float],
        limit: int,
    ) -> list[SearchHit]:
        name = self._resolve(collection)
        if name not in self.collections:
            raise VectorStoreError(f"collection does not exist: {name}")
        target = self.collections[name]
        expected = target.dimensions.get(vector_name)
        if expected is None or len(query) != expected:
            raise VectorStoreError(f"query has incompatible {vector_name!r} vector")
        scored = [
            SearchHit(
                point_id=point.point_id,
                score=_cosine(query, vector),
                payload=point.payload,
            )
            for point in target.points.values()
            if (vector := point.vectors.get(vector_name)) is not None
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]

    def activate_alias(self, collection: str, alias: str) -> None:
        if collection not in self.collections:
            raise VectorStoreError(f"collection does not exist: {collection}")
        self.aliases[alias] = collection


class QdrantVectorStore:
    def __init__(
        self,
        *,
        url: str,
        api_key: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise VectorStoreError("qdrant-client is required for Qdrant storage") from exc
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout,
            )

    @staticmethod
    def _models() -> Any:
        from qdrant_client.http import models

        return models

    def ensure_collection(
        self,
        name: str,
        vector_dimensions: Mapping[str, int],
    ) -> None:
        models = self._models()
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE,
                    )
                    for vector_name, dimension in vector_dimensions.items()
                },
            )
            return
        info = self.client.get_collection(name)
        existing = info.config.params.vectors
        if not isinstance(existing, dict):
            raise VectorStoreError(f"collection {name!r} does not use named vectors")
        actual = {vector_name: int(params.size) for vector_name, params in existing.items()}
        if actual != dict(vector_dimensions):
            raise VectorStoreError(f"collection {name!r} has incompatible vector dimensions")

    def get_point(self, collection: str, point_id: str) -> VectorPoint | None:
        records = self.client.retrieve(
            collection_name=collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return None
        record = records[0]
        vectors = record.vector if isinstance(record.vector, dict) else {}
        return VectorPoint(
            point_id=str(record.id),
            vectors={name: [float(value) for value in vector] for name, vector in vectors.items()},
            payload=dict(record.payload or {}),
        )

    def upsert(self, collection: str, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        models = self._models()
        self.client.upsert(
            collection_name=collection,
            wait=True,
            points=[
                models.PointStruct(
                    id=point.point_id,
                    vector=dict(point.vectors),
                    payload=dict(point.payload),
                )
                for point in points
            ],
        )

    def search(
        self,
        collection: str,
        vector_name: str,
        query: Sequence[float],
        limit: int,
    ) -> list[SearchHit]:
        response = self.client.query_points(
            collection_name=collection,
            query=list(query),
            using=vector_name,
            limit=limit,
            with_payload=True,
        )
        return [
            SearchHit(
                point_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    def activate_alias(self, collection: str, alias: str) -> None:
        models = self._models()
        operations: list[Any] = []
        aliases = {item.alias_name for item in self.client.get_aliases().aliases}
        if alias in aliases:
            operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection,
                    alias_name=alias,
                )
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=operations)
