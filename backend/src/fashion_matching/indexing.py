from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Sequence
from typing import Any

from fashion_matching.encoders import FashionEncoder, model_version_identifier
from fashion_matching.models import (
    IndexSummary,
    ManifestRecord,
    PreparedImage,
    VectorPoint,
)
from fashion_matching.preprocessing import ImagePreprocessor
from fashion_matching.storage import VectorStore

LOGGER = logging.getLogger("turtle.fashion_matching.indexing")
POINT_NAMESPACE = uuid.UUID("e3293886-8a96-4dbc-b5c9-a2dc841981f3")


def point_identifier(image_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, image_id))


def _content_checksum(
    record: ManifestRecord,
    image_checksum: str,
) -> str:
    metadata = json.dumps(
        {
            "text": record.text,
            "attributes": dict(record.attributes),
            "product_id": record.product_id,
            "view": record.view,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{image_checksum}\n{metadata}".encode()).hexdigest()


class CatalogueIndexer:
    def __init__(
        self,
        *,
        encoder: FashionEncoder,
        preprocessor: ImagePreprocessor,
        store: VectorStore,
        collection_prefix: str,
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.store = store
        self.batch_size = batch_size
        self.model_version = model_version_identifier(
            encoder,
            preprocessor.version,
        )
        self.collection = f"{collection_prefix}-{self.model_version}"

    def _vector_dimensions(self) -> dict[str, int]:
        dimensions = {"image": self.encoder.dimension}
        if self.encoder.supports_text:
            dimensions["text"] = self.encoder.dimension
        return dimensions

    def index(
        self,
        records: Sequence[ManifestRecord],
        *,
        activate_alias: str | None = None,
        activate_with_failures: bool = False,
    ) -> IndexSummary:
        started = time.perf_counter()
        summary = IndexSummary(total_images=len(records))
        self.store.ensure_collection(
            self.collection,
            self._vector_dimensions(),
        )
        for offset in range(0, len(records), self.batch_size):
            batch = records[offset : offset + self.batch_size]
            self._index_batch(batch, summary)
            LOGGER.info(
                "index_progress processed=%s total=%s indexed=%s updated=%s skipped=%s failed=%s",
                min(offset + len(batch), len(records)),
                len(records),
                summary.successfully_indexed,
                summary.updated_images,
                summary.skipped_images,
                summary.failed_images,
            )
        summary.processing_time_seconds = time.perf_counter() - started
        if activate_alias and (summary.failed_images == 0 or activate_with_failures):
            self.store.activate_alias(self.collection, activate_alias)
        return summary

    def _index_batch(
        self,
        records: Sequence[ManifestRecord],
        summary: IndexSummary,
    ) -> None:
        pending: list[tuple[ManifestRecord, PreparedImage, str, bool]] = []
        for record in records:
            try:
                prepared = self.preprocessor.prepare(record)
                content_checksum = _content_checksum(
                    record,
                    prepared.checksum,
                )
                existing = self.store.get_point(
                    self.collection,
                    point_identifier(record.image_id),
                )
                if (
                    existing is not None
                    and existing.payload.get("content_checksum") == content_checksum
                    and existing.payload.get("model_version") == self.model_version
                ):
                    summary.skipped_images += 1
                    continue
                pending.append(
                    (
                        record,
                        prepared,
                        content_checksum,
                        existing is not None,
                    )
                )
            except Exception as exc:
                self._record_failure(summary, record, exc)
        if not pending:
            return
        try:
            image_vectors = self.encoder.encode_images([prepared.image for _, prepared, _, _ in pending])
            if len(image_vectors) != len(pending):
                raise RuntimeError("image encoder returned an unexpected batch size")
            text_positions = [index for index, (record, _, _, _) in enumerate(pending) if record.text]
            text_vectors: dict[int, list[float]] = {}
            if text_positions and self.encoder.supports_text:
                encoded_texts = self.encoder.encode_texts([pending[index][0].text or "" for index in text_positions])
                if len(encoded_texts) != len(text_positions):
                    raise RuntimeError("text encoder returned an unexpected batch size")
                text_vectors = dict(
                    zip(
                        text_positions,
                        encoded_texts,
                        strict=True,
                    )
                )
            points = [
                self._point(
                    record,
                    prepared,
                    content_checksum,
                    image_vectors[index],
                    text_vectors.get(index),
                )
                for index, (
                    record,
                    prepared,
                    content_checksum,
                    _,
                ) in enumerate(pending)
            ]
            self.store.upsert(self.collection, points)
            summary.successfully_indexed += sum(not was_update for *_, was_update in pending)
            summary.updated_images += sum(was_update for *_, was_update in pending)
        except Exception as exc:
            for record, _, _, _ in pending:
                self._record_failure(summary, record, exc)

    def _point(
        self,
        record: ManifestRecord,
        prepared: PreparedImage,
        content_checksum: str,
        image_vector: list[float],
        text_vector: list[float] | None,
    ) -> VectorPoint:
        vectors = {"image": image_vector}
        if text_vector is not None:
            vectors["text"] = text_vector
        payload: dict[str, Any] = {
            "product_id": record.product_id,
            "image_id": record.image_id,
            "view": record.view,
            "text_available": bool(record.text),
            "text_embedded": text_vector is not None,
            "attributes": dict(record.attributes),
            "image_checksum": prepared.checksum,
            "content_checksum": content_checksum,
            "model_id": self.encoder.model_id,
            "model_revision": self.encoder.revision,
            "embedding_dimension": self.encoder.dimension,
            "preprocessing_version": self.preprocessor.version,
            "model_version": self.model_version,
        }
        return VectorPoint(
            point_id=point_identifier(record.image_id),
            vectors=vectors,
            payload=payload,
        )

    @staticmethod
    def _record_failure(
        summary: IndexSummary,
        record: ManifestRecord,
        exc: Exception,
    ) -> None:
        summary.failed_images += 1
        summary.failures.append(
            {
                "product_id": record.product_id,
                "image_id": record.image_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        LOGGER.exception(
            "index_image_failed product_id=%s image_id=%s",
            record.product_id,
            record.image_id,
            exc_info=exc,
        )
