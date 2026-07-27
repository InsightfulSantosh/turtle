from __future__ import annotations

import logging
import time
from collections.abc import Mapping

from fashion_matching.encoders import FashionEncoder, model_version_identifier
from fashion_matching.models import ManifestRecord, MatchResult, RankedMatch
from fashion_matching.preprocessing import ImagePreprocessor
from fashion_matching.scoring import (
    SignalWeights,
    attribute_similarity,
    cosine_to_unit_interval,
    fuse_signals,
)
from fashion_matching.storage import SearchHit, VectorStore

LOGGER = logging.getLogger("turtle.fashion_matching.matching")


class FashionMatcher:
    def __init__(
        self,
        *,
        encoder: FashionEncoder,
        preprocessor: ImagePreprocessor,
        store: VectorStore,
        collection: str,
        weights: SignalWeights | None = None,
        candidate_count: int = 100,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> None:
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.store = store
        self.collection = collection
        self.weights = weights or SignalWeights()
        self.candidate_count = candidate_count
        self.top_k = top_k
        self.minimum_score = minimum_score
        self.model_version = model_version_identifier(
            encoder,
            preprocessor.version,
        )

    def match(self, record: ManifestRecord) -> MatchResult:
        started = time.perf_counter()
        warnings: list[str] = []
        try:
            prepared = self.preprocessor.prepare(record)
            image_vector = self.encoder.encode_images([prepared.image])[0]
            image_hits = self.store.search(
                self.collection,
                "image",
                image_vector,
                self.candidate_count,
            )
            text_hits: list[SearchHit] = []
            if record.text:
                if self.encoder.supports_text:
                    text_vector = self.encoder.encode_texts([record.text])[0]
                    text_hits = self.store.search(
                        self.collection,
                        "text",
                        text_vector,
                        self.candidate_count,
                    )
                else:
                    warnings.append("text_encoder_unavailable")
            ranked = self._rank(record, image_hits, text_hits)
            if self.minimum_score is not None:
                ranked = [match for match in ranked if match.final_score >= self.minimum_score]
            ranked = ranked[: self.top_k]
            ranked = [
                RankedMatch(
                    product_id=match.product_id,
                    image_id=match.image_id,
                    view=match.view,
                    rank=index,
                    final_score=match.final_score,
                    image_score=match.image_score,
                    text_score=match.text_score,
                    attribute_score=match.attribute_score,
                    applied_weights=match.applied_weights,
                )
                for index, match in enumerate(ranked, start=1)
            ]
            return MatchResult(
                query_product_id=record.product_id,
                query_image_id=record.image_id,
                model_version=self.model_version,
                matches=tuple(ranked),
                no_suitable_match=not ranked,
                processing_time_ms=(time.perf_counter() - started) * 1_000,
                warnings=tuple(warnings),
            )
        except Exception as exc:
            LOGGER.exception(
                "query_failed product_id=%s image_id=%s",
                record.product_id,
                record.image_id,
                exc_info=exc,
            )
            return MatchResult(
                query_product_id=record.product_id,
                query_image_id=record.image_id,
                model_version=self.model_version,
                matches=(),
                no_suitable_match=True,
                processing_time_ms=(time.perf_counter() - started) * 1_000,
                error=f"{type(exc).__name__}: {exc}",
                warnings=tuple(warnings),
            )

    def _rank(
        self,
        query: ManifestRecord,
        image_hits: list[SearchHit],
        text_hits: list[SearchHit],
    ) -> list[RankedMatch]:
        candidates: dict[str, dict[str, object]] = {}
        self._merge_hits(candidates, image_hits, "image")
        self._merge_hits(candidates, text_hits, "text")
        image_matches: list[RankedMatch] = []
        for point_id, candidate in candidates.items():
            payload = candidate["payload"]
            if not isinstance(payload, Mapping):
                raise ValueError(f"candidate {point_id} has an invalid payload")
            if payload.get("model_version") != self.model_version:
                raise ValueError(
                    "candidate collection contains embeddings from an incompatible model or preprocessing version"
                )
            candidate_attributes = payload.get("attributes")
            attribute_score = attribute_similarity(
                query.attributes,
                candidate_attributes if isinstance(candidate_attributes, Mapping) else {},
            )
            image_score = candidate.get("image")
            text_score = candidate.get("text")
            normalized_image = cosine_to_unit_interval(float(image_score)) if image_score is not None else None
            normalized_text = cosine_to_unit_interval(float(text_score)) if text_score is not None else None
            final_score, applied = fuse_signals(
                {
                    "image": normalized_image,
                    "text": normalized_text,
                    "attributes": attribute_score,
                },
                self.weights,
            )
            image_matches.append(
                RankedMatch(
                    product_id=str(payload["product_id"]),
                    image_id=str(payload["image_id"]),
                    view=(str(payload["view"]) if payload.get("view") is not None else None),
                    rank=0,
                    final_score=round(final_score, 6),
                    image_score=(round(normalized_image, 6) if normalized_image is not None else None),
                    text_score=(round(normalized_text, 6) if normalized_text is not None else None),
                    attribute_score=(round(attribute_score, 6) if attribute_score is not None else None),
                    applied_weights={name: round(weight, 6) for name, weight in applied.items()},
                )
            )
        image_matches.sort(
            key=lambda match: match.final_score,
            reverse=True,
        )
        unique_products: dict[str, RankedMatch] = {}
        for match in image_matches:
            unique_products.setdefault(match.product_id, match)
        return list(unique_products.values())

    @staticmethod
    def _merge_hits(
        candidates: dict[str, dict[str, object]],
        hits: list[SearchHit],
        signal: str,
    ) -> None:
        for hit in hits:
            candidate = candidates.setdefault(
                hit.point_id,
                {"payload": hit.payload},
            )
            candidate[signal] = hit.score
