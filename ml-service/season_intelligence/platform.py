from __future__ import annotations

import os
import time
import uuid
from typing import Any, Mapping

from season_intelligence.contracts import BusinessConstraints, ScaleRecommendation
from season_intelligence.embeddings import EmbeddingProvider, HTTPEmbeddingProvider, SuppliedEmbeddingProvider
from season_intelligence.forecasting import DemandForecaster
from season_intelligence.optimization import BuyOptimizer
from season_intelligence.ranking import CandidateReranker
from season_intelligence.retrieval import PostgresCatalogRepository


class ScaleEngine:
    def __init__(
        self,
        repository: Any,
        embedding_provider: EmbeddingProvider,
        reranker: CandidateReranker,
        forecaster: DemandForecaster,
        optimizer: BuyOptimizer | None = None,
        model_version: str = "3.0.0",
    ):
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.forecaster = forecaster
        self.optimizer = optimizer or BuyOptimizer()
        self.model_version = model_version

    def recommend(
        self,
        product: Mapping[str, Any],
        constraints: BusinessConstraints,
        retrieval_limit: int = 200,
        top_k: int = 10,
    ) -> ScaleRecommendation:
        started = time.perf_counter()
        embedding = self.embedding_provider.embed(product)
        candidates = self.repository.search(embedding, product, retrieval_limit)
        if not candidates:
            raise LookupError("no eligible historical candidates found after metadata filters")
        matches = self.reranker.rank(product, candidates, top_k)
        forecast = self.forecaster.predict(product, matches)
        optimized = self.optimizer.optimize(forecast, constraints)
        warnings = tuple(dict.fromkeys((*forecast.warnings, *optimized.warnings)))
        result = ScaleRecommendation(
            product_id=str(product["id"]),
            request_id=str(uuid.uuid4()),
            model_version=self.model_version,
            retrieval_mode="pgvector-hnsw-filtered",
            recommendation=optimized,
            forecast=forecast,
            matches=tuple(matches),
            warnings=warnings,
        )
        if hasattr(self.repository, "record_recommendation"):
            self.repository.record_recommendation(
                product,
                serialize_recommendation(result),
                int((time.perf_counter() - started) * 1_000),
            )
        return result


def build_scale_engine() -> ScaleEngine | None:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        return None
    dimension = int(os.getenv("EMBEDDING_DIMENSION", "512"))
    repository = PostgresCatalogRepository(dsn, dimension=dimension)
    if os.getenv("EMBEDDING_BACKEND", "http") == "supplied":
        if os.getenv("APP_ENV", "development") == "production":
            raise RuntimeError("supplied embeddings are disabled in production")
        embedding_provider: EmbeddingProvider = SuppliedEmbeddingProvider(dimension)
    else:
        embedding_provider = HTTPEmbeddingProvider.from_environment()
    require_trained = os.getenv("MODEL_POLICY", "allow_fallback") == "require_trained"
    reranker = CandidateReranker(os.getenv("RANKER_MODEL_PATH"))
    if require_trained and reranker.model is None:
        raise RuntimeError("a trained CatBoost ranker is required by MODEL_POLICY")
    forecaster = DemandForecaster(os.getenv("DEMAND_MODEL_DIR"), require_trained=require_trained)
    return ScaleEngine(repository, embedding_provider, reranker, forecaster)


def serialize_recommendation(result: ScaleRecommendation) -> dict[str, Any]:
    return {
        "requestId": result.request_id,
        "productId": result.product_id,
        "modelVersion": result.model_version,
        "retrievalMode": result.retrieval_mode,
        "recommendation": {
            "quantity": result.recommendation.quantity,
            "low": result.recommendation.low,
            "high": result.recommendation.high,
            "bindingConstraints": list(result.recommendation.binding_constraints),
        },
        "forecast": {
            "p10": round(result.forecast.p10, 2),
            "p50": round(result.forecast.p50, 2),
            "p90": round(result.forecast.p90, 2),
            "model": result.forecast.model_name,
        },
        "matches": [{
            "historicalId": match.candidate.item["id"],
            "score": round(match.score, 4),
            "attributeScore": round(match.attribute_score, 4),
            "vectorScore": round(match.candidate.vector_similarity, 4),
            "normalizedDemand": round(match.candidate.normalized_demand, 2),
            "seasonsObserved": match.candidate.seasons_observed,
            "imageUrl": match.candidate.item.get("imageUrl"),
            "features": {name: round(value, 4) for name, value in match.features.items()},
        } for match in result.matches],
        "warnings": list(result.warnings),
    }
