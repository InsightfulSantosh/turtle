from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from season_intelligence.contracts import BusinessConstraints
from season_intelligence.model import (
    attribute_similarity,
    combined_similarity,
    demand_uncertainty,
    fit_demand_pipeline,
    recommend_one,
)
from season_intelligence.platform import build_scale_engine, serialize_recommendation


APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = Path(os.getenv("TURTLE_MODEL_ARTIFACT", APP_ROOT / "app" / "generated-data.json"))
LOGGER = logging.getLogger("turtle.season_intelligence")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")


class ProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=3, max_length=80)
    itemType: str = Field(min_length=1, max_length=40)
    sleeve: str = Field(default="", max_length=80)
    provision: str = Field(default="", max_length=80)
    pattern: str = Field(default="", max_length=120)
    range: str = Field(default="", max_length=120)
    fit: str = Field(default="", max_length=120)
    fabric: str = Field(default="", max_length=200)
    fashion: str = Field(default="", max_length=80)
    lifecycle: str = Field(default="", max_length=80)
    colour: str = Field(default="", max_length=120)
    mrp: float = Field(gt=0, le=1_000_000)
    visualSimilarities: dict[str, float] = Field(default_factory=dict)


class DecisionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targetSellThrough: float = Field(default=0.70, ge=0.50, le=0.90)


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: ProductInput
    settings: DecisionSettings = Field(default_factory=DecisionSettings)


class ScaleProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=3, max_length=80)
    itemType: str = Field(min_length=1, max_length=40)
    gender: str | None = Field(default=None, max_length=40)
    brand: str | None = Field(default=None, max_length=80)
    sleeve: str = Field(default="", max_length=80)
    provision: str = Field(default="", max_length=80)
    pattern: str = Field(default="", max_length=120)
    range: str = Field(default="", max_length=120)
    fit: str = Field(default="", max_length=120)
    fabric: str = Field(default="", max_length=200)
    fashion: str = Field(default="", max_length=80)
    lifecycle: str = Field(default="", max_length=80)
    colour: str = Field(default="", max_length=120)
    mrp: float = Field(gt=0, le=1_000_000)
    imageUrl: str | None = Field(default=None, max_length=2_048)
    embedding: list[float] | None = Field(default=None, min_length=1, max_length=2_048)


class ScaleConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packSize: int = Field(default=25, ge=1, le=10_000)
    minimumOrder: int = Field(default=100, ge=0, le=10_000_000)
    maximumOrder: int = Field(default=2_000, ge=0, le=10_000_000)
    unitCost: float | None = Field(default=None, gt=0, le=1_000_000)
    budget: float | None = Field(default=None, gt=0, le=10_000_000_000)
    supplierCapacity: int | None = Field(default=None, ge=0, le=10_000_000)

    def to_contract(self) -> BusinessConstraints:
        return BusinessConstraints(
            pack_size=self.packSize,
            minimum_order=self.minimumOrder,
            maximum_order=self.maximumOrder,
            unit_cost=self.unitCost,
            budget=self.budget,
            supplier_capacity=self.supplierCapacity,
        )


class ScaleRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: ScaleProductInput
    constraints: ScaleConstraints = Field(default_factory=ScaleConstraints)
    retrievalLimit: int = Field(default=200, ge=10, le=500)
    topK: int = Field(default=10, ge=1, le=50)


class BatchRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ScaleRecommendationRequest] = Field(min_length=1, max_length=1_000)


class SimilarityFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upcomingItemId: str = Field(min_length=3, max_length=80)
    historicalItemId: str = Field(min_length=3, max_length=80)
    accepted: bool
    relevance: int | None = Field(default=None, ge=0, le=4)
    plannerId: str | None = Field(default=None, max_length=120)
    requestId: str | None = Field(default=None, max_length=80)
    notes: str | None = Field(default=None, max_length=1_000)


class PlannerDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "override"]
    quantity: int = Field(ge=0, le=10_000_000)
    plannerId: str = Field(min_length=1, max_length=120)


class CatalogPerformanceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    season: str = Field(min_length=2, max_length=40)
    channel: str = Field(default="all", max_length=80)
    region: str = Field(default="all", max_length=80)
    orderQuantity: int = Field(ge=0)
    dispatchQuantity: int | None = Field(default=None, ge=0)
    salesQuantity: int = Field(ge=0)
    sellThrough: float | None = Field(default=None, ge=0, le=10)
    normalizedDemand: float = Field(ge=0)
    stockoutDays: int | None = Field(default=None, ge=0)
    markdownRate: float | None = Field(default=None, ge=0, le=1)
    grossMargin: float | None = None
    seasonEnd: str | None = Field(default=None, max_length=20)
    qualityFlags: list[str] = Field(default_factory=list, max_length=50)


class CatalogItemInput(ScaleProductInput):
    isHistorical: bool
    active: bool = True
    embeddingModel: str | None = Field(default=None, max_length=120)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    performance: list[CatalogPerformanceInput] = Field(default_factory=list, max_length=100)


class CatalogBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[CatalogItemInput] = Field(min_length=1, max_length=1_000)


class ModelRuntime:
    def __init__(self, path: Path):
        self.path = path
        self.loaded_at = time.time()
        self.artifact = json.loads(path.read_text(encoding="utf-8"))
        self.meta = self.artifact["meta"]
        self.model = dict(self.meta["model"])
        self.attribute_weights = {
            str(name): float(weight)
            for name, weight in self.model.get("attributeWeights", {}).items()
        } or None
        self.history = self.artifact["historical"]
        self.history_by_id = {item["id"]: item for item in self.history}
        self.targets = np.asarray([float(item["normalizedDemand"]) for item in self.history])
        self.demand_pipeline = fit_demand_pipeline(
            self.history,
            self.targets,
            float(self.model["ridgeAlpha"]),
        )

    def recommend(self, payload: RecommendationRequest) -> dict:
        item = payload.product.model_dump(exclude={"visualSimilarities"})
        matches = []
        attribute_weight = float(self.model["attributeWeight"])
        for historical in self.history:
            attribute, breakdown = attribute_similarity(item, historical, self.attribute_weights)
            visual = payload.product.visualSimilarities.get(historical["id"])
            hybrid = combined_similarity(attribute, visual, attribute_weight)
            matches.append({
                "historicalId": historical["id"],
                "attributeScore": round(attribute, 4),
                "visualScore": visual,
                "hybridScore": round(hybrid, 4),
                "attributeBreakdown": breakdown,
            })
        matches.sort(key=lambda match: match["hybridScore"], reverse=True)
        model = dict(self.model)
        result = recommend_one(
            item,
            self.history,
            matches,
            self.targets,
            self.demand_pipeline,
            model,
        )
        scale = float(self.model["targetSellThrough"]) / payload.settings.targetSellThrough
        for key in ("quantity", "low", "high", "analogueQuantity", "regressionQuantity", "intervalHalfWidth"):
            result[key] = max(100, min(2_000, int(round((result[key] * scale) / 25) * 25)))
        result["uncertaintyRatio"] = round(
            result["intervalHalfWidth"] / max(result["quantity"], 1),
            4,
        )
        result["demandUncertainty"] = demand_uncertainty(
            result["quantity"],
            result["intervalHalfWidth"],
        )
        return {
            "requestId": str(uuid.uuid4()),
            "productId": item["id"],
            "modelVersion": self.model["version"],
            "recommendation": result,
            "matches": matches[: int(self.model["topK"])],
            "warnings": ["attribute_only"] if not payload.product.visualSimilarities else [],
        }


RUNTIME = ModelRuntime(ARTIFACT_PATH)
SCALE_ENGINE = build_scale_engine()
app = FastAPI(
    title="Turtle Season Intelligence API",
    version=RUNTIME.model["version"],
    docs_url="/docs" if os.getenv("ENABLE_API_DOCS", "true").lower() == "true" else None,
    redoc_url=None,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("TURTLE_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Model-Version"] = (
        SCALE_ENGINE.model_version
        if request.url.path.startswith("/v2") and SCALE_ENGINE is not None
        else RUNTIME.model["version"]
    )
    LOGGER.info(
        "request_id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


@app.get("/healthz", include_in_schema=False)
def health() -> dict:
    return {"status": "ok", "modelVersion": RUNTIME.model["version"]}


@app.get("/v2/health/ready", include_in_schema=False)
def scale_readiness() -> dict:
    configured = SCALE_ENGINE is not None
    database_ready = False
    if configured:
        try:
            database_ready = bool(SCALE_ENGINE.repository.ready())
        except Exception:  # readiness must fail closed without leaking infrastructure details
            database_ready = False
    return {
        "status": "ready" if configured and database_ready else "sample_only",
        "scaleRuntimeConfigured": configured,
        "databaseReady": database_ready,
        "sampleModelVersion": RUNTIME.model["version"],
        "scaleModelVersion": SCALE_ENGINE.model_version if configured else "3.0.0",
    }


@app.get("/v1/model", dependencies=[Depends(require_api_key)])
def model_card() -> dict:
    return {
        "model": RUNTIME.model,
        "visualMethod": RUNTIME.meta["visualMethod"],
        "dataQuality": RUNTIME.meta["dataQuality"],
        "imageCoverage": {
            "historical": RUNTIME.meta["historicalImageCoverage"],
            "upcoming": RUNTIME.meta["upcomingImageCoverage"],
        },
    }


@app.get("/v1/recommendations/{item_id}", dependencies=[Depends(require_api_key)])
def existing_recommendation(item_id: str) -> dict:
    item = next((row for row in RUNTIME.artifact["upcoming"] if row["id"] == item_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="Upcoming item not found")
    return {
        "productId": item_id,
        "modelVersion": RUNTIME.model["version"],
        "recommendation": item["recommendation"],
        "matches": item["matches"][: int(RUNTIME.model["topK"])],
        "warnings": item.get("modelFlags", []),
    }


@app.post("/v1/recommendations", dependencies=[Depends(require_api_key)])
def create_recommendation(payload: RecommendationRequest) -> dict:
    return RUNTIME.recommend(payload)


def require_scale_engine():
    if SCALE_ENGINE is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Scale runtime is not configured. Set DATABASE_URL and EMBEDDING_SERVICE_URL; "
                "v1 sample endpoints remain available."
            ),
        )
    return SCALE_ENGINE


@app.post("/v2/recommendations", dependencies=[Depends(require_api_key)])
def create_scale_recommendation(payload: ScaleRecommendationRequest) -> dict:
    engine = require_scale_engine()
    try:
        result = engine.recommend(
            payload.product.model_dump(exclude_none=True),
            payload.constraints.to_contract(),
            retrieval_limit=payload.retrievalLimit,
            top_k=payload.topK,
        )
    except LookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("scale recommendation dependency failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return serialize_recommendation(result)


@app.post("/v2/recommendations:batch", status_code=202, dependencies=[Depends(require_api_key)])
def create_recommendation_batch(payload: BatchRecommendationRequest) -> dict:
    engine = require_scale_engine()
    job_id = engine.repository.create_job("recommendation_batch", payload.model_dump(exclude_none=True))
    return {"jobId": job_id, "status": "queued", "itemCount": len(payload.items)}


@app.post("/v2/catalog/items:batch", status_code=202, dependencies=[Depends(require_api_key)])
def create_catalog_batch(payload: CatalogBatchRequest) -> dict:
    engine = require_scale_engine()
    job_id = engine.repository.create_job("catalog_ingestion", payload.model_dump(exclude_none=True))
    return {"jobId": job_id, "status": "queued", "itemCount": len(payload.items)}


@app.get("/v2/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_batch_job(job_id: str) -> dict:
    engine = require_scale_engine()
    job = engine.repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/v2/feedback/similarity", status_code=201, dependencies=[Depends(require_api_key)])
def create_similarity_feedback(payload: SimilarityFeedbackRequest) -> dict:
    engine = require_scale_engine()
    feedback_id = engine.repository.record_feedback(payload.model_dump(exclude_none=True))
    return {"feedbackId": feedback_id, "status": "recorded"}


@app.post("/v2/recommendations/{request_id}/decision", dependencies=[Depends(require_api_key)])
def create_planner_decision(request_id: str, payload: PlannerDecisionRequest) -> dict:
    engine = require_scale_engine()
    recorded = engine.repository.record_planner_decision(request_id, payload.model_dump())
    if not recorded:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"requestId": request_id, "status": "recorded", "decision": payload.decision}
