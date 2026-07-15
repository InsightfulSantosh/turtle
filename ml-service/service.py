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

from season_intelligence.model import (
    FeatureEncoder,
    attribute_similarity,
    combined_similarity,
    normalized_demand,
    recommend_one,
)


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


class ModelRuntime:
    def __init__(self, path: Path):
        self.path = path
        self.loaded_at = time.time()
        self.artifact = json.loads(path.read_text(encoding="utf-8"))
        self.meta = self.artifact["meta"]
        self.model = dict(self.meta["model"])
        self.history = self.artifact["historical"]
        self.history_by_id = {item["id"]: item for item in self.history}
        self.encoder = FeatureEncoder(self.history)
        self.features = self.encoder.transform(self.history)
        self.targets = np.asarray([float(item["normalizedDemand"]) for item in self.history])

    def recommend(self, payload: RecommendationRequest) -> dict:
        item = payload.product.model_dump(exclude={"visualSimilarities"})
        matches = []
        attribute_weight = float(self.model["attributeWeight"])
        for historical in self.history:
            attribute, breakdown = attribute_similarity(item, historical)
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
            self.encoder,
            self.features,
            model,
        )
        scale = float(self.model["targetSellThrough"]) / payload.settings.targetSellThrough
        for key in ("quantity", "low", "high", "analogueQuantity", "regressionQuantity", "intervalHalfWidth"):
            result[key] = max(100, min(2_000, int(round((result[key] * scale) / 25) * 25)))
        return {
            "requestId": str(uuid.uuid4()),
            "productId": item["id"],
            "modelVersion": self.model["version"],
            "recommendation": result,
            "matches": matches[: int(self.model["topK"])],
            "warnings": ["attribute_only"] if not payload.product.visualSimilarities else [],
        }


RUNTIME = ModelRuntime(ARTIFACT_PATH)
app = FastAPI(
    title="Turtle Season Intelligence API",
    version=RUNTIME.model["version"],
    docs_url="/docs" if os.getenv("ENABLE_API_DOCS", "true").lower() == "true" else None,
    redoc_url=None,
)

allowed_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
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
    response.headers["X-Model-Version"] = RUNTIME.model["version"]
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
