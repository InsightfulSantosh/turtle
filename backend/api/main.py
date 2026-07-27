"""FastAPI transport layer for the current real-data recommendation runtime."""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from ai.recommendation_engine import RecommendationRuntime
from core.config import paths


LOGGER = logging.getLogger("turtle.backend")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)


class ProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=3, max_length=80)
    itemType: str = Field(min_length=1, max_length=40)
    design: str = Field(default="", max_length=120)
    categoryType: str = Field(default="", max_length=120)
    fabric: str = Field(default="", max_length=200)
    colour: str = Field(default="", max_length=120)
    visualSimilarities: dict[str, float] = Field(default_factory=dict)


class DecisionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    targetSellThrough: float = Field(default=0.70, ge=0.50, le=0.90)


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: ProductInput
    settings: DecisionSettings = Field(default_factory=DecisionSettings)


RUNTIME = RecommendationRuntime(paths.model_artifact)
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
    response.headers["X-Model-Version"] = RUNTIME.model["version"]
    LOGGER.info(
        "request_id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1_000,
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
    item = next(
        (row for row in RUNTIME.artifact["upcoming"] if row["id"] == item_id),
        None,
    )
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
    product = payload.product.model_dump(exclude={"visualSimilarities"})
    return RUNTIME.recommend(
        product,
        target_sell_through=payload.settings.targetSellThrough,
        visual_similarities=payload.product.visualSimilarities,
    )
