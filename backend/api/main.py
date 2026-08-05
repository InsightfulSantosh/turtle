"""FastAPI transport layer for the current real-data recommendation runtime."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ai.recommendation_engine import RecommendationRuntime
from core.config import paths
from data_pipeline.images import resolve_catalog_image

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
    minimumSimilarity: float = Field(default=0.50, ge=0.10, le=0.90)


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
    origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if origin.strip()
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


@lru_cache(maxsize=2_048)
def _product_image_path(catalog: str, image_id: str) -> Path | None:
    roots = {
        "historical": paths.data / "raw" / "historical_matched_images",
        "upcoming": paths.data / "raw" / "upcoming_ss27_matched_images",
    }
    root = roots.get(catalog)
    if root is None or re.fullmatch(r"[A-Za-z0-9-]{3,80}", image_id) is None:
        return None
    return resolve_catalog_image(root, image_id)


@app.get("/v1/product-images/{catalog}/{image_id}", include_in_schema=False)
def product_image(catalog: str, image_id: str) -> FileResponse:
    """Serve a mapped local product image to the browser without exposing paths."""

    image_path = _product_image_path(catalog, image_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Product image not found")
    return FileResponse(
        image_path,
        headers={"Cache-Control": "public, max-age=86400"},
    )


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
    no_suitable_match = bool(item["recommendation"].get("noSuitableMatch"))
    warnings = list(item.get("modelFlags", []))
    if no_suitable_match:
        warnings.append("no_suitable_visual_match")
    return {
        "productId": item_id,
        "modelVersion": RUNTIME.model["version"],
        "recommendation": item["recommendation"],
        "matches": item["matches"][: int(RUNTIME.model["topK"])],
        "warnings": warnings,
    }


@app.post("/v1/recommendations", dependencies=[Depends(require_api_key)])
def create_recommendation(payload: RecommendationRequest) -> dict:
    product = payload.product.model_dump(exclude={"visualSimilarities"})
    return RUNTIME.recommend(
        product,
        target_sell_through=payload.settings.targetSellThrough,
        minimum_visual_score=payload.settings.minimumSimilarity,
        visual_similarities=payload.product.visualSimilarities,
    )
