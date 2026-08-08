"""FastAPI surface for versioned catalogue uploads and planner results."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.requests import ClientDisconnect

from analysis_service.catalogs import SUPPORTED_CATALOGUE_SUFFIXES
from analysis_service.processor import AnalysisProcessor
from analysis_service.store import RunStore
from core.config import paths

SERVICE_ROOT = Path(os.getenv("TURTLE_ANALYSIS_ROOT", paths.data / "analysis-service"))
store = RunStore(SERVICE_ROOT)
processor = AnalysisProcessor(store)

app = FastAPI(title="Turtle Season Intelligence API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip() for origin in os.getenv("TURTLE_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "HEAD"],
    allow_headers=["Content-Type", "Upload-Length", "Upload-Offset"],
    expose_headers=["Upload-Length", "Upload-Offset"],
)

# Catalogue workbooks commonly contain business punctuation in their names
# (for example, ``ORDERING & SALE THRU DATA.xlsb``).  Ampersands and the other
# characters below are harmless after the path-component check in
# ``_upload_path`` and are URL-encoded by the client.  Keeping this allow-list
# explicit still rejects separators, control characters, shell metacharacters,
# and traversal names.
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ &()+,-]{0,179}$")
SAFE_PRODUCT_ID = re.compile(r"^[A-Za-z0-9-]{3,80}$")
CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


class RunCreate(BaseModel):
    mode: Literal["full_replace", "reuse_historical"]


def _public_run(run: dict) -> dict:
    return {
        "id": run["id"],
        "mode": run["mode"],
        "status": run["status"],
        "stage": run["stage"],
        "progress": run["progress"],
        "message": run["message"],
        "error": run["error"],
        "buildId": run["build_id"],
        "cancelRequested": run["cancelRequested"],
        "processedCount": run["processed_count"],
        "totalCount": run["total_count"],
        "cacheHits": run["cache_hits"],
        "etaSeconds": run["eta_seconds"],
        "historicalProcessed": run["historical_processed"],
        "historicalTotal": run["historical_total"],
        "upcomingProcessed": run["upcoming_processed"],
        "upcomingTotal": run["upcoming_total"],
        "createdAt": run["created_at"],
        "updatedAt": run["updated_at"],
        "startedAt": run["analysis_started_at"],
        "completedAt": run["completed_at"],
        "validationReportUrl": f"/api/runs/{run['id']}/validation-report" if run["issues_path"] else None,
    }


def _run_or_404(run_id: str) -> dict:
    try:
        return store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


@app.on_event("startup")
def cleanup_staging() -> None:
    store.cleanup_failed_staging()
    store.cleanup_paths(store.purge_superseded_run_history())
    processor.resume_incomplete()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/historical/active")
def active_historical() -> dict | None:
    active = store.active_build()
    if active is None:
        return None
    return {
        "id": active["historical_version_id"],
        "createdAt": active["historical_created_at"],
        "productCount": active["historical_product_count"],
        "imageCoverage": active["historical_image_coverage"],
        "modelVersion": active["model_version"],
    }


@app.post("/api/runs", status_code=201)
def create_run(payload: RunCreate) -> dict:
    try:
        return _public_run(store.create_run(payload.mode))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def _upload_path(run: dict, catalog: str, kind: str, filename: str) -> Path:
    if catalog not in {"historical", "upcoming"} or kind not in {"catalogue", "csv", "images"}:
        raise HTTPException(404, "upload target not found")
    if run["mode"] == "reuse_historical" and catalog == "historical":
        raise HTTPException(409, "this run reuses the active historical version")
    safe = Path(filename).name
    if safe != filename or not SAFE_FILENAME.fullmatch(safe):
        raise HTTPException(400, "unsafe filename")
    if kind in {"catalogue", "csv"} and Path(safe).suffix.lower() not in SUPPORTED_CATALOGUE_SUFFIXES:
        raise HTTPException(415, "catalogue must be CSV, XLSX, XLSM, XLS, XLSB, or ODS")
    if kind == "images" and Path(safe).suffix.lower() not in CONTENT_TYPES:
        raise HTTPException(415, "image must be JPEG, PNG, or WebP")
    target_kind = "catalogue" if kind in {"catalogue", "csv"} else kind
    target = store.staging_path(run["id"]) / catalog / target_kind / safe
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@app.head("/api/runs/{run_id}/uploads/{catalog}/{kind}/{filename}")
def upload_offset(run_id: str, catalog: str, kind: str, filename: str) -> Response:
    run = _run_or_404(run_id)
    target = _upload_path(run, catalog, kind, filename)
    size = target.stat().st_size if target.is_file() else 0
    return Response(status_code=204, headers={"Upload-Offset": str(size)})


@app.put("/api/runs/{run_id}/uploads/{catalog}/{kind}/{filename}")
async def upload_file(
    run_id: str,
    catalog: str,
    kind: str,
    filename: str,
    request: Request,
    upload_offset: int = Header(0, alias="Upload-Offset"),
    upload_length: int | None = Header(None, alias="Upload-Length"),
) -> Response:
    run = _run_or_404(run_id)
    if run["status"] != "uploading":
        raise HTTPException(409, "run no longer accepts uploads")
    target = _upload_path(run, catalog, kind, filename)
    current = target.stat().st_size if target.exists() else 0
    if current != upload_offset:
        return Response(status_code=409, headers={"Upload-Offset": str(current)})
    limit = 100 * 1024 * 1024 if kind in {"catalogue", "csv"} else 16 * 1024 * 1024
    if upload_length is not None and upload_length > limit:
        raise HTTPException(413, f"file exceeds {limit} byte limit")
    written = current
    try:
        with target.open("ab") as handle:
            async for chunk in request.stream():
                written += len(chunk)
                if written > limit:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(413, f"file exceeds {limit} byte limit")
                handle.write(chunk)
    except ClientDisconnect:
        # The browser, or the dev-server proxy in front of it, dropped the
        # connection part-way through the body. That is not a server fault and
        # must not surface as a 500: the bytes already appended stay on disk so
        # the next HEAD reports the new offset and the client resumes from
        # there, which is the whole point of the offset protocol.
        partial = target.stat().st_size if target.is_file() else 0
        return Response(status_code=499, headers={"Upload-Offset": str(partial)})
    if upload_length is not None and written > upload_length:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "uploaded more bytes than declared")
    return Response(
        status_code=204, headers={"Upload-Offset": str(written), "Upload-Length": str(upload_length or written)}
    )


@app.post("/api/runs/{run_id}/complete-upload", status_code=202)
def complete_upload(run_id: str) -> dict:
    _run_or_404(run_id)
    try:
        processor.submit(run_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _public_run(store.get_run(run_id))


@app.get("/api/runs/active")
def active_run() -> dict | None:
    run = store.active_run()
    return _public_run(run) if run is not None else None


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _public_run(_run_or_404(run_id))


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    _run_or_404(run_id)

    async def events():
        previous = ""
        while True:
            run = _public_run(_run_or_404(run_id))
            payload = json.dumps(run, separators=(",", ":"))
            if payload != previous:
                yield f"data: {payload}\n\n"
                previous = payload
            if run["status"] in {"succeeded", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.75)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    _run_or_404(run_id)
    return _public_run(store.cancel_run(run_id))


@app.get("/api/runs/{run_id}/validation-report")
def validation_report(run_id: str) -> FileResponse:
    run = _run_or_404(run_id)
    path = Path(run["issues_path"]) if run["issues_path"] else None
    if path is None or not path.is_file():
        raise HTTPException(404, "validation report is not available")
    return FileResponse(path, media_type="text/csv", filename=f"turtle-validation-{run_id}.csv")


def _artifact_for(build_id: str) -> dict:
    try:
        return store.artifact(build_id)
    except KeyError as exc:
        raise HTTPException(404, "build not found") from exc


@app.get("/api/active")
def active_build() -> dict:
    active = store.active_build()
    if active is None:
        raise HTTPException(404, "no analysis has been uploaded yet")
    return {
        "id": active["id"],
        "createdAt": active["created_at"],
        "artifactUrl": f"/api/builds/{active['id']}/artifact",
        "historicalVersionId": active["historical_version_id"],
        "historicalReusable": True,
    }


@app.get("/api/builds/{build_id}/artifact")
def build_artifact(build_id: str) -> JSONResponse:
    return JSONResponse(_artifact_for(build_id), headers={"Cache-Control": "no-cache"})


@app.get("/api/runs/{run_id}/results")
def run_results(run_id: str, cursor: int = 0, limit: int = 100) -> dict:
    run = _run_or_404(run_id)
    if run["status"] != "succeeded" or not run["build_id"]:
        return {"items": [], "nextCursor": None, "finalized": 0}
    artifact = _artifact_for(run["build_id"])
    bounded = max(1, min(limit, 500))
    items = artifact["upcoming"][cursor : cursor + bounded]
    next_cursor = cursor + len(items)
    return {
        "items": items,
        "nextCursor": next_cursor if next_cursor < len(artifact["upcoming"]) else None,
        "finalized": len(artifact["upcoming"]),
    }


def _image_root(build_id: str, catalog: str) -> Path:
    if catalog not in {"historical", "upcoming"}:
        raise HTTPException(404, "image catalog not found")
    build = store.build(build_id)
    if build is None:
        raise HTTPException(404, "build not found")
    return Path(build[f"{catalog}_path"]) / "images"


@app.get("/api/builds/{build_id}/images/{catalog}/{product_id}")
def product_image(build_id: str, catalog: str, product_id: str) -> FileResponse:
    if not SAFE_PRODUCT_ID.fullmatch(product_id):
        raise HTTPException(404, "image not found")
    root = _image_root(build_id, catalog)
    match = (
        next(
            (
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() in CONTENT_TYPES and path.stem.upper() == product_id.upper()
            ),
            None,
        )
        if root.is_dir()
        else None
    )
    if match is None:
        raise HTTPException(404, "image not found")
    return FileResponse(
        match,
        media_type=CONTENT_TYPES[match.suffix.lower()],
        headers={"Cache-Control": "public, max-age=86400", "X-Content-Type-Options": "nosniff"},
    )


def _safe_cell(value: object) -> object:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


@app.get("/api/builds/{build_id}/exports/{format}")
def export_build(build_id: str, format: Literal["csv", "json"]):
    artifact = _artifact_for(build_id)
    if format == "json":
        return JSONResponse(artifact, headers={"Content-Disposition": f'attachment; filename="turtle-{build_id}.json"'})
    output = io.StringIO()
    fields = ["product_id", "item_type", "match_confidence", "historical_id", "visual_score", "recommended_quantity"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for item in artifact["upcoming"]:
        match = item.get("matches", [{}])[0] if item.get("matches") else {}
        recommendation = item.get("recommendation", {})
        writer.writerow(
            {
                "product_id": _safe_cell(item.get("id", "")),
                "item_type": _safe_cell(item.get("itemType", "")),
                "match_confidence": recommendation.get("matchConfidence", "Low"),
                "historical_id": _safe_cell(match.get("historicalId", "")),
                "visual_score": match.get("visualScore", ""),
                "recommended_quantity": recommendation.get("quantity", 0),
            }
        )
    return Response(
        output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="turtle-{build_id}.csv"'},
    )
