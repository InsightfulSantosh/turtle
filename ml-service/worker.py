from __future__ import annotations

import logging
import os
import signal
import time
from typing import Any, Mapping

from season_intelligence.contracts import BusinessConstraints
from season_intelligence.platform import build_scale_engine, serialize_recommendation


LOGGER = logging.getLogger("turtle.worker")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
RUNNING = True


def _constraints(payload: Mapping[str, Any]) -> BusinessConstraints:
    return BusinessConstraints(
        pack_size=int(payload.get("packSize", 25)),
        minimum_order=int(payload.get("minimumOrder", 100)),
        maximum_order=int(payload.get("maximumOrder", 2_000)),
        unit_cost=payload.get("unitCost"),
        budget=payload.get("budget"),
        supplier_capacity=payload.get("supplierCapacity"),
    )


def process_job(engine, job: Mapping[str, Any]) -> dict[str, Any]:
    payload = job["payload"]
    items = payload.get("items", [])
    results: list[dict[str, Any]] = []
    if job["job_type"] == "recommendation_batch":
        for index, item in enumerate(items):
            result = engine.recommend(
                item["product"],
                _constraints(item.get("constraints", {})),
                retrieval_limit=int(item.get("retrievalLimit", 200)),
                top_k=int(item.get("topK", 10)),
            )
            results.append(serialize_recommendation(result))
            engine.repository.update_job_progress(str(job["job_id"]), int((index + 1) / len(items) * 99))
    elif job["job_type"] == "catalog_ingestion":
        for index, item in enumerate(items):
            embedding = engine.embedding_provider.embed(item)
            engine.repository.upsert_catalog_item(item, embedding)
            results.append({"itemId": item["id"], "status": "upserted"})
            engine.repository.update_job_progress(str(job["job_id"]), int((index + 1) / len(items) * 99))
    else:
        raise ValueError(f"unsupported job type: {job['job_type']}")
    return {"itemCount": len(results), "items": results}


def stop_worker(_signum, _frame) -> None:
    global RUNNING
    RUNNING = False


def main() -> None:
    engine = build_scale_engine()
    if engine is None:
        raise RuntimeError("DATABASE_URL is required for the worker")
    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)
    poll_seconds = max(float(os.getenv("WORKER_POLL_SECONDS", "2")), 0.25)
    while RUNNING:
        job = engine.repository.claim_job()
        if job is None:
            time.sleep(poll_seconds)
            continue
        job_id = str(job["job_id"])
        try:
            result = process_job(engine, job)
            engine.repository.finish_job(job_id, result=result)
        except Exception as exc:
            LOGGER.exception("job_id=%s failed", job_id)
            engine.repository.finish_job(job_id, error=str(exc)[:2_000])


if __name__ == "__main__":
    main()
