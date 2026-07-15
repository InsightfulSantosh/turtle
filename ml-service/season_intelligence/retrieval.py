from __future__ import annotations

import json
import math
from typing import Any, Mapping, Protocol, Sequence

from season_intelligence.contracts import Candidate


class CandidateRepository(Protocol):
    def search(self, embedding: Sequence[float], product: Mapping[str, Any], limit: int = 200) -> list[Candidate]: ...


def vector_literal(values: Sequence[float], dimension: int) -> str:
    if len(values) != dimension:
        raise ValueError(f"expected {dimension}-dimension embedding, received {len(values)}")
    cleaned = [float(value) for value in values]
    if not all(math.isfinite(value) for value in cleaned):
        raise ValueError("embedding contains a non-finite value")
    return "[" + ",".join(f"{value:.8g}" for value in cleaned) + "]"


class PostgresCatalogRepository:
    """Filtered HNSW retrieval and durable operational storage in PostgreSQL/pgvector."""

    def __init__(self, dsn: str, dimension: int = 512, min_pool: int = 1, max_pool: int = 8):
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - exercised in the scale container
            raise RuntimeError("install requirements-scale.txt to use PostgreSQL scale mode") from exc
        self.dimension = dimension
        self.pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_pool,
            max_size=max_pool,
            kwargs={"row_factory": dict_row, "autocommit": False},
            open=True,
        )

    def ready(self) -> bool:
        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT EXISTS (SELECT 1 FROM catalog_items WHERE is_historical AND embedding IS NOT NULL) AS ready"
            ).fetchone()
            return bool(row["ready"])

    def search(self, embedding: Sequence[float], product: Mapping[str, Any], limit: int = 200) -> list[Candidate]:
        vector = vector_literal(embedding, self.dimension)
        mrp = max(float(product.get("mrp") or 1), 1)
        params = {
            "embedding": vector,
            "item_type": str(product.get("itemType") or ""),
            "gender": product.get("gender"),
            "brand": product.get("brand"),
            "price_low": mrp / 1.75,
            "price_high": mrp * 1.75,
            "limit": min(max(int(limit), 10), 500),
        }
        query = """
            WITH nearest AS (
                SELECT c.*, 1 - (c.embedding <=> %(embedding)s::halfvec) AS vector_similarity
                FROM catalog_items c
                WHERE c.is_historical
                  AND c.active
                  AND c.embedding IS NOT NULL
                  AND c.item_type = %(item_type)s
                  AND (%(gender)s IS NULL OR c.gender = %(gender)s)
                  AND (%(brand)s IS NULL OR c.brand = %(brand)s)
                  AND c.mrp BETWEEN %(price_low)s AND %(price_high)s
                ORDER BY c.embedding <=> %(embedding)s::halfvec
                LIMIT %(limit)s
            )
            SELECT n.*,
                   COALESCE(p.normalized_demand, 0) AS normalized_demand,
                   COALESCE(p.seasons_observed, 0) AS seasons_observed,
                   COALESCE(f.feedback_score, 0.5) AS feedback_score
            FROM nearest n
            LEFT JOIN LATERAL (
                SELECT AVG(normalized_demand) AS normalized_demand,
                       COUNT(DISTINCT season) AS seasons_observed
                FROM item_performance
                WHERE item_id = n.item_id
            ) p ON TRUE
            LEFT JOIN LATERAL (
                SELECT AVG(CASE WHEN accepted THEN 1.0 ELSE 0.0 END) AS feedback_score
                FROM similarity_feedback
                WHERE historical_item_id = n.item_id
            ) f ON TRUE
            ORDER BY n.vector_similarity DESC
        """
        with self.pool.connection() as connection, connection.transaction():
            connection.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            rows = connection.execute(query, params).fetchall()
        candidates: list[Candidate] = []
        for row in rows:
            item = {
                "id": row["item_id"],
                "itemType": row["item_type"],
                "gender": row["gender"],
                "brand": row["brand"],
                "sleeve": row["sleeve"],
                "provision": row["provision"],
                "pattern": row["pattern"],
                "range": row["range_name"],
                "fit": row["fit"],
                "fabric": row["fabric"],
                "fashion": row["fashion"],
                "lifecycle": row["lifecycle"],
                "colour": row["colour"],
                "mrp": float(row["mrp"]),
                "imageUrl": row["image_url"],
            }
            candidates.append(Candidate(
                item=item,
                vector_similarity=max(0.0, min(1.0, float(row["vector_similarity"]))),
                normalized_demand=max(0.0, float(row["normalized_demand"])),
                seasons_observed=int(row["seasons_observed"]),
                feedback_score=float(row["feedback_score"]),
            ))
        return candidates

    def create_job(self, job_type: str, payload: Mapping[str, Any]) -> str:
        with self.pool.connection() as connection, connection.transaction():
            row = connection.execute(
                "INSERT INTO batch_jobs (job_type, payload) VALUES (%s, %s::jsonb) RETURNING job_id",
                (job_type, json.dumps(payload)),
            ).fetchone()
            return str(row["job_id"])

    def upsert_catalog_item(self, product: Mapping[str, Any], embedding: Sequence[float]) -> None:
        vector = vector_literal(embedding, self.dimension)
        values = (
            product["id"], bool(product.get("isHistorical", False)), bool(product.get("active", True)),
            product["itemType"], product.get("gender"), product.get("brand"), product.get("sleeve"),
            product.get("provision"), product.get("pattern"), product.get("range"), product.get("fit"),
            product.get("fabric"), product.get("fashion"), product.get("lifecycle"), product.get("colour"),
            product["mrp"], product.get("imageUrl"), vector, product.get("embeddingModel", "fashion-clip"),
            json.dumps(product.get("metadata", {})),
        )
        with self.pool.connection() as connection, connection.transaction():
            connection.execute("""
                INSERT INTO catalog_items
                    (item_id, is_historical, active, item_type, gender, brand, sleeve, provision, pattern,
                     range_name, fit, fabric, fashion, lifecycle, colour, mrp, image_url, embedding,
                     embedding_model, embedding_created_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s::halfvec, %s, now(), %s::jsonb)
                ON CONFLICT (item_id) DO UPDATE SET
                    is_historical=EXCLUDED.is_historical, active=EXCLUDED.active, item_type=EXCLUDED.item_type,
                    gender=EXCLUDED.gender, brand=EXCLUDED.brand, sleeve=EXCLUDED.sleeve,
                    provision=EXCLUDED.provision, pattern=EXCLUDED.pattern, range_name=EXCLUDED.range_name,
                    fit=EXCLUDED.fit, fabric=EXCLUDED.fabric, fashion=EXCLUDED.fashion,
                    lifecycle=EXCLUDED.lifecycle, colour=EXCLUDED.colour, mrp=EXCLUDED.mrp,
                    image_url=EXCLUDED.image_url, embedding=EXCLUDED.embedding,
                    embedding_model=EXCLUDED.embedding_model, embedding_created_at=now(),
                    metadata=EXCLUDED.metadata, updated_at=now()
            """, values)
            for performance in product.get("performance", []):
                connection.execute("""
                    INSERT INTO item_performance
                        (item_id, season, channel, region, order_quantity, dispatch_quantity,
                         sales_quantity, sell_through, normalized_demand, stockout_days,
                         markdown_rate, gross_margin, season_end, quality_flags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (item_id, season, channel, region) DO UPDATE SET
                        order_quantity=EXCLUDED.order_quantity, dispatch_quantity=EXCLUDED.dispatch_quantity,
                        sales_quantity=EXCLUDED.sales_quantity, sell_through=EXCLUDED.sell_through,
                        normalized_demand=EXCLUDED.normalized_demand, stockout_days=EXCLUDED.stockout_days,
                        markdown_rate=EXCLUDED.markdown_rate, gross_margin=EXCLUDED.gross_margin,
                        season_end=EXCLUDED.season_end, quality_flags=EXCLUDED.quality_flags
                """, (
                    product["id"], performance["season"], performance.get("channel", "all"),
                    performance.get("region", "all"), performance["orderQuantity"],
                    performance.get("dispatchQuantity"), performance["salesQuantity"],
                    performance.get("sellThrough"), performance["normalizedDemand"],
                    performance.get("stockoutDays"), performance.get("markdownRate"),
                    performance.get("grossMargin"), performance.get("seasonEnd"),
                    performance.get("qualityFlags", []),
                ))

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.pool.connection() as connection:
            row = connection.execute(
                """SELECT job_id, job_type, status, progress, result, error, created_at, updated_at
                   FROM batch_jobs WHERE job_id = %s""",
                (job_id,),
            ).fetchone()
            return dict(row) if row else None

    def claim_job(self) -> dict[str, Any] | None:
        with self.pool.connection() as connection, connection.transaction():
            row = connection.execute("""
                UPDATE batch_jobs SET status = 'running', started_at = now(), updated_at = now()
                WHERE job_id = (
                    SELECT job_id FROM batch_jobs WHERE status = 'queued'
                    ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
                )
                RETURNING job_id, job_type, payload
            """).fetchone()
            return dict(row) if row else None

    def finish_job(
        self,
        job_id: str,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        status = "failed" if error else "succeeded"
        with self.pool.connection() as connection, connection.transaction():
            connection.execute(
                """UPDATE batch_jobs SET status=%s, progress=100, result=%s::jsonb, error=%s,
                   finished_at=now(), updated_at=now() WHERE job_id=%s""",
                (status, json.dumps(result) if result is not None else None, error, job_id),
            )

    def update_job_progress(self, job_id: str, progress: int) -> None:
        with self.pool.connection() as connection, connection.transaction():
            connection.execute(
                "UPDATE batch_jobs SET progress=%s, updated_at=now() WHERE job_id=%s AND status='running'",
                (min(max(progress, 0), 99), job_id),
            )

    def record_feedback(self, payload: Mapping[str, Any]) -> str:
        with self.pool.connection() as connection, connection.transaction():
            row = connection.execute("""
                INSERT INTO similarity_feedback
                    (upcoming_item_id, historical_item_id, accepted, relevance, planner_id, recommendation_id, notes)
                VALUES (%s, %s, %s, %s, %s,
                        (SELECT recommendation_id FROM recommendation_runs WHERE request_id = %s), %s)
                RETURNING feedback_id
            """, (
                payload["upcomingItemId"], payload["historicalItemId"], payload["accepted"],
                payload.get("relevance"), payload.get("plannerId"),
                payload.get("requestId"), payload.get("notes"),
            )).fetchone()
            return str(row["feedback_id"])

    def record_recommendation(
        self,
        product: Mapping[str, Any],
        response: Mapping[str, Any],
        latency_ms: int,
    ) -> str:
        with self.pool.connection() as connection, connection.transaction():
            row = connection.execute("""
                INSERT INTO recommendation_runs
                    (request_id, upcoming_item_id, model_version, status, request, response, latency_ms)
                VALUES (%s, %s, %s, 'succeeded', %s::jsonb, %s::jsonb, %s)
                RETURNING recommendation_id
            """, (
                response["requestId"], product["id"], response["modelVersion"],
                json.dumps(product), json.dumps(response), latency_ms,
            )).fetchone()
            return str(row["recommendation_id"])

    def record_planner_decision(self, request_id: str, payload: Mapping[str, Any]) -> bool:
        status = "approved" if payload["decision"] == "approve" else "overridden"
        with self.pool.connection() as connection, connection.transaction():
            cursor = connection.execute("""
                UPDATE recommendation_runs
                SET status=%s, planner_id=%s, approved_quantity=%s
                WHERE request_id=%s
            """, (status, payload["plannerId"], payload["quantity"], request_id))
            return cursor.rowcount == 1
