"""SQLite lifecycle store with transactional active-build replacement.

The SQL and storage boundaries are intentionally isolated so production can
point the same service contract at PostgreSQL/MinIO without changing the API
or analysis workflow. SQLite is the durable single-node on-prem default.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")


def ist_now() -> str:
    return datetime.now(IST).isoformat()


class RunStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "analysis.sqlite3"
        self._write_lock = threading.RLock()
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as db:
            db.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS historical_versions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    path TEXT NOT NULL,
                    product_count INTEGER NOT NULL,
                    image_coverage INTEGER NOT NULL,
                    model_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS upcoming_versions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    path TEXT NOT NULL,
                    product_count INTEGER NOT NULL,
                    image_coverage INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS builds (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    historical_version_id TEXT NOT NULL REFERENCES historical_versions(id),
                    upcoming_version_id TEXT NOT NULL REFERENCES upcoming_versions(id),
                    artifact_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active_build (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    build_id TEXT NOT NULL REFERENCES builds(id)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    error TEXT,
                    issues_path TEXT,
                    build_id TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    processed_count INTEGER NOT NULL DEFAULT 0,
                    total_count INTEGER NOT NULL DEFAULT 0,
                    cache_hits INTEGER NOT NULL DEFAULT 0,
                    eta_seconds INTEGER,
                    historical_processed INTEGER NOT NULL DEFAULT 0,
                    historical_total INTEGER NOT NULL DEFAULT 0,
                    upcoming_processed INTEGER NOT NULL DEFAULT 0,
                    upcoming_total INTEGER NOT NULL DEFAULT 0,
                    analysis_started_at TEXT,
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
            for name, definition in {
                "processed_count": "INTEGER NOT NULL DEFAULT 0",
                "total_count": "INTEGER NOT NULL DEFAULT 0",
                "cache_hits": "INTEGER NOT NULL DEFAULT 0",
                "eta_seconds": "INTEGER",
                "historical_processed": "INTEGER NOT NULL DEFAULT 0",
                "historical_total": "INTEGER NOT NULL DEFAULT 0",
                "upcoming_processed": "INTEGER NOT NULL DEFAULT 0",
                "upcoming_total": "INTEGER NOT NULL DEFAULT 0",
                "analysis_started_at": "TEXT",
                "completed_at": "TEXT",
            }.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
            # Older databases stored ISO timestamps in UTC. Keep every SQLite
            # table human-readable in the business timezone requested by the
            # planner; the explicit +05:30 offset preserves absolute ordering.
            for table, timestamp_columns in {
                "historical_versions": ("created_at",),
                "upcoming_versions": ("created_at",),
                "builds": ("created_at",),
                "runs": ("created_at", "updated_at", "analysis_started_at", "completed_at"),
            }.items():
                for column in timestamp_columns:
                    rows = db.execute(f"SELECT id, {column} FROM {table}").fetchall()
                    for row in rows:
                        if not row[column]:
                            continue
                        parsed = datetime.fromisoformat(row[column])
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=UTC)
                        converted = parsed.astimezone(IST).isoformat()
                        if converted != row[column]:
                            db.execute(f"UPDATE {table} SET {column} = ? WHERE id = ?", (converted, row["id"]))
            # Existing installations predate dedicated analysis timing. Use
            # their durable lifecycle timestamps as the best truthful
            # fallback, while new runs record the exact Start analysis click.
            db.execute(
                "UPDATE runs SET analysis_started_at = created_at "
                "WHERE analysis_started_at IS NULL AND status != 'uploading'"
            )
            db.execute(
                "UPDATE runs SET completed_at = updated_at "
                "WHERE completed_at IS NULL AND status IN ('succeeded', 'failed', 'cancelled')"
            )
            db.commit()

    def create_run(self, mode: str) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        now = ist_now()
        with self._write_lock, self.connection() as db:
            if mode == "reuse_historical" and self.active_build(db) is None:
                raise ValueError("no successful historical version is available")
            db.execute(
                "INSERT INTO runs(id, mode, status, stage, created_at, updated_at) "
                "VALUES (?, ?, 'uploading', 'uploading', ?, ?)",
                (run_id, mode, now, now),
            )
            db.commit()
        self.staging_path(run_id).mkdir(parents=True, exist_ok=False)
        return self.get_run(run_id)

    def staging_path(self, run_id: str) -> Path:
        return self.root / "staging" / run_id

    def historical_path(self, version_id: str) -> Path:
        return self.root / "historical" / version_id

    def upcoming_path(self, version_id: str) -> Path:
        return self.root / "upcoming" / version_id

    def build_path(self, build_id: str) -> Path:
        return self.root / "builds" / build_id

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        result = dict(row)
        result["cancelRequested"] = bool(result.pop("cancel_requested"))
        return result

    def active_run(self) -> dict[str, Any] | None:
        """Return the newest run whose analysis worker is still active.

        Upload-only runs are excluded because a reloaded browser cannot restore
        its local File objects. Queued/processing runs need only their durable ID.
        """
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM runs WHERE status IN ('queued', 'processing') "
                "ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["cancelRequested"] = bool(result.pop("cancel_requested"))
        return result

    def incomplete_analysis_runs(self) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM runs WHERE status IN ('queued', 'processing') ORDER BY created_at"
            ).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["cancelRequested"] = bool(result.pop("cancel_requested"))
            results.append(result)
        return results

    def update_run(self, run_id: str, **changes: Any) -> None:
        allowed = {
            "status",
            "stage",
            "progress",
            "message",
            "error",
            "issues_path",
            "build_id",
            "cancel_requested",
            "processed_count",
            "total_count",
            "cache_hits",
            "eta_seconds",
            "historical_processed",
            "historical_total",
            "upcoming_processed",
            "upcoming_total",
            "analysis_started_at",
            "completed_at",
        }
        values = {key: value for key, value in changes.items() if key in allowed}
        values["updated_at"] = ist_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._write_lock, self.connection() as db:
            db.execute(f"UPDATE runs SET {assignments} WHERE id = ?", (*values.values(), run_id))
            db.commit()

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] in {"succeeded", "failed", "cancelled"}:
            return run
        if run["status"] == "uploading":
            # No worker exists yet to observe cancel_requested, so an upload-
            # stage cancellation must become terminal immediately.
            self.update_run(
                run_id,
                status="cancelled",
                stage="cancelled",
                cancel_requested=1,
                eta_seconds=0,
                completed_at=ist_now(),
                message="Run cancelled",
            )
        else:
            self.update_run(run_id, cancel_requested=1, message="Cancellation requested")
        return self.get_run(run_id)

    def active_build(self, db: sqlite3.Connection | None = None) -> dict[str, Any] | None:
        if db is None:
            with self.connection() as connection:
                return self.active_build(connection)
        row = db.execute(
            """
            SELECT b.*, h.created_at AS historical_created_at,
                   h.path AS historical_path, h.product_count AS historical_product_count,
                   h.image_coverage AS historical_image_coverage, h.model_version,
                   u.path AS upcoming_path, u.product_count AS upcoming_product_count,
                   u.image_coverage AS upcoming_image_coverage
            FROM active_build a
            JOIN builds b ON b.id = a.build_id
            JOIN historical_versions h ON h.id = b.historical_version_id
            JOIN upcoming_versions u ON u.id = b.upcoming_version_id
            WHERE a.singleton = 1
            """
        ).fetchone()
        return self._row(row)

    def build(self, build_id: str) -> dict[str, Any] | None:
        with self.connection() as db:
            row = db.execute(
                """
                SELECT b.*, h.path AS historical_path, u.path AS upcoming_path
                FROM builds b
                JOIN historical_versions h ON h.id = b.historical_version_id
                JOIN upcoming_versions u ON u.id = b.upcoming_version_id
                WHERE b.id = ?
                """,
                (build_id,),
            ).fetchone()
        return self._row(row)

    def activate(
        self,
        *,
        run_id: str,
        historical_version_id: str,
        historical_path: Path,
        historical_count: int,
        historical_coverage: int,
        model_version: str,
        upcoming_version_id: str,
        upcoming_path: Path,
        upcoming_count: int,
        upcoming_coverage: int,
        artifact_path: Path,
        replace_historical: bool,
    ) -> list[Path]:
        now = ist_now()
        cleanup: list[Path] = []
        with self._write_lock, self.connection() as db:
            old = self.active_build(db)
            db.execute("BEGIN IMMEDIATE")
            if replace_historical:
                db.execute(
                    "INSERT INTO historical_versions VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        historical_version_id,
                        now,
                        str(historical_path),
                        historical_count,
                        historical_coverage,
                        model_version,
                    ),
                )
            db.execute(
                "INSERT INTO upcoming_versions VALUES (?, ?, ?, ?, ?)",
                (upcoming_version_id, now, str(upcoming_path), upcoming_count, upcoming_coverage),
            )
            db.execute(
                "INSERT INTO builds VALUES (?, ?, ?, ?, ?)",
                (run_id, now, historical_version_id, upcoming_version_id, str(artifact_path)),
            )
            db.execute(
                "INSERT INTO active_build(singleton, build_id) VALUES (1, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET build_id = excluded.build_id",
                (run_id,),
            )
            db.execute(
                "UPDATE runs SET status='succeeded', stage='succeeded', progress=1, "
                "message='Build activated', build_id=?, eta_seconds=0, "
                "processed_count=total_count, completed_at=?, updated_at=? WHERE id=?",
                (run_id, now, now, run_id),
            )
            if old is not None:
                db.execute("DELETE FROM builds WHERE id = ?", (old["id"],))
                db.execute("DELETE FROM upcoming_versions WHERE id = ?", (old["upcoming_version_id"],))
                cleanup.extend([Path(old["upcoming_path"]), Path(old["artifact_path"]).parent])
                if replace_historical and old["historical_version_id"] != historical_version_id:
                    db.execute("DELETE FROM historical_versions WHERE id = ?", (old["historical_version_id"],))
                    cleanup.append(Path(old["historical_path"]))
            db.commit()
        return cleanup

    def artifact(self, build_id: str) -> dict[str, Any]:
        build = self.build(build_id)
        if build is None:
            raise KeyError(build_id)
        return json.loads(Path(build["artifact_path"]).read_text(encoding="utf-8"))

    def cleanup_paths(self, paths: list[Path]) -> None:
        for path in paths:
            if path.exists() and self.root in path.resolve().parents:
                shutil.rmtree(path, ignore_errors=True)

    def purge_superseded_run_history(self) -> list[Path]:
        """Delete terminal run audit rows except the currently active build.

        A reused historical version may have an ID matching an older run. Only
        the run audit row and its staging directory are removed here; the
        historical_versions row and durable catalogue path remain untouched.
        Non-terminal work is also preserved so concurrent uploads/runs survive.
        """
        with self._write_lock, self.connection() as db:
            active = db.execute("SELECT build_id FROM active_build WHERE singleton = 1").fetchone()
            if active is None:
                return []
            active_run_id = str(active["build_id"])
            retired = db.execute(
                "SELECT id FROM runs WHERE id != ? AND status IN ('succeeded', 'failed', 'cancelled')",
                (active_run_id,),
            ).fetchall()
            retired_ids = [str(row["id"]) for row in retired]
            if retired_ids:
                placeholders = ",".join("?" for _ in retired_ids)
                db.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", retired_ids)
                db.commit()
        return [self.staging_path(run_id) for run_id in retired_ids]

    def cleanup_failed_staging(self) -> None:
        cutoff = datetime.now(IST) - timedelta(hours=24)
        with self.connection() as db:
            rows = db.execute("SELECT id, updated_at FROM runs WHERE status IN ('failed', 'cancelled')").fetchall()
        for row in rows:
            if datetime.fromisoformat(row["updated_at"]) < cutoff:
                shutil.rmtree(self.staging_path(row["id"]), ignore_errors=True)
