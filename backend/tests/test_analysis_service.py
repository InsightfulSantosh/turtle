from __future__ import annotations

import asyncio
import csv
from datetime import datetime
from pathlib import Path

import pytest
from analysis_service import api as api_module
from analysis_service.api import SAFE_FILENAME
from analysis_service.catalogs import (
    HISTORICAL_FIELDS,
    SUPPORTED_CATALOGUE_SUFFIXES,
    UPCOMING_FIELDS,
    validate_catalog,
)
from analysis_service.processor import AnalysisProcessor
from analysis_service.store import RunStore
from openpyxl import Workbook
from PIL import Image
from starlette.requests import ClientDisconnect


def write_csv(path: Path, fields: tuple[str, ...], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "navy").save(path)


def write_xlsx(path: Path, fields: tuple[str, ...], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    worksheet.append(list(fields))
    worksheet.append([row[field] for field in fields])
    workbook.save(path)


def historical_row(product_id: str = "OTSH-100-1001") -> dict[str, object]:
    return {
        "product_id": product_id,
        "season": "AW25",
        "item_type": "OTSH",
        "style_code": "100",
        "colour_code": "1001",
        "design": "PLAINS",
        "category_type": "FORMAL",
        "fabric": "100% cotton",
        "colour": "NAVY",
        "total_order_quantity": 500,
        "dispatch_quantity": 500,
        "sales_quantity": 350,
        "sell_through": 0.7,
        "ageing_days": 140,
        "weekly_sell_through": 0.035,
    }


def upcoming_row(product_id: str = "OTSH-200-1002") -> dict[str, object]:
    return {
        "product_id": product_id,
        "season": "SS27",
        "item_type": "OTSH",
        "style_code": "200",
        "colour_code": "1002",
        "design": "PLAINS",
        "category_type": "FORMAL",
        "fabric": "cotton stretch",
        "colour": "BLUE",
        "collection_world": "PROFESSIONAL",
    }


def populate_run(store: RunStore, run_id: str, *, historical_id: str | None, upcoming_id: str) -> None:
    staging = store.staging_path(run_id)
    if historical_id is not None:
        write_csv(staging / "historical" / "csv" / "historical.csv", HISTORICAL_FIELDS, historical_row(historical_id))
        write_image(staging / "historical" / "images" / f"{historical_id}.jpg")
    write_csv(staging / "upcoming" / "csv" / "upcoming.csv", UPCOMING_FIELDS, upcoming_row(upcoming_id))
    write_image(staging / "upcoming" / "images" / f"{upcoming_id}.jpg")


def test_canonical_upload_validation_reports_missing_images(tmp_path: Path) -> None:
    csv_path = tmp_path / "upcoming.csv"
    write_csv(csv_path, UPCOMING_FIELDS, upcoming_row())

    validated = validate_catalog(csv_path, tmp_path / "images", "upcoming")

    assert validated.rows[0]["product_id"] == "OTSH-200-1002"
    assert validated.rows[0]["category_type"] == "FORMAL"
    assert [issue.code for issue in validated.issues] == ["missing_image"]
    assert validated.issues[0].severity == "warning"
    assert [record.code for record in validated.report_records] == [
        "image_coverage_summary",
        "missing_image",
    ]
    assert "0 of 1 catalogue products matched" in validated.report_records[0].message


def test_xlsx_catalogue_uses_the_same_canonical_validation(tmp_path: Path) -> None:
    workbook_path = tmp_path / "upcoming.xlsx"
    write_xlsx(workbook_path, UPCOMING_FIELDS, upcoming_row())
    write_image(tmp_path / "images" / "OTSH-200-1002.jpg")

    validated = validate_catalog(workbook_path, tmp_path / "images", "upcoming")

    assert validated.rows[0]["product_id"] == "OTSH-200-1002"
    assert validated.image_index["OTSH-200-1002"].suffix == ".jpg"
    assert validated.issues == []
    assert [record.code for record in validated.report_records] == [
        "image_coverage_summary",
        "matched_image",
    ]
    assert validated.report_records[1].severity == "passed"


def test_supported_catalogue_formats_include_legacy_and_open_formats() -> None:
    assert {
        ".csv",
        ".xlsx",
        ".xlsm",
        ".xls",
        ".xlsb",
        ".ods",
    } == SUPPORTED_CATALOGUE_SUFFIXES


def test_upload_filename_accepts_business_workbook_punctuation() -> None:
    assert SAFE_FILENAME.fullmatch("LAST SEASONES ORDERING & SALE THRU DATA.xlsb")
    assert not SAFE_FILENAME.fullmatch("../historical.xlsb")
    assert not SAFE_FILENAME.fullmatch("historical/other.xlsb")


class DisconnectingRequest:
    """Delivers part of a body, then drops the connection like a real client."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def stream(self):
        for chunk in self._chunks:
            yield chunk
        raise ClientDisconnect()


def test_client_disconnect_mid_upload_is_resumable_not_a_server_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A browser or proxy dropping a large body is not a server fault. Reporting
    it as a 500 made the client abandon an otherwise healthy multi-hundred-
    megabyte run; the partial bytes must survive so the next HEAD resumes."""

    store = RunStore(tmp_path / "analysis")
    monkeypatch.setattr(api_module, "store", store)
    run = store.create_run("full_replace")

    response = asyncio.run(
        api_module.upload_file(
            run["id"],
            "upcoming",
            "images",
            "OTSH-200-1002.jpg",
            DisconnectingRequest([b"a" * 100, b"b" * 50]),  # type: ignore[arg-type]
            upload_offset=0,
            upload_length=500,
        )
    )

    assert response.status_code == 499
    assert response.headers["Upload-Offset"] == "150"
    target = store.staging_path(run["id"]) / "upcoming" / "images" / "OTSH-200-1002.jpg"
    assert target.stat().st_size == 150

    # The run stays open, so the client can resume from the reported offset.
    assert store.get_run(run["id"])["status"] == "uploading"
    resumed = api_module.upload_offset(run["id"], "upcoming", "images", "OTSH-200-1002.jpg")
    assert resumed.headers["Upload-Offset"] == "150"


def test_cancelling_an_uploading_run_is_immediately_terminal(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "analysis")
    created = store.create_run("full_replace")

    cancelled = store.cancel_run(created["id"])

    assert cancelled["status"] == "cancelled"
    assert cancelled["stage"] == "cancelled"
    assert cancelled["cancelRequested"] is True


def test_active_run_finds_the_latest_analysis_but_not_an_upload(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "analysis")
    first = store.create_run("full_replace")
    store.update_run(first["id"], status="processing", stage="validating")

    second = store.create_run("full_replace")
    assert store.active_run()["id"] == first["id"]
    assert [run["id"] for run in store.incomplete_analysis_runs()] == [first["id"]]

    store.update_run(second["id"], status="queued", stage="queued")
    assert store.active_run()["id"] == second["id"]
    assert [run["id"] for run in store.incomplete_analysis_runs()] == [first["id"], second["id"]]

    store.update_run(second["id"], status="failed", stage="failed")
    assert store.active_run()["id"] == first["id"]


def test_full_replace_then_reuse_preserves_only_the_reused_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunStore(tmp_path / "analysis")
    processor = AnalysisProcessor(store, vision_enabled=False, max_workers=1)

    first = store.create_run("full_replace")
    populate_run(store, first["id"], historical_id="OTSH-100-1001", upcoming_id="OTSH-200-1002")
    processor.process(first["id"])
    first_active = store.active_build()
    assert first_active is not None
    first_historical_created_at = first_active["historical_created_at"]
    first_history = Path(first_active["historical_path"])
    first_upcoming = Path(first_active["upcoming_path"])
    first_run = store.get_run(first["id"])
    assert first_run["status"] == "succeeded"
    assert first_run["analysis_started_at"] is not None
    assert first_run["completed_at"] is not None
    assert datetime.fromisoformat(first_run["completed_at"]) >= datetime.fromisoformat(first_run["analysis_started_at"])
    assert Path(first_run["issues_path"]).is_file()
    with Path(first_run["issues_path"]).open(encoding="utf-8", newline="") as handle:
        report_rows = list(csv.DictReader(handle))
    assert list(report_rows[0]) == [
        "recordType", "catalog", "row", "productId", "field", "code", "message", "severity"
    ]
    assert [row["code"] for row in report_rows[:2]] == [
        "image_coverage_summary",
        "image_coverage_summary",
    ]
    assert {row["code"] for row in report_rows} == {"image_coverage_summary", "matched_image"}

    second = store.create_run("reuse_historical")
    populate_run(store, second["id"], historical_id=None, upcoming_id="OTSH-201-1003")
    processor.process(second["id"])
    second_active = store.active_build()
    assert second_active is not None
    assert second_active["historical_version_id"] == first_active["historical_version_id"]
    assert second_active["historical_created_at"] == first_historical_created_at
    assert second_active["created_at"] != first_historical_created_at
    monkeypatch.setattr(api_module, "store", store)
    assert api_module.active_historical()["createdAt"] == first_historical_created_at
    assert first_history.is_dir()
    assert not first_upcoming.exists()
    assert store.get_run(second["id"])["status"] == "succeeded"
    with pytest.raises(KeyError):
        store.get_run(first["id"])

    third = store.create_run("full_replace")
    populate_run(store, third["id"], historical_id="OTSH-110-1004", upcoming_id="OTSH-210-1005")
    processor.process(third["id"])
    third_active = store.active_build()
    assert third_active is not None
    assert third_active["historical_version_id"] == third["id"]
    assert not first_history.exists()
    assert not Path(second_active["upcoming_path"]).exists()
    assert store.get_run(third["id"])["status"] == "succeeded"
    with pytest.raises(KeyError):
        store.get_run(second["id"])


def test_purge_superseded_history_preserves_active_dependencies_and_open_work(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "analysis")
    processor = AnalysisProcessor(store, vision_enabled=False, max_workers=1)
    active = store.create_run("full_replace")
    populate_run(store, active["id"], historical_id="OTSH-100-1001", upcoming_id="OTSH-200-1002")
    processor.process(active["id"])

    cancelled = store.create_run("full_replace")
    cancelled_staging = store.staging_path(cancelled["id"])
    store.cancel_run(cancelled["id"])
    uploading = store.create_run("full_replace")

    cleanup = store.purge_superseded_run_history()
    store.cleanup_paths(cleanup)

    assert store.get_run(active["id"])["status"] == "succeeded"
    assert store.get_run(uploading["id"])["status"] == "uploading"
    with pytest.raises(KeyError):
        store.get_run(cancelled["id"])
    assert not cancelled_staging.exists()
    current = store.active_build()
    assert current is not None
    assert Path(current["historical_path"]).is_dir()
    assert Path(current["upcoming_path"]).is_dir()
    assert Path(current["artifact_path"]).is_file()


def test_all_sqlite_timestamps_use_ist_and_legacy_utc_is_migrated(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    store = RunStore(root)
    processor = AnalysisProcessor(store, vision_enabled=False, max_workers=1)
    run = store.create_run("full_replace")
    populate_run(store, run["id"], historical_id="OTSH-100-1001", upcoming_id="OTSH-200-1002")
    processor.process(run["id"])

    legacy_utc = "2026-01-01T00:00:00+00:00"
    with store.connection() as db:
        db.execute("UPDATE historical_versions SET created_at = ?", (legacy_utc,))
        db.execute("UPDATE upcoming_versions SET created_at = ?", (legacy_utc,))
        db.execute("UPDATE builds SET created_at = ?", (legacy_utc,))
        db.execute(
            "UPDATE runs SET created_at = ?, updated_at = ?, analysis_started_at = ?, completed_at = ?",
            (legacy_utc, legacy_utc, legacy_utc, legacy_utc),
        )
        db.commit()

    migrated = RunStore(root)
    expected_ist = "2026-01-01T05:30:00+05:30"
    with migrated.connection() as db:
        assert db.execute("SELECT created_at FROM historical_versions").fetchone()[0] == expected_ist
        assert db.execute("SELECT created_at FROM upcoming_versions").fetchone()[0] == expected_ist
        assert db.execute("SELECT created_at FROM builds").fetchone()[0] == expected_ist
        run_row = db.execute(
            "SELECT created_at, updated_at, analysis_started_at, completed_at FROM runs"
        ).fetchone()
        assert tuple(run_row) == (expected_ist, expected_ist, expected_ist, expected_ist)


def test_activated_build_publishes_the_planner_artifact_contract(tmp_path: Path) -> None:
    """The frontend repools the forecast from these fields whenever a planner
    moves a slider, so an activated build has to carry the fitted priors and
    ceilings — not just the recommendations they produced."""

    store = RunStore(tmp_path / "analysis")
    processor = AnalysisProcessor(store, vision_enabled=False, max_workers=1)
    run = store.create_run("full_replace")
    populate_run(store, run["id"], historical_id="OTSH-100-1001", upcoming_id="OTSH-200-1002")
    processor.process(run["id"])

    active = store.active_build()
    assert active is not None
    artifact = store.artifact(active["id"])
    model = artifact["meta"]["model"]

    assert model["version"] == "5.1.0"
    assert model["evidencePolicy"] == "pooled_visual_analogue_forecast"
    assert model["noMachineLearningForecast"] is False
    assert model["noAttributeMatching"] is True
    assert model["visualOnlyRanking"] is True
    assert model["topK"] == 4
    assert model["targetSellThrough"] == 0.70
    assert model["minimumVisualScore"] == 0.5
    assert "regressionBlend" not in model
    assert "attributeWeight" not in model

    demand_model = model["demandModel"]
    assert demand_model["horizonWeeks"] > 0
    assert demand_model["shrinkageTau"] > 0
    assert demand_model["wideUncertaintyEffectiveN"] > 0
    assert demand_model["wideUncertaintySkewRatio"] > 1
    # A global prior must always exist, whatever the uploaded catalogue covers.
    assert demand_model["groups"][""]["rows"] > 0
    assert model["buyCeilings"]["globalCeiling"] > 0

    # Images are served by the analysis service for this build, never from a
    # developer's local DATA/ directory.
    for item in artifact["upcoming"]:
        if item["imageUrl"] is not None:
            assert item["imageUrl"].startswith(f"/api/builds/{active['id']}/images/upcoming/")
    for item in artifact["upcoming"]:
        recommendation = item["recommendation"]
        assert len(item["matches"]) <= 4
        assert all("attributeScore" not in match for match in item["matches"])
        if recommendation["noSuitableMatch"]:
            assert recommendation["quantity"] == 0
            assert recommendation["expectedSales"] == 0
            assert "demand" not in recommendation


def test_failed_replacement_never_changes_the_active_build(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "analysis")
    processor = AnalysisProcessor(store, vision_enabled=False, max_workers=1)
    first = store.create_run("full_replace")
    populate_run(store, first["id"], historical_id="OTSH-100-1001", upcoming_id="OTSH-200-1002")
    processor.process(first["id"])
    active_id = store.active_build()["id"]  # type: ignore[index]

    failed = store.create_run("full_replace")
    # Deliberately omit all required uploads.
    processor.process(failed["id"])

    assert store.get_run(failed["id"])["status"] == "failed"
    assert store.active_build()["id"] == active_id  # type: ignore[index]
