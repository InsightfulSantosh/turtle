from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fashion_matching.models import ManifestRecord


class ManifestError(ValueError):
    """Raised when a catalogue or query manifest is invalid."""


def _read_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), start=2):
                yield line_number, dict(row)
        return
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(value, dict):
                    raise ManifestError(f"{path}:{line_number}: each JSONL row must be an object")
                yield line_number, value
        return
    raise ManifestError("manifest must be a CSV, JSONL, or NDJSON file")


def read_manifest(path: Path) -> list[ManifestRecord]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ManifestError(f"manifest does not exist: {resolved}")
    records: list[ManifestRecord] = []
    seen_image_ids: set[str] = set()
    for line_number, row in _read_rows(resolved):
        try:
            record = ManifestRecord.from_mapping(
                row,
                base_directory=resolved.parent,
            )
        except ValueError as exc:
            raise ManifestError(f"{resolved}:{line_number}: {exc}") from exc
        if record.image_id in seen_image_ids:
            raise ManifestError(f"{resolved}:{line_number}: duplicate image_id {record.image_id!r}")
        seen_image_ids.add(record.image_id)
        records.append(record)
    if not records:
        raise ManifestError(f"manifest has no records: {resolved}")
    return records


def records_from_directory(path: Path) -> list[ManifestRecord]:
    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise ManifestError(f"query directory does not exist: {directory}")
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    records = []
    for image_path in sorted(directory.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in extensions:
            records.append(
                ManifestRecord(
                    product_id=image_path.stem,
                    image_id=image_path.stem,
                    image_path=image_path,
                )
            )
    if not records:
        raise ManifestError(f"query directory has no supported images: {directory}")
    return records
