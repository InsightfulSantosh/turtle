"""Configuration for the real-data ingestion and model-artifact pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.config import paths


@dataclass(frozen=True)
class PipelineSettings:
    data_root: Path
    temporary_root: Path
    output_path: Path
    historical_filename: str = "LAST SEASONES ORDERING & SALE THRU DATA.xlsb"
    upcoming_filename: str = "SEG WISE SS27 MASTER SHEET TILL.xlsb"
    sheet_name: str = "Sheet1"
    upcoming_season: str = "SS27"

    @classmethod
    def from_project(cls, output_path: Path | None = None) -> "PipelineSettings":
        return cls(
            data_root=paths.data,
            temporary_root=paths.temporary,
            output_path=(output_path or paths.model_artifact).resolve(),
        )

    @property
    def historical_source(self) -> Path:
        return self.raw_root / self.historical_filename

    @property
    def upcoming_source(self) -> Path:
        return self.raw_root / self.upcoming_filename

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def historical_processed_output(self) -> Path:
        return self.processed_root / "historical_cleaned.csv"

    @property
    def upcoming_processed_output(self) -> Path:
        return self.processed_root / "upcoming_cleaned.csv"

    @property
    def validation_output(self) -> Path:
        return self.processed_root / "data_cleaning_validation.csv"

    @property
    def converted_root(self) -> Path:
        return self.temporary_root / "real-data-converted"

    @property
    def libreoffice_profile(self) -> Path:
        return self.temporary_root / "lo-profile-real-data"
