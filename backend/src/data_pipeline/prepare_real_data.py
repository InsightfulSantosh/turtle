"""Command-line entry point for the real-data ML pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.pipeline import RealDataPipeline
from data_pipeline.settings import PipelineSettings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest, validate and preprocess the real workbooks, then build "
            "the frontend ML artifact"
        )
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    settings = PipelineSettings.from_project(args.output)
    summary = RealDataPipeline(settings).run()
    print(
        f"Wrote {summary.output_path} with "
        f"{summary.historical_items} historical and "
        f"{summary.upcoming_items} upcoming products; "
        f"model v{summary.model_version}; "
        f"{summary.selection_method}; "
        f"WAPE={summary.backtest_wape:.1%}"
    )


if __name__ == "__main__":
    main()
