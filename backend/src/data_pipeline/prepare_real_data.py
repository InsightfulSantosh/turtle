"""Command-line entry point for the real-data ML pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.pipeline import RealDataPipeline
from data_pipeline.settings import PipelineSettings
from fashion_matching.artifact_vision import build_artifact_vision_output
from fashion_matching.config import MatchingSettings
from fashion_matching.encoders import create_dino_encoder, create_encoder
from fashion_matching.preprocessing import ImagePreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Ingest, validate and preprocess the real workbooks, then build the frontend ML artifact")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--with-vision",
        action="store_true",
        help="Embed mapped product images and include visual similarity",
    )
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    settings = PipelineSettings.from_project(args.output)
    matching = MatchingSettings.from_environment()
    vision_builder = None
    if args.with_vision:
        model_id = args.model_id or matching.model_id
        revision = args.model_revision or matching.model_revision
        device = args.device or matching.device
        batch_size = args.batch_size or matching.batch_size
        encoder = create_encoder(
            model_id,
            revision=revision,
            device=device,
        )
        reranker = (
            create_dino_encoder(
                matching.dino_model_id,
                revision=matching.dino_model_revision,
                device=device,
            )
            if matching.dino_reranker_enabled
            else None
        )
        preprocessor = ImagePreprocessor(
            version=matching.preprocessing_version,
            max_bytes=matching.max_image_bytes,
            max_pixels=matching.max_image_pixels,
            minimum_dimension=matching.min_image_dimension,
            pad_to_square=matching.pad_to_square,
            crop_uniform_background=matching.crop_uniform_background,
        )

        def vision_builder(source):
            return build_artifact_vision_output(
                source,
                settings=settings,
                encoder=encoder,
                preprocessor=preprocessor,
                reranker=reranker,
                candidate_count=matching.dino_candidate_count,
                reranker_weight_grid=matching.dino_weight_grid,
                require_same_item_type=matching.dino_require_same_item_type,
                batch_size=batch_size,
            )

    summary = RealDataPipeline(settings).run(vision_builder)
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
