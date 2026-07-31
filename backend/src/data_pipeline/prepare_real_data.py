"""Command-line entry point for the real-data ML pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.pipeline import RealDataPipeline
from data_pipeline.settings import PipelineSettings
from fashion_matching.artifact_vision import build_artifact_vision_output
from fashion_matching.config import MatchingSettings
from fashion_matching.encoders import create_dino_encoder, create_encoder
from fashion_matching.garment_segmentation import GroundedSam2GarmentSegmenter
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
    parser.add_argument(
        "--item-type",
        help="Build a scoped artifact containing only one canonical item type, such as OTSH",
    )
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
        garment_segmenter = (
            GroundedSam2GarmentSegmenter(
                detector_model_id=matching.garment_detector_model_id,
                detector_revision=matching.garment_detector_revision,
                sam2_model_id=matching.garment_sam2_model_id,
                sam2_revision=matching.garment_sam2_revision,
                configured_device=device,
                detector_threshold=matching.garment_detector_threshold,
                text_threshold=matching.garment_text_threshold,
                minimum_coverage=matching.garment_minimum_coverage,
                maximum_coverage=matching.garment_maximum_coverage,
            )
            if matching.appearance_mask_enabled and matching.garment_segmentation_enabled
            else None
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
                require_same_design=matching.dino_require_same_design,
                require_same_colour_family=matching.dino_require_same_colour_family,
                pattern_gate_enabled=matching.pattern_gate_enabled,
                pattern_max_distance=matching.pattern_max_distance,
                appearance_mask_enabled=matching.appearance_mask_enabled,
                garment_segmenter=garment_segmenter,
                allow_border_mask_fallback=matching.garment_segmentation_border_fallback,
                appearance_weights={
                    "neural": matching.appearance_neural_weight,
                    "colour": matching.appearance_colour_weight,
                    "texture": matching.appearance_texture_weight,
                },
                batch_size=batch_size,
            )

    summary = RealDataPipeline(settings).run(vision_builder, item_type=args.item_type)
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
