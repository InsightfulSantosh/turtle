"""Command-line entry point for the real-data visual recommendation pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_pipeline.pipeline import RealDataPipeline
from data_pipeline.settings import PipelineSettings
from fashion_matching.artifact_vision import build_artifact_vision_output
from fashion_matching.config import MatchingSettings
from fashion_matching.encoders import create_dino_encoder, create_encoder
from fashion_matching.preprocessing import ImagePreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Ingest, validate and preprocess the real workbooks, then build the visual recommendation artifact")
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
        "--verbose",
        action="store_true",
        help="Show model loading and per-batch visual encoding progress",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Also write pipeline logs to this file for live terminal monitoring",
    )
    parser.add_argument(
        "--item-type",
        help="Build a scoped artifact containing only one canonical item type, such as OTSH",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help=(
            "Build a deterministic upcoming-product preview, balanced across "
            "every item type and category type"
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=27,
        help="Stable seed used to select products inside each preview stratum (default: 27)",
    )
    args = parser.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

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
                visual_only_ranking=matching.visual_only_ranking,
                pattern_gate_enabled=matching.pattern_gate_enabled,
                pattern_max_distance=matching.pattern_max_distance,
                colour_gate_enabled=matching.colour_gate_enabled,
                colour_max_distance=matching.colour_max_distance,
                appearance_weights={
                    "neural": matching.appearance_neural_weight,
                    "colour": matching.appearance_colour_weight,
                    "texture": matching.appearance_texture_weight,
                },
                batch_size=batch_size,
            )

    summary = RealDataPipeline(settings).run(
        vision_builder,
        item_type=args.item_type,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    print(
        f"Wrote {summary.output_path} with "
        f"{summary.historical_items} historical and "
        f"{summary.upcoming_items} upcoming products; "
        f"model v{summary.model_version}; "
        f"{summary.evidence_policy}; "
        f"accepted visual matches={summary.accepted_matches}"
    )


if __name__ == "__main__":
    main()
