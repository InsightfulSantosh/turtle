from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from fashion_matching.config import MatchingSettings
from fashion_matching.indexing import CatalogueIndexer
from fashion_matching.manifests import read_manifest
from fashion_matching.runtime import (
    build_encoder,
    build_preprocessor,
    build_store,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Idempotently index historical fashion catalogue images.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Activate the completed version through the configured Qdrant alias.",
    )
    parser.add_argument(
        "--activate-with-failures",
        action="store_true",
        help="Allow alias activation even when some manifest records failed.",
    )
    parser.add_argument(
        "--failure-report",
        type=Path,
        default=Path("outputs/index-failures.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = MatchingSettings.from_environment()
    overrides = {
        name: value
        for name, value in {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "device": args.device,
            "batch_size": args.batch_size,
        }.items()
        if value is not None
    }
    settings = replace(settings, **overrides)
    records = read_manifest(args.manifest)
    encoder = build_encoder(settings)
    indexer = CatalogueIndexer(
        encoder=encoder,
        preprocessor=build_preprocessor(settings),
        store=build_store(settings),
        collection_prefix=settings.collection_prefix,
        batch_size=settings.batch_size,
    )
    summary = indexer.index(
        records,
        activate_alias=settings.collection_alias if args.activate else None,
        activate_with_failures=args.activate_with_failures,
    )
    report = {
        "collection": indexer.collection,
        "active_alias": settings.collection_alias if args.activate else None,
        "model_id": encoder.model_id,
        "model_revision": encoder.revision,
        "embedding_dimension": encoder.dimension,
        "preprocessing_version": indexer.preprocessor.version,
        **summary.to_dict(),
    }
    args.failure_report.parent.mkdir(parents=True, exist_ok=True)
    args.failure_report.write_text(
        json.dumps(summary.failures, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if summary.failed_images:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
