from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import replace
from pathlib import Path

from fashion_matching.config import MatchingSettings
from fashion_matching.manifests import read_manifest, records_from_directory
from fashion_matching.matching import FashionMatcher
from fashion_matching.models import ManifestRecord, MatchResult
from fashion_matching.runtime import (
    build_encoder,
    build_preprocessor,
    build_store,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match upcoming fashion images against the active catalogue.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query-image", type=Path)
    source.add_argument("--query-directory", type=Path)
    source.add_argument("--query-manifest", type=Path)
    parser.add_argument("--query-product-id")
    parser.add_argument("--query-image-id")
    parser.add_argument("--query-text")
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--device")
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--minimum-score", type=float)
    parser.add_argument("--collection")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument(
        "--failure-report",
        type=Path,
        default=Path("outputs/query-failures.json"),
    )
    return parser


def _records(args: argparse.Namespace) -> list[ManifestRecord]:
    if args.query_manifest:
        return read_manifest(args.query_manifest)
    if args.query_directory:
        return records_from_directory(args.query_directory)
    path = args.query_image.expanduser().resolve()
    return [
        ManifestRecord(
            product_id=args.query_product_id or path.stem,
            image_id=args.query_image_id or path.stem,
            image_path=path,
            text=args.query_text,
        )
    ]


def _write_csv(path: Path, results: list[MatchResult]) -> None:
    fields = (
        "query_product_id",
        "query_image_id",
        "no_suitable_match",
        "error",
        "rank",
        "product_id",
        "image_id",
        "view",
        "final_score",
        "image_score",
        "text_score",
        "attribute_score",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            if not result.matches:
                writer.writerow(
                    {
                        "query_product_id": result.query_product_id,
                        "query_image_id": result.query_image_id,
                        "no_suitable_match": result.no_suitable_match,
                        "error": result.error,
                    }
                )
                continue
            for match in result.matches:
                writer.writerow(
                    {
                        "query_product_id": result.query_product_id,
                        "query_image_id": result.query_image_id,
                        "no_suitable_match": result.no_suitable_match,
                        "error": result.error,
                        "rank": match.rank,
                        "product_id": match.product_id,
                        "image_id": match.image_id,
                        "view": match.view,
                        "final_score": match.final_score,
                        "image_score": match.image_score,
                        "text_score": match.text_score,
                        "attribute_score": match.attribute_score,
                    }
                )


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
            "candidate_count": args.candidate_count,
            "top_k": args.top_k,
            "minimum_score": args.minimum_score,
        }.items()
        if value is not None
    }
    settings = replace(settings, **overrides)
    encoder = build_encoder(settings)
    matcher = FashionMatcher(
        encoder=encoder,
        preprocessor=build_preprocessor(settings),
        store=build_store(settings),
        collection=args.collection or settings.collection_alias,
        weights=settings.weights,
        candidate_count=settings.candidate_count,
        top_k=settings.top_k,
        minimum_score=settings.minimum_score,
    )
    results = [matcher.match(record) for record in _records(args)]
    payload = {
        "model": {
            "id": encoder.model_id,
            "revision": encoder.revision,
            "dimension": encoder.dimension,
            "preprocessing_version": matcher.preprocessor.version,
            "model_version": matcher.model_version,
        },
        "summary": {
            "total_queries": len(results),
            "successful_queries": sum(result.error is None for result in results),
            "failed_queries": sum(result.error is not None for result in results),
            "no_suitable_match": sum(result.no_suitable_match for result in results),
        },
        "results": [result.to_dict() for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    _write_csv(csv_output, results)
    failures = [result.to_dict() for result in results if result.error is not None]
    args.failure_report.parent.mkdir(parents=True, exist_ok=True)
    args.failure_report.write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
