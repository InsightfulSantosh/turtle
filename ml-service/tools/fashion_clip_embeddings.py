from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
DEFAULT_SOURCE = ROOT / "app" / "generated-data.json"
DEFAULT_IMAGE_MAP = PROJECT_ROOT / "tmp" / "vision-images-map.json"


def load_image_paths(path: Path) -> dict[tuple[str, str], Path]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(str(row["group"]), str(row["key"])): Path(row["path"]) for row in rows}


def available_items(
    items: list[dict[str, Any]],
    group: str,
    image_paths: dict[tuple[str, str], Path],
) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    for item in items:
        image_path = image_paths.get((group, str(item["id"])))
        if image_path and image_path.exists():
            output.append((str(item["id"]), image_path))
    return output


@torch.inference_mode()
def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    items: list[tuple[str, Path]],
    device: torch.device,
    batch_size: int,
) -> tuple[list[str], np.ndarray]:
    identifiers: list[str] = []
    vectors: list[np.ndarray] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        images: list[Image.Image] = []
        for _, path in batch:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        features = model.get_image_features(pixel_values=pixel_values)
        features = torch.nn.functional.normalize(features, dim=-1)
        vectors.append(features.cpu().float().numpy())
        identifiers.extend(identifier for identifier, _ in batch)
    if not vectors:
        return identifiers, np.empty((0, model.config.projection_dim), dtype=np.float32)
    return identifiers, np.concatenate(vectors, axis=0)


def append_distances(
    rows: list[dict[str, float | str]],
    left_ids: list[str],
    left_vectors: np.ndarray,
    right_ids: list[str],
    right_vectors: np.ndarray,
) -> None:
    similarities = np.clip(left_vectors @ right_vectors.T, -1.0, 1.0)
    for left_index, left_id in enumerate(left_ids):
        for right_index, right_id in enumerate(right_ids):
            rows.append({
                "leftId": left_id,
                "rightId": right_id,
                "distance": round(float(1.0 - similarities[left_index, right_index]), 7),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FashionCLIP cosine distances for the POC catalogue")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--image-map", type=Path, default=DEFAULT_IMAGE_MAP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="patrickjohncyh/fashion-clip")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    image_paths = load_image_paths(args.image_map)
    historical = available_items(source["historical"], "past", image_paths)
    upcoming = available_items(source["upcoming"], "upcoming", image_paths)
    if not historical:
        raise RuntimeError("no historical images are available for FashionCLIP")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Keep preprocessing pinned to the processor configuration shipped with the
    # checkpoint. Explicitly selecting the slow processor avoids a future
    # transformers default change from silently altering stored similarities.
    processor = CLIPProcessor.from_pretrained(args.model, use_fast=False)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    historical_ids, historical_vectors = encode_images(
        model, processor, historical, device, max(args.batch_size, 1)
    )
    upcoming_ids, upcoming_vectors = encode_images(
        model, processor, upcoming, device, max(args.batch_size, 1)
    )

    distances: list[dict[str, float | str]] = []
    append_distances(
        distances,
        historical_ids,
        historical_vectors,
        historical_ids,
        historical_vectors,
    )
    append_distances(
        distances,
        upcoming_ids,
        upcoming_vectors,
        historical_ids,
        historical_vectors,
    )
    revision = getattr(model.config, "_commit_hash", None)
    output = {
        "engine": "FashionCLIP 2.0 image embeddings (512D cosine distance)",
        "modelId": args.model,
        "modelRevision": revision,
        "embeddingDimension": int(historical_vectors.shape[1]),
        "device": device.type,
        "historicalCoverage": len(historical_ids),
        "upcomingCoverage": len(upcoming_ids),
        "calibrationMethod": "Robust logistic calibration of FashionCLIP cosine distance",
        "distances": distances,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    temporary.replace(args.output)
    print(
        f"FashionCLIP embedded {len(historical_ids)} historical and {len(upcoming_ids)} upcoming images "
        f"on {device.type}; wrote {len(distances)} distances"
    )


if __name__ == "__main__":
    main()
