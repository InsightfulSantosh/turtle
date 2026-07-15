from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, CLIPModel


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.rows = frame.to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        upcoming = Image.open(row["upcoming_image_path"]).convert("RGB")
        historical = Image.open(row["historical_image_path"]).convert("RGB")
        label = 1.0 if float(row["relevance"]) >= 2 else -1.0
        return upcoming, historical, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune FashionCLIP image similarity on planner-reviewed pairs.")
    parser.add_argument("pairs", type=Path, help="CSV with image paths, relevance and season")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-model", default="patrickjohncyh/fashion-clip")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    frame = pd.read_csv(args.pairs)
    required = {"upcoming_image_path", "historical_image_path", "relevance", "season"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    seasons = sorted(frame["season"].astype(str).unique())
    if len(seasons) < 3:
        raise ValueError("at least three seasons are required for a temporal fine-tuning holdout")
    holdout = seasons[-1]
    train = frame[frame["season"].astype(str) != holdout]
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = CLIPModel.from_pretrained(args.base_model)
    model.text_model.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    def collate(batch):
        left, right, labels = zip(*batch, strict=True)
        left_inputs = processor(images=list(left), return_tensors="pt")
        right_inputs = processor(images=list(right), return_tensors="pt")
        return left_inputs, right_inputs, torch.tensor(labels, dtype=torch.float32)

    loader = DataLoader(PairDataset(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-6, weight_decay=0.02
    )
    loss_function = torch.nn.CosineEmbeddingLoss(margin=0.25)
    losses: list[float] = []
    for _epoch in range(args.epochs):
        for left_inputs, right_inputs, labels in loader:
            left_inputs = {key: value.to(device) for key, value in left_inputs.items()}
            right_inputs = {key: value.to(device) for key, value in right_inputs.items()}
            left_vector = model.get_image_features(**left_inputs)
            right_vector = model.get_image_features(**right_inputs)
            loss = loss_function(left_vector, right_vector, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    metrics = {
        "baseModel": args.base_model,
        "holdoutSeason": holdout,
        "trainingPairs": len(train),
        "holdoutPairs": int((frame["season"].astype(str) == holdout).sum()),
        "epochs": args.epochs,
        "lastTrainingLoss": losses[-1] if losses else None,
    }
    (args.output_dir / "training-metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
