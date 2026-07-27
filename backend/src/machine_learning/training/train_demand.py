from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "top_match_score", "top_attribute_score", "top_vector_score",
    "analogue_mean", "analogue_std", "analogue_count",
]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def pinball(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal P10/P50/P90 demand models.")
    parser.add_argument("data", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--target", default="normalized_demand")
    args = parser.parse_args()
    frame = read_table(args.data)
    required = {"season", args.target, *args.features}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    seasons = sorted(frame["season"].astype(str).unique())
    if len(seasons) < 3:
        raise ValueError("at least three seasons are required for temporal validation")
    holdout = seasons[-1]
    train = frame[frame["season"].astype(str) != holdout]
    valid = frame[frame["season"].astype(str) == holdout]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"featureNames": args.features, "holdoutSeason": holdout, "metrics": {}}
    for name, quantile in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=quantile, n_estimators=2_000, learning_rate=0.025,
            num_leaves=31, min_child_samples=50, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=5.0, random_state=42, n_jobs=-1,
        )
        model.fit(
            train[args.features], train[args.target],
            eval_set=[(valid[args.features], valid[args.target])],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )
        path = args.output_dir / f"{name}.txt"
        model.booster_.save_model(path)
        prediction = np.maximum(model.predict(valid[args.features]), 0)
        manifest[name] = path.name
        manifest["metrics"][name] = {"pinballLoss": pinball(valid[args.target].to_numpy(), prediction, quantile)}
        if name == "p50":
            actual = valid[args.target].to_numpy()
            manifest["metrics"][name]["wape"] = float(
                np.abs(actual - prediction).sum() / max(np.abs(actual).sum(), 1e-9)
            )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
