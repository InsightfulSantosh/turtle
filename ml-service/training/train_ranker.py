from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRanker, Pool

from season_intelligence.ranking import RANK_FEATURES


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the historical-style learning-to-rank model.")
    parser.add_argument("data", type=Path, help="CSV/Parquet with query_id, season, relevance and rank features")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frame = read_table(args.data)
    required = {"query_id", "season", "relevance", *RANK_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    seasons = sorted(frame["season"].astype(str).unique())
    if len(seasons) < 3:
        raise ValueError("at least three seasons are required for a temporal ranker validation")
    holdout = seasons[-1]
    train = frame[frame["season"].astype(str) != holdout].sort_values("query_id")
    valid = frame[frame["season"].astype(str) == holdout].sort_values("query_id")
    train_pool = Pool(train[list(RANK_FEATURES)], train["relevance"], group_id=train["query_id"])
    valid_pool = Pool(valid[list(RANK_FEATURES)], valid["relevance"], group_id=valid["query_id"])
    model = CatBoostRanker(
        loss_function="YetiRankPairwise", eval_metric="NDCG:top=10", iterations=1_500,
        learning_rate=0.04, depth=8, l2_leaf_reg=8, random_seed=42,
        allow_writing_files=False, verbose=100,
    )
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output)
    metrics = {
        "holdoutSeason": holdout, "trainingRows": len(train), "validationRows": len(valid),
        "features": list(RANK_FEATURES), "bestIteration": model.get_best_iteration(),
        "bestScore": model.get_best_score(),
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
