from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the hierarchy summing matrix used by MinTrace reconciliation.")
    parser.add_argument("data", type=Path, help="CSV/Parquet with one row per bottom demand series")
    parser.add_argument("output", type=Path)
    parser.add_argument("--bottom", default="series_id")
    parser.add_argument("--levels", nargs="+", default=["category", "channel", "region"])
    args = parser.parse_args()
    frame = pd.read_parquet(args.data) if args.data.suffix.lower() == ".parquet" else pd.read_csv(args.data)
    required = {args.bottom, *args.levels}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    bottom_frame = frame[[args.bottom, *args.levels]].drop_duplicates(args.bottom).sort_values(args.bottom)
    if bottom_frame[args.bottom].duplicated().any():
        raise ValueError("each bottom series must map to exactly one hierarchy path")
    bottom_names = bottom_frame[args.bottom].astype(str).tolist()
    nodes: list[tuple[str, set[str]]] = [("total", set(bottom_names))]
    for depth in range(1, len(args.levels) + 1):
        columns = args.levels[:depth]
        grouper = columns[0] if len(columns) == 1 else columns
        for key, group in bottom_frame.groupby(grouper, sort=True, dropna=False):
            values = key if isinstance(key, tuple) else (key,)
            name = "/".join(f"{column}={value}" for column, value in zip(columns, values, strict=True))
            nodes.append((name, set(group[args.bottom].astype(str))))
    existing = {name for name, _ in nodes}
    nodes.extend((f"series={name}", {name}) for name in bottom_names if f"series={name}" not in existing)
    matrix = [[1.0 if bottom in members else 0.0 for bottom in bottom_names] for _, members in nodes]
    artifact = {
        "nodeNames": [name for name, _ in nodes],
        "bottomNames": bottom_names,
        "summingMatrix": matrix,
        "residualCovariance": None,
        "note": "Fit residual covariance from rolling temporal validation before production activation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
