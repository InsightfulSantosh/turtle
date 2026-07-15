from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from season_intelligence import build_model_artifact


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the Turtle Season Intelligence model artifact")
    parser.add_argument("--source", type=Path, default=ROOT / "app" / "generated-data.json")
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "app" / "generated-data.json")
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    vision = json.loads(args.vision.read_text(encoding="utf-8"))
    artifact = build_model_artifact(source, vision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, args.output)

    model = artifact["meta"]["model"]
    print(
        f"Exported model v{model['version']} with {model['trainingRows']} rows; "
        f"LOO WAPE={model['backtest']['wape']:.1%}; "
        f"weights={model['attributeWeight']:.0%}/{model['visualWeight']:.0%}; "
        f"topK={model['topK']}"
    )


if __name__ == "__main__":
    main()
