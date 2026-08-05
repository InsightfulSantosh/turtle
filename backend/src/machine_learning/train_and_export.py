from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.config import paths
from machine_learning.model import build_model_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and export the Turtle visual recommendation artifact")
    parser.add_argument("--source", type=Path, default=paths.model_artifact)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=paths.model_artifact)
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
        f"Exported model v{model['version']}; "
        f"policy={model['evidencePolicy']}; "
        f"visual threshold={model['minimumVisualScore']:.0%}; "
        f"topK={model['topK']}"
    )


if __name__ == "__main__":
    main()
