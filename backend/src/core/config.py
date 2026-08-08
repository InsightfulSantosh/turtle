"""Central path and environment configuration for local and deployed runtimes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _repository_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "backend").is_dir() and (candidate / "frontend").is_dir():
            return candidate
    return Path.cwd().resolve()


def _environment_path(name: str, default: Path) -> Path:
    configured = os.getenv(name)
    return Path(configured).expanduser().resolve() if configured else default.resolve()


@dataclass(frozen=True)
class ProjectPaths:
    repository: Path
    data: Path


REPOSITORY_ROOT = _repository_root()

paths = ProjectPaths(
    repository=REPOSITORY_ROOT,
    data=_environment_path("TURTLE_DATA_ROOT", REPOSITORY_ROOT / "DATA"),
)
