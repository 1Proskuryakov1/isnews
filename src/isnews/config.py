"""Базовая конфигурация путей проекта."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Хранит основные пути репозитория для последующего переиспользования."""

    root: Path
    docs_dir: Path
    src_dir: Path
    data_dir: Path
    models_dir: Path
    notebooks_dir: Path
    reports_dir: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "ProjectPaths":
        """Собирает объект путей, вычисляя директории относительно корня проекта."""
        resolved_root = (root or Path(__file__).resolve().parents[2]).resolve()
        return cls(
            root=resolved_root,
            docs_dir=resolved_root / "docs",
            src_dir=resolved_root / "src",
            data_dir=resolved_root / "data",
            models_dir=resolved_root / "models",
            notebooks_dir=resolved_root / "notebooks",
            reports_dir=resolved_root / "reports",
        )


PROJECT_PATHS = ProjectPaths.from_root()
