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
    raw_data_dir: Path
    processed_data_dir: Path
    split_data_dir: Path
    feature_data_dir: Path
    models_dir: Path
    vectorizers_dir: Path
    classifiers_dir: Path
    notebooks_dir: Path
    reports_dir: Path
    dataset_reports_dir: Path
    preprocessing_reports_dir: Path
    split_reports_dir: Path
    vectorization_reports_dir: Path
    training_reports_dir: Path
    metrics_reports_dir: Path
    detailed_metrics_reports_dir: Path
    loading_reports_dir: Path
    inference_reports_dir: Path
    experiment_reports_dir: Path
    comparison_reports_dir: Path
    confidence_reports_dir: Path
    error_analysis_reports_dir: Path
    html_reports_dir: Path
    markdown_reports_dir: Path
    thesis_tables_reports_dir: Path
    docx_reports_dir: Path
    plots_reports_dir: Path
    heatmaps_reports_dir: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "ProjectPaths":
        """Собирает объект путей, вычисляя директории относительно корня проекта."""
        resolved_root = (root or Path(__file__).resolve().parents[2]).resolve()
        return cls(
            root=resolved_root,
            docs_dir=resolved_root / "docs",
            src_dir=resolved_root / "src",
            data_dir=resolved_root / "data",
            raw_data_dir=resolved_root / "data" / "raw",
            processed_data_dir=resolved_root / "data" / "processed",
            split_data_dir=resolved_root / "data" / "splits",
            feature_data_dir=resolved_root / "data" / "features",
            models_dir=resolved_root / "models",
            vectorizers_dir=resolved_root / "models" / "vectorizers",
            classifiers_dir=resolved_root / "models" / "classifiers",
            notebooks_dir=resolved_root / "notebooks",
            reports_dir=resolved_root / "reports",
            dataset_reports_dir=resolved_root / "reports" / "datasets",
            preprocessing_reports_dir=resolved_root / "reports" / "preprocessing",
            split_reports_dir=resolved_root / "reports" / "splits",
            vectorization_reports_dir=resolved_root / "reports" / "vectorization",
            training_reports_dir=resolved_root / "reports" / "training",
            metrics_reports_dir=resolved_root / "reports" / "metrics",
            detailed_metrics_reports_dir=resolved_root / "reports" / "detailed_metrics",
            loading_reports_dir=resolved_root / "reports" / "loading",
            inference_reports_dir=resolved_root / "reports" / "inference",
            experiment_reports_dir=resolved_root / "reports" / "experiments",
            comparison_reports_dir=resolved_root / "reports" / "comparisons",
            confidence_reports_dir=resolved_root / "reports" / "confidence",
            error_analysis_reports_dir=resolved_root / "reports" / "errors",
            html_reports_dir=resolved_root / "reports" / "html",
            markdown_reports_dir=resolved_root / "reports" / "markdown",
            thesis_tables_reports_dir=resolved_root / "reports" / "tables",
            docx_reports_dir=resolved_root / "reports" / "docx",
            plots_reports_dir=resolved_root / "reports" / "plots",
            heatmaps_reports_dir=resolved_root / "reports" / "heatmaps",
        )

    @property
    def managed_directories(self) -> tuple[Path, ...]:
        """Возвращает список директорий, которые должны существовать в проекте."""
        return (
            self.docs_dir,
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.split_data_dir,
            self.feature_data_dir,
            self.models_dir,
            self.vectorizers_dir,
            self.classifiers_dir,
            self.notebooks_dir,
            self.reports_dir,
            self.dataset_reports_dir,
            self.preprocessing_reports_dir,
            self.split_reports_dir,
            self.vectorization_reports_dir,
            self.training_reports_dir,
            self.metrics_reports_dir,
            self.detailed_metrics_reports_dir,
            self.loading_reports_dir,
            self.inference_reports_dir,
            self.experiment_reports_dir,
            self.comparison_reports_dir,
            self.confidence_reports_dir,
            self.error_analysis_reports_dir,
            self.html_reports_dir,
            self.markdown_reports_dir,
            self.thesis_tables_reports_dir,
            self.docx_reports_dir,
            self.plots_reports_dir,
            self.heatmaps_reports_dir,
        )

    def ensure_directories(self) -> None:
        """Создает отсутствующие каталоги проекта перед сохранением артефактов."""
        for directory in self.managed_directories:
            directory.mkdir(parents=True, exist_ok=True)


PROJECT_PATHS = ProjectPaths.from_root()
