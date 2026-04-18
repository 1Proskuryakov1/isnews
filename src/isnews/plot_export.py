"""Экспорт PNG-графиков по результатам обучения и сравнению моделей."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class PlotExportError(ValueError):
    """Ошибка экспорта графиков."""


@dataclass(frozen=True)
class PlotExportPaths:
    """Хранит пути к сохраненным графикам и manifest-файлу."""

    metrics_plot_path: Path | None
    comparison_plot_path: Path | None
    manifest_path: Path


@dataclass(frozen=True)
class PlotExportResult:
    """Возвращает информацию о сформированных графиках."""

    exported_plot_names: tuple[str, ...]
    paths: PlotExportPaths


def _get_available_path(target_path: Path) -> Path:
    """Подбирает свободный путь к файлу, если имя уже занято."""
    if not target_path.exists():
        return target_path

    counter = 1
    while True:
        candidate = target_path.with_name(
            f"{target_path.stem}_{counter}{target_path.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def _build_metrics_dataframe(evaluation_result: Any) -> pd.DataFrame:
    """Строит таблицу метрик по выборкам."""
    report = evaluation_result.report
    return pd.DataFrame(
        [
            {
                "split_name": "train",
                "accuracy": report.train_metrics.accuracy,
                "f1_macro": report.train_metrics.f1_macro,
            },
            {
                "split_name": "validation",
                "accuracy": report.validation_metrics.accuracy,
                "f1_macro": report.validation_metrics.f1_macro,
            },
            {
                "split_name": "test",
                "accuracy": report.test_metrics.accuracy,
                "f1_macro": report.test_metrics.f1_macro,
            },
        ]
    )


def _export_metrics_plot(
    evaluation_result: Any,
    *,
    target_path: Path,
) -> None:
    """Сохраняет столбчатый график accuracy и F1 по выборкам."""
    dataframe = _build_metrics_dataframe(evaluation_result)
    figure, axis = plt.subplots(figsize=(8, 5))
    x_positions = range(len(dataframe))
    bar_width = 0.35

    axis.bar(
        [position - bar_width / 2 for position in x_positions],
        dataframe["accuracy"],
        width=bar_width,
        label="Accuracy",
        color="#1f77b4",
    )
    axis.bar(
        [position + bar_width / 2 for position in x_positions],
        dataframe["f1_macro"],
        width=bar_width,
        label="F1 macro",
        color="#ff7f0e",
    )
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(dataframe["split_name"].tolist())
    axis.set_ylim(0, 1)
    axis.set_ylabel("Значение метрики")
    axis.set_title("Качество модели по выборкам")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(target_path, dpi=180)
    plt.close(figure)


def _export_comparison_plot(
    comparison_result: Any,
    *,
    target_path: Path,
) -> None:
    """Сохраняет график сравнения моделей по validation accuracy."""
    comparison_dataframe = comparison_result.dataframe.copy()
    if comparison_dataframe.empty:
        raise PlotExportError(
            "Таблица сравнения моделей пуста. Невозможно построить график."
        )

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        comparison_dataframe["model_name"].astype(str).tolist(),
        comparison_dataframe["validation_accuracy"].fillna(0).tolist(),
        color="#2ca02c",
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Validation Accuracy")
    axis.set_title("Сравнение моделей по validation accuracy")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(target_path, dpi=180)
    plt.close(figure)


def export_plots(
    *,
    evaluation_result: Any = None,
    comparison_result: Any = None,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> PlotExportResult:
    """Экспортирует PNG-графики по текущим результатам проекта."""
    if evaluation_result is None and comparison_result is None:
        raise PlotExportError(
            "Для экспорта графиков пока нет данных. Сначала рассчитайте метрики модели или сравнение моделей."
        )

    project_paths.ensure_directories()

    exported_plot_names: list[str] = []
    metrics_plot_path = None
    comparison_plot_path = None

    if evaluation_result is not None:
        metrics_plot_path = _get_available_path(
            project_paths.plots_reports_dir / "metrics_plot.png"
        )
        _export_metrics_plot(evaluation_result, target_path=metrics_plot_path)
        exported_plot_names.append("metrics")

    if comparison_result is not None:
        comparison_plot_path = _get_available_path(
            project_paths.plots_reports_dir / "model_comparison_plot.png"
        )
        _export_comparison_plot(comparison_result, target_path=comparison_plot_path)
        exported_plot_names.append("comparison")

    manifest_path = _get_available_path(
        project_paths.plots_reports_dir / "plots_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "exported_plot_names": exported_plot_names,
                "paths": {
                    "metrics_plot_path": str(metrics_plot_path) if metrics_plot_path else "",
                    "comparison_plot_path": str(comparison_plot_path) if comparison_plot_path else "",
                    "manifest_path": str(manifest_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return PlotExportResult(
        exported_plot_names=tuple(exported_plot_names),
        paths=PlotExportPaths(
            metrics_plot_path=metrics_plot_path,
            comparison_plot_path=comparison_plot_path,
            manifest_path=manifest_path,
        ),
    )
