"""Экспорт PNG-графиков по результатам transformers-экспериментов."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class TransformersPlotExportError(ValueError):
    """Ошибка экспорта графиков по transformers-экспериментам."""


@dataclass(frozen=True)
class TransformersPlotExportPaths:
    """Хранит пути к сохраненным графикам и manifest-файлу."""

    evaluation_plot_path: Path | None
    comparison_plot_path: Path | None
    manifest_path: Path


@dataclass(frozen=True)
class TransformersPlotExportResult:
    """Возвращает информацию о сформированных графиках."""

    exported_plot_names: tuple[str, ...]
    paths: TransformersPlotExportPaths


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


def _build_evaluation_dataframe(evaluation_result: Any) -> pd.DataFrame:
    """Строит таблицу метрик пакетной оценки."""
    report = evaluation_result.report
    return pd.DataFrame(
        [
            {
                "metric_name": "accuracy",
                "metric_value": report.accuracy,
            },
            {
                "metric_name": "precision_macro",
                "metric_value": report.precision_macro,
            },
            {
                "metric_name": "recall_macro",
                "metric_value": report.recall_macro,
            },
            {
                "metric_name": "f1_macro",
                "metric_value": report.f1_macro,
            },
        ]
    )


def _export_evaluation_plot(
    evaluation_result: Any,
    *,
    target_path: Path,
) -> None:
    """Сохраняет столбчатый график метрик пакетного transformers-инференса."""
    dataframe = _build_evaluation_dataframe(evaluation_result)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        dataframe["metric_name"].tolist(),
        dataframe["metric_value"].tolist(),
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Значение метрики")
    axis.set_title("Метрики пакетного transformers-инференса")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(target_path, dpi=180)
    plt.close(figure)


def _export_comparison_plot(
    comparison_result: Any,
    *,
    target_path: Path,
) -> None:
    """Сохраняет график сравнения запусков по accuracy."""
    comparison_dataframe = comparison_result.dataframe.copy()
    if comparison_dataframe.empty:
        raise TransformersPlotExportError(
            "Таблица сравнения transformers-экспериментов пуста."
        )

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        comparison_dataframe["source_name"].astype(str).tolist(),
        comparison_dataframe["accuracy"].fillna(0).tolist(),
        color="#6b8e23",
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("Сравнение transformers-запусков по accuracy")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(target_path, dpi=180)
    plt.close(figure)


def export_transformers_plots(
    *,
    evaluation_result: Any = None,
    comparison_result: Any = None,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> TransformersPlotExportResult:
    """Экспортирует PNG-графики по текущим результатам transformers-экспериментов."""
    if evaluation_result is None and comparison_result is None:
        raise TransformersPlotExportError(
            "Для экспорта графиков по transformers-экспериментам пока нет данных."
        )

    project_paths.ensure_directories()

    exported_plot_names: list[str] = []
    evaluation_plot_path = None
    comparison_plot_path = None

    if evaluation_result is not None:
        evaluation_plot_path = _get_available_path(
            project_paths.plots_reports_dir / "transformers_evaluation_plot.png"
        )
        _export_evaluation_plot(evaluation_result, target_path=evaluation_plot_path)
        exported_plot_names.append("evaluation")

    if comparison_result is not None:
        comparison_plot_path = _get_available_path(
            project_paths.plots_reports_dir / "transformers_comparison_plot.png"
        )
        _export_comparison_plot(comparison_result, target_path=comparison_plot_path)
        exported_plot_names.append("comparison")

    manifest_path = _get_available_path(
        project_paths.plots_reports_dir / "transformers_plots_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "exported_plot_names": exported_plot_names,
                "paths": {
                    "evaluation_plot_path": str(evaluation_plot_path) if evaluation_plot_path else "",
                    "comparison_plot_path": str(comparison_plot_path) if comparison_plot_path else "",
                    "manifest_path": str(manifest_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return TransformersPlotExportResult(
        exported_plot_names=tuple(exported_plot_names),
        paths=TransformersPlotExportPaths(
            evaluation_plot_path=evaluation_plot_path,
            comparison_plot_path=comparison_plot_path,
            manifest_path=manifest_path,
        ),
    )
