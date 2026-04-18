"""Экспорт отдельных CSV-таблиц по transformers-экспериментам для ВКР."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class TransformersThesisTablesExportError(ValueError):
    """Ошибка экспорта таблиц по transformers-экспериментам для ВКР."""


@dataclass(frozen=True)
class TransformersThesisTablesExportPaths:
    """Хранит пути к сохраненным CSV-таблицам и manifest-файлу."""

    metrics_table_path: Path | None
    comparison_table_path: Path | None
    error_table_path: Path | None
    manifest_path: Path


@dataclass(frozen=True)
class TransformersThesisTablesExportResult:
    """Возвращает информацию о сформированных таблицах для ВКР."""

    exported_table_names: tuple[str, ...]
    paths: TransformersThesisTablesExportPaths


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
    """Строит таблицу основных метрик пакетной transformers-оценки."""
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
            {
                "metric_name": "evaluated_rows",
                "metric_value": report.evaluated_rows,
            },
            {
                "metric_name": "skipped_rows_without_label",
                "metric_value": report.skipped_rows_without_label,
            },
        ]
    )


def export_transformers_thesis_tables(
    *,
    evaluation_result: Any = None,
    comparison_result: Any = None,
    error_analysis_result: Any = None,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> TransformersThesisTablesExportResult:
    """Экспортирует отдельные CSV-таблицы по текущим результатам transformers-экспериментов."""
    if evaluation_result is None and comparison_result is None and error_analysis_result is None:
        raise TransformersThesisTablesExportError(
            "Для экспорта таблиц по transformers-экспериментам пока нет данных."
        )

    project_paths.ensure_directories()

    exported_table_names: list[str] = []
    metrics_table_path = None
    comparison_table_path = None
    error_table_path = None

    if evaluation_result is not None:
        metrics_table_path = _get_available_path(
            project_paths.thesis_tables_reports_dir / "transformers_metrics_table.csv"
        )
        _build_metrics_dataframe(evaluation_result).to_csv(
            metrics_table_path,
            index=False,
            encoding="utf-8",
        )
        exported_table_names.append("metrics")

    if comparison_result is not None:
        comparison_table_path = _get_available_path(
            project_paths.thesis_tables_reports_dir / "transformers_model_comparison_table.csv"
        )
        comparison_result.dataframe.to_csv(
            comparison_table_path,
            index=False,
            encoding="utf-8",
        )
        exported_table_names.append("comparison")

    if error_analysis_result is not None:
        error_table_path = _get_available_path(
            project_paths.thesis_tables_reports_dir / "transformers_error_analysis_table.csv"
        )
        error_analysis_result.misclassified_dataframe.to_csv(
            error_table_path,
            index=False,
            encoding="utf-8",
        )
        exported_table_names.append("error_analysis")

    manifest_path = _get_available_path(
        project_paths.thesis_tables_reports_dir / "transformers_tables_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "exported_table_names": exported_table_names,
                "paths": {
                    "metrics_table_path": str(metrics_table_path) if metrics_table_path else "",
                    "comparison_table_path": str(comparison_table_path) if comparison_table_path else "",
                    "error_table_path": str(error_table_path) if error_table_path else "",
                    "manifest_path": str(manifest_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return TransformersThesisTablesExportResult(
        exported_table_names=tuple(exported_table_names),
        paths=TransformersThesisTablesExportPaths(
            metrics_table_path=metrics_table_path,
            comparison_table_path=comparison_table_path,
            error_table_path=error_table_path,
            manifest_path=manifest_path,
        ),
    )
