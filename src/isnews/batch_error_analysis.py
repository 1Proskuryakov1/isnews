"""Анализ ошибочных предсказаний на размеченном CSV-файле."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.isnews.batch_inference_evaluation import (
    BatchInferenceEvaluationReport,
    _find_label_column,
)
from src.isnews.batch_text_inference import BatchTextInferenceResult
from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.text_preprocessing import clean_label_value


class BatchErrorAnalysisError(ValueError):
    """Ошибка анализа неверно классифицированных строк."""


@dataclass(frozen=True)
class BatchErrorAnalysisReport:
    """Содержит сводку по ошибочным предсказаниям модели."""

    source_name: str
    label_column: str
    analyzed_rows: int
    misclassified_rows: int
    correct_rows: int
    error_rate: float
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class BatchErrorAnalysisPaths:
    """Хранит пути к CSV и JSON артефактам анализа ошибок."""

    misclassified_rows_path: Path
    report_path: Path


@dataclass(frozen=True)
class BatchErrorAnalysisResult:
    """Возвращает таблицу ошибочных строк и пути к сохраненным артефактам."""

    misclassified_dataframe: pd.DataFrame
    report: BatchErrorAnalysisReport
    paths: BatchErrorAnalysisPaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла для артефактов анализа."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "batch_error_analysis"


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


def _build_misclassified_dataframe(
    dataframe: pd.DataFrame,
    *,
    label_column: str,
) -> pd.DataFrame:
    """Строит таблицу только с неверно классифицированными строками."""
    normalized_true_labels = dataframe[label_column].astype("string").fillna("").map(
        lambda value: clean_label_value(str(value))
    )
    predicted_labels = dataframe["predicted_label"].astype("string").fillna("")
    valid_mask = normalized_true_labels.ne("") & predicted_labels.ne("")
    error_mask = valid_mask & normalized_true_labels.ne(predicted_labels)

    misclassified_dataframe = dataframe.loc[error_mask].copy()
    misclassified_dataframe.insert(
        0,
        "true_label",
        normalized_true_labels.loc[error_mask].to_numpy(),
    )
    misclassified_dataframe.insert(
        1,
        "predicted_label_normalized",
        predicted_labels.loc[error_mask].to_numpy(),
    )

    preferred_columns: list[str] = [
        "true_label",
        "predicted_label_normalized",
        "predicted_probability",
    ]
    for candidate in ("text", "cleaned_text"):
        if candidate in misclassified_dataframe.columns:
            preferred_columns.append(candidate)
    preferred_columns.extend(
        column
        for column in misclassified_dataframe.columns
        if str(column).startswith("probability_")
    )
    preferred_columns.extend(
        column
        for column in misclassified_dataframe.columns
        if column not in preferred_columns
    )
    return misclassified_dataframe.loc[:, preferred_columns]


def _save_error_analysis_report(
    *,
    report: BatchErrorAnalysisReport,
    misclassified_dataframe: pd.DataFrame,
    paths: BatchErrorAnalysisPaths,
    project_paths: ProjectPaths,
) -> None:
    """Сохраняет JSON-отчет и CSV с ошибочными строками."""
    project_paths.ensure_directories()
    paths.report_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "report": asdict(report),
                "paths": {
                    "misclassified_rows_path": str(paths.misclassified_rows_path),
                    "report_path": str(paths.report_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    misclassified_dataframe.to_csv(
        paths.misclassified_rows_path,
        index=False,
        encoding="utf-8",
    )


def analyze_batch_errors(
    batch_inference_result: BatchTextInferenceResult,
    evaluation_report: BatchInferenceEvaluationReport,
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> BatchErrorAnalysisResult:
    """Сохраняет таблицу неверно классифицированных строк для размеченного CSV."""
    dataframe = batch_inference_result.dataframe.copy()
    if dataframe.empty:
        raise BatchErrorAnalysisError(
            "Таблица пакетного инференса пуста. Анализ ошибок невозможен."
        )
    if "predicted_label" not in dataframe.columns:
        raise BatchErrorAnalysisError(
            "В таблице пакетного инференса отсутствует колонка `predicted_label`."
        )

    label_column = _find_label_column(dataframe.columns)
    if label_column != evaluation_report.label_column:
        raise BatchErrorAnalysisError(
            "Колонка истинного класса в анализе ошибок не совпадает с колонкой из отчета оценки."
        )

    misclassified_dataframe = _build_misclassified_dataframe(
        dataframe,
        label_column=label_column,
    )
    analyzed_rows = evaluation_report.evaluated_rows
    misclassified_rows = len(misclassified_dataframe)
    correct_rows = analyzed_rows - misclassified_rows

    warning_messages: list[str] = []
    if misclassified_rows == 0:
        warning_messages.append(
            "Ошибочных классификаций не найдено. Таблица сохранена пустой."
        )

    report = BatchErrorAnalysisReport(
        source_name=batch_inference_result.report.source_name,
        label_column=label_column,
        analyzed_rows=analyzed_rows,
        misclassified_rows=misclassified_rows,
        correct_rows=correct_rows,
        error_rate=round(misclassified_rows / analyzed_rows, 6),
        warning_messages=tuple(warning_messages),
    )

    base_name = _sanitize_name(batch_inference_result.report.source_name)
    paths = BatchErrorAnalysisPaths(
        misclassified_rows_path=_get_available_path(
            project_paths.error_analysis_reports_dir / f"{base_name}_misclassified.csv"
        ),
        report_path=_get_available_path(
            project_paths.error_analysis_reports_dir / f"{base_name}_error_analysis.json"
        ),
    )
    _save_error_analysis_report(
        report=report,
        misclassified_dataframe=misclassified_dataframe,
        paths=paths,
        project_paths=project_paths,
    )
    return BatchErrorAnalysisResult(
        misclassified_dataframe=misclassified_dataframe,
        report=report,
        paths=paths,
    )
