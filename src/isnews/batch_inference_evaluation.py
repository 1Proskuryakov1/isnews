"""Оценка качества пакетного инференса на размеченном CSV-файле."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.isnews.batch_text_inference import BatchTextInferenceResult
from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.data_loading import LABEL_COLUMN_CANDIDATES
from src.isnews.text_preprocessing import clean_label_value


class BatchInferenceEvaluationError(ValueError):
    """Ошибка расчета метрик по результатам пакетного инференса."""


@dataclass(frozen=True)
class BatchInferenceEvaluationReport:
    """Содержит сводку по качеству пакетной классификации."""

    source_name: str
    label_column: str
    evaluated_rows: int
    skipped_rows_without_label: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    class_labels: tuple[str, ...]
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class BatchInferenceEvaluationPaths:
    """Хранит пути к JSON-отчету и CSV матрицы ошибок."""

    report_path: Path
    confusion_matrix_path: Path


@dataclass(frozen=True)
class BatchInferenceEvaluationResult:
    """Возвращает сводку по метрикам и матрицу ошибок."""

    report: BatchInferenceEvaluationReport
    confusion_matrix_dataframe: pd.DataFrame
    paths: BatchInferenceEvaluationPaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла для артефактов оценки."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "batch_inference_evaluation"


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


def _find_label_column(columns: pd.Index) -> str:
    """Находит колонку с истинным классом по тем же алиасам, что и при обучении."""
    normalized_mapping = {str(column).strip().casefold(): str(column) for column in columns}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in normalized_mapping:
            return normalized_mapping[candidate]

    raise BatchInferenceEvaluationError(
        "Во входном CSV не найдена колонка с истинным классом. "
        "Для оценки пакетного инференса поддерживаются алиасы вроде `label`, "
        "`category`, `topic`, `метка`, `категория`."
    )


def _build_class_labels(batch_inference_result: BatchTextInferenceResult, true_labels: pd.Series) -> tuple[str, ...]:
    """Собирает порядок классов для метрик и матрицы ошибок."""
    ordered_labels = list(batch_inference_result.report.class_labels)
    for label in sorted(set(true_labels.tolist())):
        if label not in ordered_labels:
            ordered_labels.append(label)
    return tuple(ordered_labels)


def _save_evaluation_report(
    *,
    report: BatchInferenceEvaluationReport,
    confusion_matrix_dataframe: pd.DataFrame,
    paths: BatchInferenceEvaluationPaths,
    project_paths: ProjectPaths,
) -> None:
    """Сохраняет JSON-отчет и CSV матрицы ошибок по пакетной оценке."""
    project_paths.ensure_directories()

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "report": asdict(report),
        "paths": {
            "report_path": str(paths.report_path),
            "confusion_matrix_path": str(paths.confusion_matrix_path),
        },
    }

    paths.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    confusion_matrix_dataframe.to_csv(paths.confusion_matrix_path, encoding="utf-8")


def evaluate_batch_inference(
    batch_inference_result: BatchTextInferenceResult,
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> BatchInferenceEvaluationResult:
    """Рассчитывает метрики качества, если в таблице пакетного инференса есть истинные метки."""
    dataframe = batch_inference_result.dataframe.copy()
    if dataframe.empty:
        raise BatchInferenceEvaluationError(
            "Таблица пакетного инференса пуста. Оценка качества невозможна."
        )
    if "predicted_label" not in dataframe.columns:
        raise BatchInferenceEvaluationError(
            "В таблице пакетного инференса отсутствует колонка `predicted_label`."
        )

    label_column = _find_label_column(dataframe.columns)
    true_labels = dataframe[label_column].astype("string").fillna("").map(
        lambda value: clean_label_value(str(value))
    )
    predicted_labels = dataframe["predicted_label"].astype("string").fillna("")

    valid_mask = true_labels.ne("") & predicted_labels.ne("")
    evaluated_rows = int(valid_mask.sum())
    skipped_rows_without_label = int((~valid_mask).sum())

    if evaluated_rows == 0:
        raise BatchInferenceEvaluationError(
            "Для оценки пакетного инференса не найдено строк, где одновременно есть истинная метка и предсказание."
        )

    filtered_true_labels = true_labels.loc[valid_mask]
    filtered_predicted_labels = predicted_labels.loc[valid_mask]
    class_labels = _build_class_labels(batch_inference_result, filtered_true_labels)

    accuracy = round(float(accuracy_score(filtered_true_labels, filtered_predicted_labels)), 6)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        filtered_true_labels,
        filtered_predicted_labels,
        labels=list(class_labels),
        average="macro",
        zero_division=0,
    )

    confusion_matrix_values = confusion_matrix(
        filtered_true_labels,
        filtered_predicted_labels,
        labels=list(class_labels),
    )
    confusion_matrix_dataframe = pd.DataFrame(
        confusion_matrix_values,
        index=class_labels,
        columns=class_labels,
    )
    confusion_matrix_dataframe.index.name = "Истинный класс"
    confusion_matrix_dataframe.columns.name = "Предсказанный класс"

    warning_messages: list[str] = []
    if accuracy < 0.7:
        warning_messages.append(
            "Accuracy на размеченном CSV ниже 0.70. Стоит проверить качество модели и состав тестового файла."
        )
    if skipped_rows_without_label > 0:
        warning_messages.append(
            "Часть строк исключена из оценки, потому что в них отсутствовала истинная метка или предсказание."
        )

    report = BatchInferenceEvaluationReport(
        source_name=batch_inference_result.report.source_name,
        label_column=label_column,
        evaluated_rows=evaluated_rows,
        skipped_rows_without_label=skipped_rows_without_label,
        accuracy=accuracy,
        precision_macro=round(float(precision), 6),
        recall_macro=round(float(recall), 6),
        f1_macro=round(float(f1_score), 6),
        class_labels=class_labels,
        warning_messages=tuple(warning_messages),
    )

    report_path = _get_available_path(
        project_paths.inference_reports_dir
        / f"{_sanitize_name(batch_inference_result.report.source_name)}_batch_evaluation.json"
    )
    confusion_matrix_path = _get_available_path(
        project_paths.inference_reports_dir
        / f"{_sanitize_name(batch_inference_result.report.source_name)}_batch_confusion_matrix.csv"
    )
    paths = BatchInferenceEvaluationPaths(
        report_path=report_path,
        confusion_matrix_path=confusion_matrix_path,
    )
    _save_evaluation_report(
        report=report,
        confusion_matrix_dataframe=confusion_matrix_dataframe,
        paths=paths,
        project_paths=project_paths,
    )

    return BatchInferenceEvaluationResult(
        report=report,
        confusion_matrix_dataframe=confusion_matrix_dataframe,
        paths=paths,
    )
