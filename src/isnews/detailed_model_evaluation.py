"""Подробная оценка модели: поклассовые метрики и матрицы ошибок."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.dataset_split import DatasetSplitResult
from src.isnews.logistic_regression_training import LogisticRegressionTrainingResult
from src.isnews.tfidf_vectorization import TfidfVectorizationResult


class DetailedModelEvaluationError(ValueError):
    """Ошибка построения подробного отчета по качеству модели."""


@dataclass(frozen=True)
class PerClassMetrics:
    """Хранит precision, recall, f1-score и support для одного класса."""

    label: str
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class DetailedSplitEvaluation:
    """Содержит детальный отчет по одной подвыборке."""

    split_name: str
    class_labels: tuple[str, ...]
    per_class_metrics: tuple[PerClassMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class DetailedModelEvaluationReport:
    """Содержит детальные отчеты по validation и test."""

    validation: DetailedSplitEvaluation
    test: DetailedSplitEvaluation


@dataclass(frozen=True)
class DetailedModelEvaluationPaths:
    """Хранит пути к JSON-отчету и CSV-матрицам ошибок."""

    report_path: Path
    validation_confusion_matrix_path: Path
    test_confusion_matrix_path: Path


@dataclass(frozen=True)
class DetailedModelEvaluationResult:
    """Возвращает детальный отчет и пути к сохраненным артефактам."""

    report: DetailedModelEvaluationReport
    paths: DetailedModelEvaluationPaths


def _get_available_path(target_path: Path) -> Path:
    """Подбирает свободный путь к файлу."""
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


def _round_metric(value: float) -> float:
    """Округляет метрику до четырех знаков после запятой."""
    return round(float(value), 4)


def _validate_inputs(
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
) -> None:
    """Проверяет согласованность подвыборок и матриц признаков."""
    if len(split_result.validation_dataframe) != vectorization_result.validation_matrix.shape[0]:
        raise DetailedModelEvaluationError(
            "Validation-выборка и validation-матрица признаков имеют разный размер."
        )
    if len(split_result.test_dataframe) != vectorization_result.test_matrix.shape[0]:
        raise DetailedModelEvaluationError(
            "Test-выборка и test-матрица признаков имеют разный размер."
        )


def _build_split_evaluation(
    *,
    split_name: str,
    true_labels: list[str],
    predicted_labels: np.ndarray,
    class_labels: tuple[str, ...],
) -> DetailedSplitEvaluation:
    """Строит поклассовый отчет и матрицу ошибок для одной подвыборки."""
    raw_report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(class_labels),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(class_labels),
    )

    per_class_metrics = tuple(
        PerClassMetrics(
            label=label,
            precision=_round_metric(raw_report[label]["precision"]),
            recall=_round_metric(raw_report[label]["recall"]),
            f1_score=_round_metric(raw_report[label]["f1-score"]),
            support=int(raw_report[label]["support"]),
        )
        for label in class_labels
    )

    warning_messages: list[str] = []
    missing_support_labels = [
        metrics.label for metrics in per_class_metrics if metrics.support == 0
    ]
    if missing_support_labels:
        warning_messages.append(
            f"В выборке `{split_name}` отсутствуют примеры классов: "
            f"{', '.join(missing_support_labels)}."
        )

    return DetailedSplitEvaluation(
        split_name=split_name,
        class_labels=class_labels,
        per_class_metrics=per_class_metrics,
        confusion_matrix=tuple(
            tuple(int(value) for value in row)
            for row in matrix.tolist()
        ),
        warning_messages=tuple(warning_messages),
    )


def _save_confusion_matrix_csv(
    *,
    matrix: tuple[tuple[int, ...], ...],
    labels: tuple[str, ...],
    target_path: Path,
) -> None:
    """Сохраняет матрицу ошибок в CSV-файл."""
    dataframe = pd.DataFrame(matrix, index=labels, columns=labels)
    dataframe.index.name = "Истинный класс"
    dataframe.columns.name = "Предсказанный класс"
    dataframe.to_csv(target_path, encoding="utf-8")


def _save_detailed_report(
    *,
    report: DetailedModelEvaluationReport,
    training_result: LogisticRegressionTrainingResult,
    paths: DetailedModelEvaluationPaths,
) -> None:
    """Сохраняет JSON-файл с детальным отчетом по классам."""
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_path": str(training_result.paths.model_path),
        "training_report_path": str(training_result.paths.report_path),
        "paths": {
            "validation_confusion_matrix_path": str(paths.validation_confusion_matrix_path),
            "test_confusion_matrix_path": str(paths.test_confusion_matrix_path),
        },
        "report": asdict(report),
    }
    paths.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def evaluate_model_in_detail(
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
    training_result: LogisticRegressionTrainingResult,
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> DetailedModelEvaluationResult:
    """Строит поклассовый отчет и матрицы ошибок для validation и test."""
    _validate_inputs(split_result, vectorization_result)

    model = training_result.model
    class_labels = tuple(str(label) for label in model.classes_)

    validation_true = split_result.validation_dataframe["label"].astype("string").tolist()
    test_true = split_result.test_dataframe["label"].astype("string").tolist()

    try:
        validation_predictions = model.predict(vectorization_result.validation_matrix)
        test_predictions = model.predict(vectorization_result.test_matrix)
    except ValueError as error:
        raise DetailedModelEvaluationError(
            f"Не удалось получить предсказания модели для подробного отчета: {error}"
        ) from error

    validation_report = _build_split_evaluation(
        split_name="validation",
        true_labels=validation_true,
        predicted_labels=validation_predictions,
        class_labels=class_labels,
    )
    test_report = _build_split_evaluation(
        split_name="test",
        true_labels=test_true,
        predicted_labels=test_predictions,
        class_labels=class_labels,
    )

    project_paths.ensure_directories()
    report_path = _get_available_path(
        project_paths.detailed_metrics_reports_dir
        / f"{Path(training_result.paths.model_path).stem}_detailed_report.json"
    )
    validation_confusion_matrix_path = _get_available_path(
        project_paths.detailed_metrics_reports_dir
        / f"{Path(training_result.paths.model_path).stem}_validation_confusion_matrix.csv"
    )
    test_confusion_matrix_path = _get_available_path(
        project_paths.detailed_metrics_reports_dir
        / f"{Path(training_result.paths.model_path).stem}_test_confusion_matrix.csv"
    )

    paths = DetailedModelEvaluationPaths(
        report_path=report_path,
        validation_confusion_matrix_path=validation_confusion_matrix_path,
        test_confusion_matrix_path=test_confusion_matrix_path,
    )
    _save_confusion_matrix_csv(
        matrix=validation_report.confusion_matrix,
        labels=validation_report.class_labels,
        target_path=validation_confusion_matrix_path,
    )
    _save_confusion_matrix_csv(
        matrix=test_report.confusion_matrix,
        labels=test_report.class_labels,
        target_path=test_confusion_matrix_path,
    )

    report = DetailedModelEvaluationReport(
        validation=validation_report,
        test=test_report,
    )
    _save_detailed_report(
        report=report,
        training_result=training_result,
        paths=paths,
    )

    return DetailedModelEvaluationResult(
        report=report,
        paths=paths,
    )
