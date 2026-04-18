"""Расчет метрик качества для обученной модели классификации."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.dataset_split import DatasetSplitResult
from src.isnews.logistic_regression_training import LogisticRegressionTrainingResult
from src.isnews.tfidf_vectorization import TfidfVectorizationResult


class ModelEvaluationError(ValueError):
    """Ошибка расчета метрик качества модели."""


@dataclass(frozen=True)
class MetricScores:
    """Хранит основные метрики классификации для одной подвыборки."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    support: int


@dataclass(frozen=True)
class ModelEvaluationReport:
    """Содержит метрики для train, validation и test."""

    class_labels: tuple[str, ...]
    train_metrics: MetricScores
    validation_metrics: MetricScores
    test_metrics: MetricScores
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class ModelEvaluationPaths:
    """Хранит путь к сохраненному отчету по метрикам."""

    report_path: Path


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Возвращает рассчитанные метрики и путь к JSON-отчету."""

    report: ModelEvaluationReport
    paths: ModelEvaluationPaths


def _get_available_path(target_path: Path) -> Path:
    """Подбирает свободный путь к файлу отчета."""
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


def _calculate_metric_scores(true_labels: list[str], predicted_labels: np.ndarray) -> MetricScores:
    """Рассчитывает Accuracy, Precision, Recall и F1-score для одной подвыборки."""
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    return MetricScores(
        accuracy=_round_metric(accuracy),
        precision_macro=_round_metric(precision_macro),
        recall_macro=_round_metric(recall_macro),
        f1_macro=_round_metric(f1_macro),
        precision_weighted=_round_metric(precision_weighted),
        recall_weighted=_round_metric(recall_weighted),
        f1_weighted=_round_metric(f1_weighted),
        support=len(true_labels),
    )


def _validate_inputs(
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
) -> None:
    """Проверяет согласованность выборок и матриц признаков перед оценкой."""
    if len(split_result.train_dataframe) != vectorization_result.train_matrix.shape[0]:
        raise ModelEvaluationError(
            "Train-выборка и train-матрица признаков имеют разный размер."
        )
    if len(split_result.validation_dataframe) != vectorization_result.validation_matrix.shape[0]:
        raise ModelEvaluationError(
            "Validation-выборка и validation-матрица признаков имеют разный размер."
        )
    if len(split_result.test_dataframe) != vectorization_result.test_matrix.shape[0]:
        raise ModelEvaluationError(
            "Test-выборка и test-матрица признаков имеют разный размер."
        )


def _build_warning_messages(report: ModelEvaluationReport) -> tuple[str, ...]:
    """Формирует предупреждения по результатам оценки качества."""
    warning_messages: list[str] = []

    if report.validation_metrics.accuracy < 0.7:
        warning_messages.append(
            "Accuracy на validation ниже 0.70. На следующих этапах потребуется "
            "подобрать параметры модели и улучшить признаки."
        )
    if report.test_metrics.accuracy < 0.7:
        warning_messages.append(
            "Accuracy на test ниже 0.70. Итоговая версия ВКР потребует доработки "
            "модели или данных."
        )

    return tuple(warning_messages)


def _save_evaluation_report(
    *,
    report: ModelEvaluationReport,
    training_result: LogisticRegressionTrainingResult,
    project_paths: ProjectPaths,
) -> Path:
    """Сохраняет JSON-отчет с итоговыми метриками модели."""
    project_paths.ensure_directories()

    report_path = _get_available_path(
        project_paths.metrics_reports_dir
        / f"{Path(training_result.paths.model_path).stem}_metrics_report.json"
    )

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_path": str(training_result.paths.model_path),
        "training_report_path": str(training_result.paths.report_path),
        "metrics": asdict(report),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def evaluate_trained_model(
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
    training_result: LogisticRegressionTrainingResult,
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> ModelEvaluationResult:
    """Рассчитывает метрики качества обученной модели на train/validation/test."""
    _validate_inputs(split_result, vectorization_result)

    model = training_result.model

    train_true = split_result.train_dataframe["label"].astype("string").tolist()
    validation_true = split_result.validation_dataframe["label"].astype("string").tolist()
    test_true = split_result.test_dataframe["label"].astype("string").tolist()

    try:
        train_predictions = model.predict(vectorization_result.train_matrix)
        validation_predictions = model.predict(vectorization_result.validation_matrix)
        test_predictions = model.predict(vectorization_result.test_matrix)
    except ValueError as error:
        raise ModelEvaluationError(
            f"Не удалось получить предсказания модели: {error}"
        ) from error

    base_report = ModelEvaluationReport(
        class_labels=tuple(str(label) for label in model.classes_),
        train_metrics=_calculate_metric_scores(train_true, train_predictions),
        validation_metrics=_calculate_metric_scores(validation_true, validation_predictions),
        test_metrics=_calculate_metric_scores(test_true, test_predictions),
        warning_messages=tuple(),
    )
    final_report = ModelEvaluationReport(
        class_labels=base_report.class_labels,
        train_metrics=base_report.train_metrics,
        validation_metrics=base_report.validation_metrics,
        test_metrics=base_report.test_metrics,
        warning_messages=_build_warning_messages(base_report),
    )

    report_path = _save_evaluation_report(
        report=final_report,
        training_result=training_result,
        project_paths=project_paths,
    )

    return ModelEvaluationResult(
        report=final_report,
        paths=ModelEvaluationPaths(report_path=report_path),
    )
