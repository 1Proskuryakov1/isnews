"""Обучение базовой модели Logistic Regression на TF-IDF-признаках."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.dataset_split import DatasetSplitResult
from src.isnews.tfidf_vectorization import TfidfVectorizationResult


class LogisticRegressionTrainingError(ValueError):
    """Ошибка обучения базовой модели Logistic Regression."""


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Параметры обучения базового линейного классификатора."""

    max_iter: int = 1000
    solver: str = "lbfgs"
    C: float = 1.0
    random_state: int = 42


@dataclass(frozen=True)
class LogisticRegressionTrainingReport:
    """Содержит краткую сводку по процессу обучения модели."""

    class_labels: tuple[str, ...]
    training_seconds: float
    train_rows: int
    validation_rows: int
    test_rows: int
    coefficient_shape: tuple[int, int]
    intercept_shape: tuple[int, ...]
    iterations_per_class: tuple[int, ...]
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class LogisticRegressionPaths:
    """Хранит пути к сохраненной модели и отчету по обучению."""

    model_path: Path
    report_path: Path


@dataclass(frozen=True)
class LogisticRegressionTrainingResult:
    """Возвращает обученную модель и связанные артефакты."""

    model: LogisticRegression
    config: LogisticRegressionConfig
    report: LogisticRegressionTrainingReport
    paths: LogisticRegressionPaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла для артефактов модели."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "logistic_regression_model"


def _get_available_path(target_path: Path) -> Path:
    """Подбирает свободный путь к файлу, если такой файл уже существует."""
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


def _validate_inputs(
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
) -> None:
    """Проверяет согласованность матриц признаков и разбиения по строкам."""
    if len(split_result.train_dataframe) != vectorization_result.train_matrix.shape[0]:
        raise LogisticRegressionTrainingError(
            "Число строк в train-выборке не совпадает с размером train-матрицы признаков."
        )
    if len(split_result.validation_dataframe) != vectorization_result.validation_matrix.shape[0]:
        raise LogisticRegressionTrainingError(
            "Число строк в validation-выборке не совпадает с размером validation-матрицы."
        )
    if len(split_result.test_dataframe) != vectorization_result.test_matrix.shape[0]:
        raise LogisticRegressionTrainingError(
            "Число строк в test-выборке не совпадает с размером test-матрицы."
        )

    unique_labels = split_result.train_dataframe["label"].astype("string").nunique()
    if unique_labels < 2:
        raise LogisticRegressionTrainingError(
            "Для обучения Logistic Regression в train-выборке нужны минимум два класса."
        )


def _validate_solver_for_class_count(
    *,
    solver_name: str,
    class_count: int,
) -> None:
    """Проверяет, что выбранный solver совместим с числом классов задачи."""
    if class_count > 2 and solver_name == "liblinear":
        raise LogisticRegressionTrainingError(
            "Solver `liblinear` не подходит для многоклассовой классификации в текущей "
            "версии scikit-learn. Используйте, например, `lbfgs` или `saga`."
        )


def _build_model(config: LogisticRegressionConfig) -> LogisticRegression:
    """Создает объект Logistic Regression по заданной конфигурации."""
    return LogisticRegression(
        max_iter=config.max_iter,
        solver=config.solver,
        C=config.C,
        random_state=config.random_state,
    )


def _save_model(
    *,
    model: LogisticRegression,
    vectorization_result: TfidfVectorizationResult,
    project_paths: ProjectPaths,
) -> Path:
    """Сохраняет обученный классификатор в каталог `models/classifiers`."""
    project_paths.ensure_directories()

    model_name = _sanitize_name(
        f"{vectorization_result.paths.feature_directory.name}_logreg.joblib"
    )
    model_path = _get_available_path(project_paths.classifiers_dir / model_name)
    joblib.dump(model, model_path)
    return model_path


def _save_training_report(
    *,
    report: LogisticRegressionTrainingReport,
    config: LogisticRegressionConfig,
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
    paths: LogisticRegressionPaths,
    project_paths: ProjectPaths,
) -> None:
    """Сохраняет JSON-отчет по обучению базовой модели."""
    project_paths.ensure_directories()

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": asdict(config),
        "report": asdict(report),
        "source_split_directory": str(split_result.paths.directory),
        "source_vectorizer_path": str(vectorization_result.paths.vectorizer_path),
        "paths": {
            "model_path": str(paths.model_path),
            "report_path": str(paths.report_path),
        },
    }

    paths.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_logistic_regression(
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
    config: LogisticRegressionConfig | None = None,
) -> LogisticRegressionTrainingResult:
    """Обучает Logistic Regression на TF-IDF-признаках и сохраняет модель."""
    resolved_config = config or LogisticRegressionConfig()
    _validate_inputs(split_result, vectorization_result)
    train_class_count = split_result.train_dataframe["label"].astype("string").nunique()
    _validate_solver_for_class_count(
        solver_name=resolved_config.solver,
        class_count=int(train_class_count),
    )

    train_labels = split_result.train_dataframe["label"].astype("string")
    model = _build_model(resolved_config)
    warning_messages: list[str] = []

    start_time = time.perf_counter()
    try:
        model.fit(vectorization_result.train_matrix, train_labels)
    except ValueError as error:
        raise LogisticRegressionTrainingError(
            f"Не удалось обучить Logistic Regression: {error}"
        ) from error
    training_seconds = round(time.perf_counter() - start_time, 4)

    iterations_per_class = tuple(int(value) for value in np.atleast_1d(model.n_iter_))
    if max(iterations_per_class) >= resolved_config.max_iter:
        warning_messages.append(
            "Достигнут предел `max_iter`. На следующем этапе имеет смысл "
            "оценить качество и при необходимости увеличить число итераций."
        )

    class_labels = tuple(str(label) for label in model.classes_)
    coefficient_shape = (
        int(model.coef_.shape[0]),
        int(model.coef_.shape[1]),
    )
    intercept_shape = tuple(int(value) for value in model.intercept_.shape)

    model_path = _save_model(
        model=model,
        vectorization_result=vectorization_result,
        project_paths=project_paths,
    )
    report_path = _get_available_path(
        project_paths.training_reports_dir
        / f"{Path(model_path).stem}_training_report.json"
    )
    paths = LogisticRegressionPaths(
        model_path=model_path,
        report_path=report_path,
    )

    report = LogisticRegressionTrainingReport(
        class_labels=class_labels,
        training_seconds=training_seconds,
        train_rows=len(split_result.train_dataframe),
        validation_rows=len(split_result.validation_dataframe),
        test_rows=len(split_result.test_dataframe),
        coefficient_shape=coefficient_shape,
        intercept_shape=intercept_shape,
        iterations_per_class=iterations_per_class,
        warning_messages=tuple(warning_messages),
    )
    _save_training_report(
        report=report,
        config=resolved_config,
        split_result=split_result,
        vectorization_result=vectorization_result,
        paths=paths,
        project_paths=project_paths,
    )

    return LogisticRegressionTrainingResult(
        model=model,
        config=resolved_config,
        report=report,
        paths=paths,
    )
