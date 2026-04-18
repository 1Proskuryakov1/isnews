"""Обучение базовой модели Multinomial Naive Bayes на TF-IDF-признаках."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.naive_bayes import MultinomialNB

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.dataset_split import DatasetSplitResult
from src.isnews.tfidf_vectorization import TfidfVectorizationResult


class MultinomialNBTrainingError(ValueError):
    """Ошибка обучения модели Multinomial Naive Bayes."""


@dataclass(frozen=True)
class MultinomialNBConfig:
    """Параметры обучения базового вероятностного классификатора."""

    alpha: float = 1.0


@dataclass(frozen=True)
class MultinomialNBTrainingReport:
    """Содержит краткую сводку по обучению MultinomialNB."""

    model_name: str
    class_labels: tuple[str, ...]
    training_seconds: float
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_count: int
    class_log_prior_shape: tuple[int, ...]
    alpha: float
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class MultinomialNBPaths:
    """Хранит пути к сохраненной модели и отчету по обучению."""

    model_path: Path
    report_path: Path


@dataclass(frozen=True)
class MultinomialNBTrainingResult:
    """Возвращает обученную модель и связанные артефакты."""

    model: MultinomialNB
    config: MultinomialNBConfig
    report: MultinomialNBTrainingReport
    paths: MultinomialNBPaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла для артефактов модели."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "multinomial_nb_model"


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


def _validate_inputs(
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
) -> None:
    """Проверяет согласованность выборок и матриц признаков."""
    if len(split_result.train_dataframe) != vectorization_result.train_matrix.shape[0]:
        raise MultinomialNBTrainingError(
            "Число строк в train-выборке не совпадает с размером train-матрицы признаков."
        )
    if len(split_result.validation_dataframe) != vectorization_result.validation_matrix.shape[0]:
        raise MultinomialNBTrainingError(
            "Число строк в validation-выборке не совпадает с размером validation-матрицы."
        )
    if len(split_result.test_dataframe) != vectorization_result.test_matrix.shape[0]:
        raise MultinomialNBTrainingError(
            "Число строк в test-выборке не совпадает с размером test-матрицы."
        )
    unique_labels = split_result.train_dataframe["label"].astype("string").nunique()
    if unique_labels < 2:
        raise MultinomialNBTrainingError(
            "Для обучения MultinomialNB в train-выборке нужны минимум два класса."
        )


def _save_model(
    *,
    model: MultinomialNB,
    vectorization_result: TfidfVectorizationResult,
    project_paths: ProjectPaths,
) -> Path:
    """Сохраняет обученный классификатор в каталог `models/classifiers`."""
    project_paths.ensure_directories()

    model_name = _sanitize_name(
        f"{vectorization_result.paths.feature_directory.name}_multinomial_nb.joblib"
    )
    model_path = _get_available_path(project_paths.classifiers_dir / model_name)
    joblib.dump(model, model_path)
    return model_path


def _save_training_report(
    *,
    report: MultinomialNBTrainingReport,
    config: MultinomialNBConfig,
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
    paths: MultinomialNBPaths,
    project_paths: ProjectPaths,
) -> None:
    """Сохраняет JSON-отчет по обучению модели."""
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


def train_multinomial_nb(
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
    config: MultinomialNBConfig | None = None,
) -> MultinomialNBTrainingResult:
    """Обучает MultinomialNB на TF-IDF-признаках и сохраняет модель."""
    resolved_config = config or MultinomialNBConfig()
    _validate_inputs(split_result, vectorization_result)

    train_labels = split_result.train_dataframe["label"].astype("string")
    model = MultinomialNB(alpha=resolved_config.alpha)
    warning_messages: list[str] = []

    start_time = time.perf_counter()
    try:
        model.fit(vectorization_result.train_matrix, train_labels)
    except ValueError as error:
        raise MultinomialNBTrainingError(
            f"Не удалось обучить MultinomialNB: {error}"
        ) from error
    training_seconds = round(time.perf_counter() - start_time, 4)

    if vectorization_result.train_matrix.shape[1] < 20:
        warning_messages.append(
            "Словарь признаков получился очень маленьким. Для устойчивого сравнения моделей нужен более богатый датасет."
        )

    model_path = _save_model(
        model=model,
        vectorization_result=vectorization_result,
        project_paths=project_paths,
    )
    report_path = _get_available_path(
        project_paths.training_reports_dir
        / f"{Path(model_path).stem}_training_report.json"
    )
    paths = MultinomialNBPaths(
        model_path=model_path,
        report_path=report_path,
    )

    report = MultinomialNBTrainingReport(
        model_name="MultinomialNB",
        class_labels=tuple(str(label) for label in model.classes_),
        training_seconds=training_seconds,
        train_rows=len(split_result.train_dataframe),
        validation_rows=len(split_result.validation_dataframe),
        test_rows=len(split_result.test_dataframe),
        feature_count=int(vectorization_result.train_matrix.shape[1]),
        class_log_prior_shape=tuple(int(value) for value in model.class_log_prior_.shape),
        alpha=resolved_config.alpha,
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

    return MultinomialNBTrainingResult(
        model=model,
        config=resolved_config,
        report=report,
        paths=paths,
    )
