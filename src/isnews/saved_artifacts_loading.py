"""Загрузка сохраненных артефактов модели и TF-IDF-векторизатора из файлов."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class SavedArtifactsLoadingError(ValueError):
    """Ошибка загрузки или проверки сохраненных артефактов модели."""


@dataclass(frozen=True)
class SavedArtifactsLoadingReport:
    """Содержит сведения о загруженных артефактах и результате их проверки."""

    model_type: str
    vectorizer_type: str
    class_labels: tuple[str, ...]
    class_count: int
    vocabulary_size: int
    feature_count: int
    coefficient_shape: tuple[int, int]
    intercept_shape: tuple[int, ...]
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class SavedArtifactsLoadingPaths:
    """Хранит пути к загруженным файлам и JSON-отчету о проверке."""

    model_path: Path
    vectorizer_path: Path
    report_path: Path


@dataclass(frozen=True)
class SavedArtifactsLoadingResult:
    """Возвращает загруженные артефакты и связанную сводку по валидации."""

    model: LogisticRegression
    vectorizer: TfidfVectorizer
    report: SavedArtifactsLoadingReport
    paths: SavedArtifactsLoadingPaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла для отчета о загрузке."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "saved_artifacts_report"


def _get_available_path(target_path: Path) -> Path:
    """Подбирает свободный путь к файлу, если такой уже существует."""
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


def _resolve_existing_file(path_like: str | Path, artifact_label: str) -> Path:
    """Проверяет, что путь к артефакту существует и указывает на файл."""
    resolved_path = Path(path_like).expanduser().resolve()
    if not resolved_path.exists():
        raise SavedArtifactsLoadingError(
            f"Файл `{artifact_label}` не найден: `{resolved_path}`."
        )
    if not resolved_path.is_file():
        raise SavedArtifactsLoadingError(
            f"Путь `{resolved_path}` для `{artifact_label}` не является файлом."
        )
    return resolved_path


def _load_joblib_object(file_path: Path, artifact_label: str) -> object:
    """Загружает joblib-объект из файла и преобразует ошибки к понятному виду."""
    try:
        return joblib.load(file_path)
    except Exception as error:  # pragma: no cover - зависит от поврежденного файла
        raise SavedArtifactsLoadingError(
            f"Не удалось загрузить `{artifact_label}` из файла `{file_path}`: {error}"
        ) from error


def _validate_vectorizer(vectorizer: object) -> tuple[TfidfVectorizer, int]:
    """Проверяет корректность загруженного TF-IDF-векторизатора."""
    if not isinstance(vectorizer, TfidfVectorizer):
        raise SavedArtifactsLoadingError(
            "Загруженный объект векторизатора не является `TfidfVectorizer`."
        )
    if not hasattr(vectorizer, "vocabulary_"):
        raise SavedArtifactsLoadingError(
            "У загруженного векторизатора отсутствует словарь `vocabulary_`."
        )

    vocabulary_size = len(vectorizer.vocabulary_)
    if vocabulary_size <= 0:
        raise SavedArtifactsLoadingError(
            "Словарь загруженного векторизатора пуст. Такой артефакт нельзя использовать."
        )

    idf_values = getattr(vectorizer, "idf_", None)
    if idf_values is None:
        raise SavedArtifactsLoadingError(
            "У загруженного векторизатора отсутствует массив `idf_`."
        )
    if int(np.asarray(idf_values).shape[0]) != vocabulary_size:
        raise SavedArtifactsLoadingError(
            "Размер `idf_` у загруженного векторизатора не совпадает с размером словаря."
        )

    return vectorizer, vocabulary_size


def _validate_model(model: object) -> tuple[LogisticRegression, tuple[str, ...]]:
    """Проверяет корректность загруженного классификатора Logistic Regression."""
    if not isinstance(model, LogisticRegression):
        raise SavedArtifactsLoadingError(
            "Загруженный объект модели не является `LogisticRegression`."
        )
    for attribute_name in ("classes_", "coef_", "intercept_"):
        if not hasattr(model, attribute_name):
            raise SavedArtifactsLoadingError(
                f"У загруженной модели отсутствует атрибут `{attribute_name}`."
            )

    class_labels = tuple(str(label) for label in model.classes_)
    if len(class_labels) < 2:
        raise SavedArtifactsLoadingError(
            "Загруженная модель содержит меньше двух классов и непригодна для классификации."
        )

    coefficient_shape = np.asarray(model.coef_).shape
    if len(coefficient_shape) != 2 or int(coefficient_shape[1]) <= 0:
        raise SavedArtifactsLoadingError(
            "Матрица коэффициентов загруженной модели имеет некорректную форму."
        )

    intercept_shape = np.asarray(model.intercept_).shape
    if len(intercept_shape) == 0:
        raise SavedArtifactsLoadingError(
            "Вектор смещений `intercept_` у загруженной модели имеет некорректную форму."
        )

    return model, class_labels


def _save_loading_report(
    *,
    report: SavedArtifactsLoadingReport,
    paths: SavedArtifactsLoadingPaths,
    project_paths: ProjectPaths,
) -> None:
    """Сохраняет JSON-отчет по загрузке и проверке артефактов."""
    project_paths.ensure_directories()

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "report": asdict(report),
        "paths": {
            "model_path": str(paths.model_path),
            "vectorizer_path": str(paths.vectorizer_path),
            "report_path": str(paths.report_path),
        },
    }

    paths.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_saved_artifacts(
    model_path: str | Path,
    vectorizer_path: str | Path,
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> SavedArtifactsLoadingResult:
    """Загружает классификатор и векторизатор из файлов и проверяет их совместимость."""
    resolved_model_path = _resolve_existing_file(model_path, "модель")
    resolved_vectorizer_path = _resolve_existing_file(vectorizer_path, "векторизатор")

    loaded_model = _load_joblib_object(resolved_model_path, "модель")
    loaded_vectorizer = _load_joblib_object(resolved_vectorizer_path, "векторизатор")

    model, class_labels = _validate_model(loaded_model)
    vectorizer, vocabulary_size = _validate_vectorizer(loaded_vectorizer)

    coefficient_shape = tuple(int(value) for value in np.asarray(model.coef_).shape)
    intercept_shape = tuple(int(value) for value in np.asarray(model.intercept_).shape)
    feature_count = int(coefficient_shape[1])

    if feature_count != vocabulary_size:
        raise SavedArtifactsLoadingError(
            "Загруженные модель и векторизатор несовместимы: число признаков в модели "
            f"`{feature_count}`, размер словаря векторизатора `{vocabulary_size}`."
        )

    model_features_in = getattr(model, "n_features_in_", None)
    if model_features_in is not None and int(model_features_in) != feature_count:
        raise SavedArtifactsLoadingError(
            "Атрибут `n_features_in_` у модели не совпадает с размером матрицы коэффициентов."
        )

    warning_messages: list[str] = []
    if len(class_labels) == 2 and coefficient_shape[0] == 1:
        warning_messages.append(
            "Для бинарной классификации Logistic Regression хранит одну строку `coef_`; это штатное поведение."
        )

    report = SavedArtifactsLoadingReport(
        model_type=type(model).__name__,
        vectorizer_type=type(vectorizer).__name__,
        class_labels=class_labels,
        class_count=len(class_labels),
        vocabulary_size=vocabulary_size,
        feature_count=feature_count,
        coefficient_shape=coefficient_shape,
        intercept_shape=intercept_shape,
        warning_messages=tuple(warning_messages),
    )

    report_file_name = _sanitize_name(
        f"{resolved_model_path.stem}__{resolved_vectorizer_path.stem}_loading_report.json"
    )
    report_path = _get_available_path(project_paths.loading_reports_dir / report_file_name)
    paths = SavedArtifactsLoadingPaths(
        model_path=resolved_model_path,
        vectorizer_path=resolved_vectorizer_path,
        report_path=report_path,
    )
    _save_loading_report(
        report=report,
        paths=paths,
        project_paths=project_paths,
    )

    return SavedArtifactsLoadingResult(
        model=model,
        vectorizer=vectorizer,
        report=report,
        paths=paths,
    )
