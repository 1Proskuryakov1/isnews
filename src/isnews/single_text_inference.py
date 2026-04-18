"""Инференс одной новостной публикации через обученную или загруженную модель."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.text_preprocessing import TextPreprocessingConfig, clean_text_value


class SingleTextInferenceError(ValueError):
    """Ошибка классификации одной новости через сохраненную модель."""


@dataclass(frozen=True)
class ClassProbability:
    """Хранит вероятность принадлежности текста к отдельному классу."""

    label: str
    probability: float


@dataclass(frozen=True)
class SingleTextInferenceReport:
    """Содержит результат классификации и метаданные инференса."""

    source_name: str
    model_type: str
    vectorizer_type: str
    text_length_chars: int
    predicted_label: str
    predicted_probability: float
    class_probabilities: tuple[ClassProbability, ...]
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class SingleTextInferencePaths:
    """Хранит путь к JSON-отчету с результатом классификации."""

    report_path: Path


@dataclass(frozen=True)
class SingleTextInferenceResult:
    """Возвращает очищенный текст, сводку по предсказанию и путь к отчету."""

    input_text: str
    cleaned_text: str
    report: SingleTextInferenceReport
    paths: SingleTextInferencePaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла отчета по инференсу."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "single_text_inference"


def _get_available_path(target_path: Path) -> Path:
    """Подбирает свободный путь к файлу, если такое имя уже занято."""
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
    *,
    input_text: str,
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
) -> None:
    """Проверяет корректность входного текста и наличие обученных артефактов."""
    if not isinstance(model, LogisticRegression):
        raise SingleTextInferenceError(
            "Для инференса ожидается объект модели типа `LogisticRegression`."
        )
    if not isinstance(vectorizer, TfidfVectorizer):
        raise SingleTextInferenceError(
            "Для инференса ожидается объект векторизатора типа `TfidfVectorizer`."
        )
    if not hasattr(model, "classes_") or not hasattr(model, "coef_"):
        raise SingleTextInferenceError(
            "У переданной модели отсутствуют обученные параметры. Сначала обучите или загрузите модель."
        )
    if not hasattr(vectorizer, "vocabulary_"):
        raise SingleTextInferenceError(
            "У переданного векторизатора отсутствует словарь `vocabulary_`."
        )
    if not hasattr(model, "predict_proba"):
        raise SingleTextInferenceError(
            "Переданная модель не поддерживает расчет вероятностей `predict_proba`."
        )
    if not str(input_text).strip():
        raise SingleTextInferenceError(
            "Введите текст новости для классификации."
        )


def _build_probabilities(
    *,
    model: LogisticRegression,
    probabilities: np.ndarray,
) -> tuple[ClassProbability, ...]:
    """Преобразует массив вероятностей в отсортированную сводку по классам."""
    probability_items = [
        ClassProbability(
            label=str(class_label),
            probability=round(float(class_probability), 6),
        )
        for class_label, class_probability in zip(model.classes_, probabilities, strict=True)
    ]
    probability_items.sort(key=lambda item: item.probability, reverse=True)
    return tuple(probability_items)


def _save_inference_report(
    *,
    result: SingleTextInferenceResult,
    project_paths: ProjectPaths,
) -> None:
    """Сохраняет JSON-отчет по результату классификации одной новости."""
    project_paths.ensure_directories()

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_text": result.input_text,
        "cleaned_text": result.cleaned_text,
        "report": {
            **asdict(result.report),
            "class_probabilities": [
                asdict(probability) for probability in result.report.class_probabilities
            ],
        },
        "paths": {
            "report_path": str(result.paths.report_path),
        },
    }

    result.paths.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def predict_single_news(
    input_text: str,
    *,
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    source_name: str,
    project_paths: ProjectPaths = PROJECT_PATHS,
    preprocessing_config: TextPreprocessingConfig | None = None,
) -> SingleTextInferenceResult:
    """Классифицирует одну новостную публикацию и сохраняет JSON-отчет."""
    _validate_inputs(
        input_text=input_text,
        model=model,
        vectorizer=vectorizer,
    )

    resolved_preprocessing_config = preprocessing_config or TextPreprocessingConfig()
    cleaned_text = clean_text_value(str(input_text), resolved_preprocessing_config)
    if not cleaned_text:
        raise SingleTextInferenceError(
            "После очистки текст оказался пустым. Введите более содержательный текст новости."
        )

    try:
        feature_matrix = vectorizer.transform([cleaned_text])
        probabilities = np.asarray(model.predict_proba(feature_matrix)[0], dtype=float)
    except Exception as error:  # pragma: no cover - зависит от внешних объектов модели
        raise SingleTextInferenceError(
            f"Не удалось выполнить классификацию новости: {error}"
        ) from error

    if probabilities.ndim != 1 or probabilities.size != len(model.classes_):
        raise SingleTextInferenceError(
            "Модель вернула некорректный массив вероятностей классов."
        )

    class_probabilities = _build_probabilities(
        model=model,
        probabilities=probabilities,
    )
    best_prediction = class_probabilities[0]
    warning_messages: list[str] = []
    if best_prediction.probability < 0.5:
        warning_messages.append(
            "Вероятность лучшего класса ниже 0.50. Предсказание стоит трактовать осторожно."
        )

    report_path = _get_available_path(
        project_paths.inference_reports_dir
        / f"{_sanitize_name(source_name)}_single_inference.json"
    )
    result = SingleTextInferenceResult(
        input_text=str(input_text),
        cleaned_text=cleaned_text,
        report=SingleTextInferenceReport(
            source_name=source_name,
            model_type=type(model).__name__,
            vectorizer_type=type(vectorizer).__name__,
            text_length_chars=len(cleaned_text),
            predicted_label=best_prediction.label,
            predicted_probability=best_prediction.probability,
            class_probabilities=class_probabilities,
            warning_messages=tuple(warning_messages),
        ),
        paths=SingleTextInferencePaths(report_path=report_path),
    )
    _save_inference_report(
        result=result,
        project_paths=project_paths,
    )
    return result
