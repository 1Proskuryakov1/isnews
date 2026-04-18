"""Одиночный инференс через загруженную transformers-модель классификации."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.text_preprocessing import TextPreprocessingConfig, clean_text_value


class TransformersSingleTextInferenceError(ValueError):
    """Ошибка классификации одной новости через transformers-модель."""


@dataclass(frozen=True)
class TransformersClassProbability:
    """Хранит вероятность принадлежности текста к одному классу."""

    label: str
    probability: float


@dataclass(frozen=True)
class TransformersSingleTextInferenceReport:
    """Содержит сводку по результату одиночного инференса."""

    source_name: str
    model_type: str
    tokenizer_type: str
    text_length_chars: int
    token_count: int
    predicted_label: str
    predicted_probability: float
    class_probabilities: tuple[TransformersClassProbability, ...]
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class TransformersSingleTextInferencePaths:
    """Хранит путь к JSON-отчету по одиночному инференсу."""

    report_path: Path


@dataclass(frozen=True)
class TransformersSingleTextInferenceResult:
    """Возвращает очищенный текст, отчет и путь к сохраненному JSON."""

    input_text: str
    cleaned_text: str
    report: TransformersSingleTextInferenceReport
    paths: TransformersSingleTextInferencePaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла отчета."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "transformers_single_inference"


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


def _validate_inputs(*, input_text: str, model: Any, tokenizer: Any) -> None:
    """Проверяет входной текст и наличие необходимых методов у модели и токенизатора."""
    if not str(input_text).strip():
        raise TransformersSingleTextInferenceError(
            "Введите текст новости для классификации."
        )
    if not callable(getattr(tokenizer, "__call__", None)):
        raise TransformersSingleTextInferenceError(
            "Переданный токенизатор не поддерживает токенизацию текста."
        )
    if not callable(getattr(model, "__call__", None)):
        raise TransformersSingleTextInferenceError(
            "Переданная модель не поддерживает прямой инференс."
        )
    if not hasattr(model, "config") or not hasattr(model.config, "num_labels"):
        raise TransformersSingleTextInferenceError(
            "У переданной модели отсутствует корректная конфигурация классов."
        )


def _import_torch_dependencies() -> Any:
    """Лениво импортирует torch для вычисления softmax."""
    try:
        import torch
    except Exception as error:  # pragma: no cover - зависит от окружения
        raise TransformersSingleTextInferenceError(
            "Для инференса через transformers-модель нужна библиотека `torch`."
        ) from error

    return torch


def _resolve_label(model: Any, label_index: int) -> str:
    """Возвращает текстовую метку класса по индексу."""
    id2label = getattr(model.config, "id2label", {}) or {}
    resolved_label = id2label.get(label_index)
    if resolved_label is None:
        resolved_label = id2label.get(str(label_index))
    return str(resolved_label) if resolved_label is not None else str(label_index)


def _save_inference_report(
    *,
    result: TransformersSingleTextInferenceResult,
    project_paths: ProjectPaths,
) -> None:
    """Сохраняет JSON-отчет по результату одиночного инференса."""
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


def predict_single_news_with_transformers(
    input_text: str,
    *,
    model: Any,
    tokenizer: Any,
    source_name: str,
    project_paths: ProjectPaths = PROJECT_PATHS,
    preprocessing_config: TextPreprocessingConfig | None = None,
    max_length: int = 512,
) -> TransformersSingleTextInferenceResult:
    """Классифицирует одну новость через transformers-модель и сохраняет JSON-отчет."""
    _validate_inputs(
        input_text=input_text,
        model=model,
        tokenizer=tokenizer,
    )

    if int(max_length) <= 0:
        raise TransformersSingleTextInferenceError(
            "Параметр `max_length` должен быть положительным."
        )

    resolved_preprocessing_config = preprocessing_config or TextPreprocessingConfig()
    cleaned_text = clean_text_value(str(input_text), resolved_preprocessing_config)
    if not cleaned_text:
        raise TransformersSingleTextInferenceError(
            "После очистки текст оказался пустым. Введите более содержательный текст новости."
        )

    torch = _import_torch_dependencies()

    try:
        model.eval()
        encoded_inputs = tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        logits = outputs.logits[0]
        probabilities_tensor = torch.softmax(logits, dim=-1)
    except Exception as error:  # pragma: no cover - зависит от внешних объектов модели
        raise TransformersSingleTextInferenceError(
            f"Не удалось выполнить инференс transformers-модели: {error}"
        ) from error

    probabilities = [round(float(value), 6) for value in probabilities_tensor.tolist()]
    if len(probabilities) != int(getattr(model.config, "num_labels", 0)):
        raise TransformersSingleTextInferenceError(
            "Модель вернула некорректное число вероятностей классов."
        )

    class_probabilities = tuple(
        sorted(
            [
                TransformersClassProbability(
                    label=_resolve_label(model, label_index),
                    probability=probability,
                )
                for label_index, probability in enumerate(probabilities)
            ],
            key=lambda item: item.probability,
            reverse=True,
        )
    )
    best_prediction = class_probabilities[0]

    warning_messages: list[str] = []
    if best_prediction.probability < 0.5:
        warning_messages.append(
            "Вероятность лучшего класса ниже 0.50. Предсказание стоит трактовать осторожно."
        )

    report_path = _get_available_path(
        project_paths.inference_reports_dir
        / f"{_sanitize_name(source_name)}_transformers_single_inference.json"
    )
    result = TransformersSingleTextInferenceResult(
        input_text=str(input_text),
        cleaned_text=cleaned_text,
        report=TransformersSingleTextInferenceReport(
            source_name=source_name,
            model_type=type(model).__name__,
            tokenizer_type=type(tokenizer).__name__,
            text_length_chars=len(cleaned_text),
            token_count=int(encoded_inputs["input_ids"].shape[-1]),
            predicted_label=best_prediction.label,
            predicted_probability=best_prediction.probability,
            class_probabilities=class_probabilities,
            warning_messages=tuple(warning_messages),
        ),
        paths=TransformersSingleTextInferencePaths(report_path=report_path),
    )
    _save_inference_report(
        result=result,
        project_paths=project_paths,
    )
    return result
