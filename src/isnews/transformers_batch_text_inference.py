"""Пакетный инференс новостных публикаций через загруженную transformers-модель."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.data_loading import TEXT_COLUMN_CANDIDATES
from src.isnews.text_preprocessing import TextPreprocessingConfig, clean_text_value


class TransformersBatchTextInferenceError(ValueError):
    """Ошибка пакетной классификации новостей через transformers-модель."""


@dataclass(frozen=True)
class TransformersBatchTextInferenceReport:
    """Содержит сводку по пакетному инференсу CSV-файла."""

    source_name: str
    text_column: str
    total_rows: int
    predicted_rows: int
    skipped_empty_rows: int
    class_labels: tuple[str, ...]
    output_columns: tuple[str, ...]
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class TransformersBatchTextInferencePaths:
    """Хранит пути к CSV с предсказаниями и JSON-отчету."""

    predictions_path: Path
    report_path: Path


@dataclass(frozen=True)
class TransformersBatchTextInferenceResult:
    """Возвращает таблицу предсказаний и пути к сохраненным артефактам."""

    dataframe: pd.DataFrame
    report: TransformersBatchTextInferenceReport
    paths: TransformersBatchTextInferencePaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла для CSV и JSON-отчета."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "transformers_batch_inference"


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


def _find_text_column(columns: pd.Index) -> str:
    """Находит колонку с текстом по тем же алиасам, что и в основном пайплайне."""
    normalized_mapping = {str(column).strip().casefold(): str(column) for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized_mapping:
            return normalized_mapping[candidate]

    raise TransformersBatchTextInferenceError(
        "В CSV для пакетного инференса не найдена колонка с текстом новости."
    )


def _validate_inputs(*, dataframe: pd.DataFrame, model: Any, tokenizer: Any) -> None:
    """Проверяет входную таблицу и наличие необходимых методов у модели и токенизатора."""
    if dataframe.empty:
        raise TransformersBatchTextInferenceError(
            "CSV-файл для пакетного инференса пуст."
        )
    if not callable(getattr(tokenizer, "__call__", None)):
        raise TransformersBatchTextInferenceError(
            "Переданный токенизатор не поддерживает токенизацию текста."
        )
    if not callable(getattr(model, "__call__", None)):
        raise TransformersBatchTextInferenceError(
            "Переданная модель не поддерживает прямой инференс."
        )
    if not hasattr(model, "config") or not hasattr(model.config, "num_labels"):
        raise TransformersBatchTextInferenceError(
            "У переданной модели отсутствует корректная конфигурация классов."
        )


def _import_torch_dependencies() -> Any:
    """Лениво импортирует torch для вычисления вероятностей классов."""
    try:
        import torch
    except Exception as error:  # pragma: no cover - зависит от окружения
        raise TransformersBatchTextInferenceError(
            "Для пакетного инференса через transformers-модель нужна библиотека `torch`."
        ) from error

    return torch


def _resolve_class_labels(model: Any) -> tuple[str, ...]:
    """Возвращает текстовые метки всех классов из конфигурации модели."""
    id2label = getattr(model.config, "id2label", {}) or {}
    return tuple(
        str(id2label.get(index, id2label.get(str(index), index)))
        for index in range(int(model.config.num_labels))
    )


def _build_probability_column_name(label: str) -> str:
    """Строит безопасное имя колонки вероятности для конкретного класса."""
    return f"probability_{_sanitize_name(label)}"


def _save_batch_inference_report(
    *,
    dataframe: pd.DataFrame,
    report: TransformersBatchTextInferenceReport,
    paths: TransformersBatchTextInferencePaths,
    project_paths: ProjectPaths,
) -> None:
    """Сохраняет JSON-отчет по пакетной классификации."""
    project_paths.ensure_directories()

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "report": asdict(report),
        "preview_rows": dataframe.head(10).to_dict(orient="records"),
        "paths": {
            "predictions_path": str(paths.predictions_path),
            "report_path": str(paths.report_path),
        },
    }

    paths.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def predict_batch_news_with_transformers(
    dataframe: pd.DataFrame,
    *,
    model: Any,
    tokenizer: Any,
    source_name: str,
    project_paths: ProjectPaths = PROJECT_PATHS,
    preprocessing_config: TextPreprocessingConfig | None = None,
    max_length: int = 512,
) -> TransformersBatchTextInferenceResult:
    """Классифицирует набор новостей из таблицы через transformers-модель."""
    _validate_inputs(
        dataframe=dataframe,
        model=model,
        tokenizer=tokenizer,
    )

    if int(max_length) <= 0:
        raise TransformersBatchTextInferenceError(
            "Параметр `max_length` должен быть положительным."
        )

    resolved_preprocessing_config = preprocessing_config or TextPreprocessingConfig()
    text_column = _find_text_column(dataframe.columns)

    result_dataframe = dataframe.copy()
    original_texts = result_dataframe[text_column].astype("string").fillna("")
    cleaned_texts = original_texts.map(
        lambda text: clean_text_value(str(text), resolved_preprocessing_config)
    )
    valid_mask = cleaned_texts.ne("")
    predicted_rows = int(valid_mask.sum())
    skipped_empty_rows = int((~valid_mask).sum())

    if predicted_rows == 0:
        raise TransformersBatchTextInferenceError(
            "После очистки в CSV не осталось непустых текстов для классификации."
        )

    torch = _import_torch_dependencies()
    class_labels = _resolve_class_labels(model)
    valid_indices = result_dataframe.index[valid_mask]
    valid_texts = cleaned_texts.loc[valid_mask].tolist()

    predicted_labels: list[str] = []
    predicted_probabilities: list[float] = []
    probability_rows: list[list[float]] = []
    token_counts: list[int] = []

    try:
        model.eval()
        for text in valid_texts:
            encoded_inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = model(**encoded_inputs)
            probabilities_tensor = torch.softmax(outputs.logits[0], dim=-1)
            probabilities = [round(float(value), 6) for value in probabilities_tensor.tolist()]

            if len(probabilities) != len(class_labels):
                raise TransformersBatchTextInferenceError(
                    "Модель вернула некорректное число вероятностей классов."
                )

            best_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
            predicted_labels.append(class_labels[best_index])
            predicted_probabilities.append(probabilities[best_index])
            probability_rows.append(probabilities)
            token_counts.append(int(encoded_inputs["input_ids"].shape[-1]))
    except TransformersBatchTextInferenceError:
        raise
    except Exception as error:  # pragma: no cover - зависит от внешних объектов модели
        raise TransformersBatchTextInferenceError(
            f"Не удалось выполнить пакетный инференс transformers-модели: {error}"
        ) from error

    result_dataframe["cleaned_text"] = cleaned_texts
    result_dataframe["predicted_label"] = pd.Series(pd.NA, index=result_dataframe.index, dtype="string")
    result_dataframe["predicted_probability"] = pd.NA
    result_dataframe["token_count"] = pd.NA

    result_dataframe.loc[valid_indices, "predicted_label"] = predicted_labels
    result_dataframe.loc[valid_indices, "predicted_probability"] = predicted_probabilities
    result_dataframe.loc[valid_indices, "token_count"] = token_counts

    probability_column_names: list[str] = []
    for class_position, class_label in enumerate(class_labels):
        probability_column_name = _build_probability_column_name(class_label)
        probability_column_names.append(probability_column_name)
        result_dataframe[probability_column_name] = pd.NA
        result_dataframe.loc[valid_indices, probability_column_name] = [
            row[class_position] for row in probability_rows
        ]

    project_paths.ensure_directories()
    predictions_path = _get_available_path(
        project_paths.inference_reports_dir
        / f"{_sanitize_name(source_name)}_transformers_batch_predictions.csv"
    )
    result_dataframe.to_csv(predictions_path, index=False, encoding="utf-8")

    report_path = _get_available_path(
        project_paths.inference_reports_dir
        / f"{_sanitize_name(source_name)}_transformers_batch_report.json"
    )
    warning_messages: list[str] = []
    if skipped_empty_rows > 0:
        warning_messages.append(
            "Часть строк пропущена, потому что после очистки текст оказался пустым."
        )

    report = TransformersBatchTextInferenceReport(
        source_name=source_name,
        text_column=text_column,
        total_rows=len(result_dataframe),
        predicted_rows=predicted_rows,
        skipped_empty_rows=skipped_empty_rows,
        class_labels=class_labels,
        output_columns=tuple(str(column) for column in result_dataframe.columns),
        warning_messages=tuple(warning_messages),
    )
    paths = TransformersBatchTextInferencePaths(
        predictions_path=predictions_path,
        report_path=report_path,
    )
    _save_batch_inference_report(
        dataframe=result_dataframe,
        report=report,
        paths=paths,
        project_paths=project_paths,
    )

    return TransformersBatchTextInferenceResult(
        dataframe=result_dataframe,
        report=report,
        paths=paths,
    )
