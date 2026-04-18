"""Пакетный инференс новостных публикаций из CSV-таблицы."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.data_loading import TEXT_COLUMN_CANDIDATES
from src.isnews.text_preprocessing import TextPreprocessingConfig, clean_text_value


class BatchTextInferenceError(ValueError):
    """Ошибка пакетной классификации новостных публикаций."""


@dataclass(frozen=True)
class BatchTextInferenceReport:
    """Содержит сводку по пакетной классификации CSV-файла."""

    source_name: str
    text_column: str
    total_rows: int
    predicted_rows: int
    skipped_empty_rows: int
    class_labels: tuple[str, ...]
    output_columns: tuple[str, ...]
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class BatchTextInferencePaths:
    """Хранит пути к сохраненным результатам пакетного инференса."""

    predictions_path: Path
    report_path: Path


@dataclass(frozen=True)
class BatchTextInferenceResult:
    """Возвращает таблицу предсказаний и пути к сохраненным артефактам."""

    dataframe: pd.DataFrame
    report: BatchTextInferenceReport
    paths: BatchTextInferencePaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла для CSV и JSON-отчета."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "batch_inference"


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
    """Находит колонку с текстом по тем же алиасам, что и при загрузке датасета."""
    normalized_mapping = {str(column).strip().casefold(): str(column) for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized_mapping:
            return normalized_mapping[candidate]

    raise BatchTextInferenceError(
        "В CSV для пакетного инференса не найдена колонка с текстом новости. "
        "Поддерживаются алиасы вроде `text`, `content`, `title`, `description`, `текст`."
    )


def _validate_inputs(
    *,
    dataframe: pd.DataFrame,
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
) -> None:
    """Проверяет входную таблицу и наличие обученных артефактов."""
    if dataframe.empty:
        raise BatchTextInferenceError(
            "CSV-файл для пакетного инференса пуст."
        )
    if not isinstance(model, LogisticRegression):
        raise BatchTextInferenceError(
            "Для пакетного инференса ожидается модель типа `LogisticRegression`."
        )
    if not isinstance(vectorizer, TfidfVectorizer):
        raise BatchTextInferenceError(
            "Для пакетного инференса ожидается векторизатор типа `TfidfVectorizer`."
        )
    if not hasattr(model, "classes_") or not hasattr(model, "predict_proba"):
        raise BatchTextInferenceError(
            "Переданная модель не содержит обученных параметров для инференса."
        )
    if not hasattr(vectorizer, "vocabulary_"):
        raise BatchTextInferenceError(
            "Переданный векторизатор не содержит словаря признаков."
        )


def _build_probability_column_name(label: str) -> str:
    """Строит безопасное имя колонки вероятности для конкретного класса."""
    return f"probability_{_sanitize_name(label)}"


def _save_batch_inference_report(
    *,
    dataframe: pd.DataFrame,
    report: BatchTextInferenceReport,
    paths: BatchTextInferencePaths,
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


def predict_batch_news(
    dataframe: pd.DataFrame,
    *,
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
    source_name: str,
    project_paths: ProjectPaths = PROJECT_PATHS,
    preprocessing_config: TextPreprocessingConfig | None = None,
) -> BatchTextInferenceResult:
    """Классифицирует набор новостей из таблицы и сохраняет результаты в CSV."""
    _validate_inputs(
        dataframe=dataframe,
        model=model,
        vectorizer=vectorizer,
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
        raise BatchTextInferenceError(
            "После очистки в CSV не осталось непустых текстов для классификации."
        )

    try:
        feature_matrix = vectorizer.transform(cleaned_texts.loc[valid_mask].tolist())
        probabilities = np.asarray(model.predict_proba(feature_matrix), dtype=float)
    except Exception as error:  # pragma: no cover - зависит от внешних объектов модели
        raise BatchTextInferenceError(
            f"Не удалось выполнить пакетную классификацию: {error}"
        ) from error

    if probabilities.ndim != 2 or probabilities.shape[1] != len(model.classes_):
        raise BatchTextInferenceError(
            "Модель вернула некорректную матрицу вероятностей для пакетного инференса."
        )

    predicted_indices = probabilities.argmax(axis=1)
    predicted_labels = [str(model.classes_[index]) for index in predicted_indices]
    predicted_probabilities = [
        round(float(probabilities[row_index, class_index]), 6)
        for row_index, class_index in enumerate(predicted_indices)
    ]

    result_dataframe["cleaned_text"] = cleaned_texts
    result_dataframe["predicted_label"] = pd.Series(pd.NA, index=result_dataframe.index, dtype="string")
    result_dataframe["predicted_probability"] = np.nan

    valid_indices = result_dataframe.index[valid_mask]
    result_dataframe.loc[valid_indices, "predicted_label"] = predicted_labels
    result_dataframe.loc[valid_indices, "predicted_probability"] = predicted_probabilities

    probability_column_names: list[str] = []
    for class_position, class_label in enumerate(model.classes_):
        probability_column_name = _build_probability_column_name(str(class_label))
        probability_column_names.append(probability_column_name)
        result_dataframe[probability_column_name] = np.nan
        result_dataframe.loc[valid_indices, probability_column_name] = [
            round(float(value), 6) for value in probabilities[:, class_position]
        ]

    project_paths.ensure_directories()
    predictions_path = _get_available_path(
        project_paths.inference_reports_dir
        / f"{_sanitize_name(source_name)}_batch_predictions.csv"
    )
    result_dataframe.to_csv(predictions_path, index=False, encoding="utf-8")

    report_path = _get_available_path(
        project_paths.inference_reports_dir
        / f"{_sanitize_name(source_name)}_batch_report.json"
    )
    warning_messages: list[str] = []
    if skipped_empty_rows > 0:
        warning_messages.append(
            "Часть строк пропущена, потому что после очистки текст оказался пустым."
        )

    report = BatchTextInferenceReport(
        source_name=source_name,
        text_column=text_column,
        total_rows=len(result_dataframe),
        predicted_rows=predicted_rows,
        skipped_empty_rows=skipped_empty_rows,
        class_labels=tuple(str(label) for label in model.classes_),
        output_columns=tuple(str(column) for column in result_dataframe.columns),
        warning_messages=tuple(warning_messages),
    )
    paths = BatchTextInferencePaths(
        predictions_path=predictions_path,
        report_path=report_path,
    )
    _save_batch_inference_report(
        dataframe=result_dataframe,
        report=report,
        paths=paths,
        project_paths=project_paths,
    )

    return BatchTextInferenceResult(
        dataframe=result_dataframe,
        report=report,
        paths=paths,
    )
