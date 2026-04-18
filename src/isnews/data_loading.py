"""Загрузка и базовая валидация датасетов новостей."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
from pandas.errors import EmptyDataError

from src.isnews.config import PROJECT_PATHS, ProjectPaths

TEXT_COLUMN_CANDIDATES = (
    "text",
    "content",
    "article",
    "body",
    "news",
    "title",
    "description",
    "текст",
    "новость",
    "заголовок",
    "описание",
)

LABEL_COLUMN_CANDIDATES = (
    "label",
    "category",
    "class",
    "topic",
    "target",
    "tag",
    "метка",
    "категория",
    "класс",
    "тема",
)


class DatasetValidationError(ValueError):
    """Ошибка валидации структуры или содержимого датасета."""


@dataclass(frozen=True)
class DatasetValidationReport:
    """Содержит информацию о результате проверки загруженного датасета."""

    text_column: str
    label_column: str
    empty_text_rows: int
    empty_label_rows: int
    invalid_rows: int
    usable_rows: int
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class DatasetLoadResult:
    """Хранит результат загрузки датасета и сведения о сохраненной копии."""

    dataframe: pd.DataFrame
    validation_report: DatasetValidationReport
    source_name: str
    saved_path: Path
    row_count: int
    column_count: int


def _sanitize_filename(file_name: str) -> str:
    """Очищает имя файла, чтобы его безопасно сохранить в каталоге проекта."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in file_name.strip()
    )

    if not cleaned_name:
        cleaned_name = "dataset.csv"

    if not cleaned_name.lower().endswith(".csv"):
        cleaned_name = f"{cleaned_name}.csv"

    return cleaned_name


def _get_available_path(target_path: Path) -> Path:
    """Подбирает свободное имя файла, если такой файл уже существует."""
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


def _save_dataset_copy(
    file_bytes: bytes,
    file_name: str,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> Path:
    """Сохраняет копию датасета в каталог `data/raw` внутри проекта."""
    project_paths.ensure_directories()

    target_name = _sanitize_filename(file_name)
    target_path = _get_available_path(project_paths.raw_data_dir / target_name)
    target_path.write_bytes(file_bytes)
    return target_path


def _load_dataframe_from_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Читает CSV-данные из набора байтов через pandas."""
    if not file_bytes.strip():
        raise DatasetValidationError(
            "CSV-файл пуст. Добавьте хотя бы одну строку данных и повторите загрузку."
        )

    try:
        return pd.read_csv(BytesIO(file_bytes), sep=None, engine="python")
    except EmptyDataError as error:
        raise DatasetValidationError(
            "CSV-файл пуст. Добавьте хотя бы одну строку данных и повторите загрузку."
        ) from error
    except UnicodeDecodeError as error:
        raise DatasetValidationError(
            "Не удалось прочитать CSV-файл. Проверьте кодировку и формат файла."
        ) from error
    except Exception as error:
        raise DatasetValidationError(
            f"Не удалось разобрать CSV-файл: {error}"
        ) from error


def _normalize_column_name(column_name: object) -> str:
    """Приводит имя колонки к виду, удобному для сравнения с алиасами."""
    return str(column_name).strip().lower()


def _format_candidates(candidates: Iterable[str]) -> str:
    """Форматирует список допустимых названий колонок для текста ошибки."""
    return ", ".join(f"`{candidate}`" for candidate in candidates)


def _find_required_column(
    columns: Iterable[object],
    candidates: tuple[str, ...],
    logical_name: str,
) -> str:
    """Ищет обязательную колонку по набору допустимых алиасов."""
    normalized_lookup = {
        _normalize_column_name(column_name): str(column_name)
        for column_name in columns
    }

    for candidate in candidates:
        if candidate in normalized_lookup:
            return normalized_lookup[candidate]

    available_columns = ", ".join(f"`{column}`" for column in normalized_lookup.values())
    raise DatasetValidationError(
        f"В датасете не найдена обязательная колонка `{logical_name}`. "
        f"Подходящие названия: {_format_candidates(candidates)}. "
        f"Обнаруженные колонки: {available_columns or 'отсутствуют'}."
    )


def _prepare_required_series(series: pd.Series) -> pd.Series:
    """Нормализует текстовые значения обязательных колонок для последующей проверки."""
    return series.astype("string").fillna("").str.strip()


def _validate_and_normalize_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, DatasetValidationReport]:
    """Проверяет структуру датасета и приводит ключевые колонки к стандартным именам."""
    if dataframe.columns.empty:
        raise DatasetValidationError(
            "CSV-файл не содержит колонок. Проверьте структуру исходных данных."
        )

    if dataframe.empty:
        raise DatasetValidationError(
            "CSV-файл не содержит строк данных. Добавьте записи и повторите загрузку."
        )

    text_column = _find_required_column(
        columns=dataframe.columns,
        candidates=TEXT_COLUMN_CANDIDATES,
        logical_name="text",
    )
    label_column = _find_required_column(
        columns=dataframe.columns,
        candidates=LABEL_COLUMN_CANDIDATES,
        logical_name="label",
    )

    normalized_dataframe = dataframe.copy()
    normalized_dataframe[text_column] = _prepare_required_series(
        normalized_dataframe[text_column]
    )
    normalized_dataframe[label_column] = _prepare_required_series(
        normalized_dataframe[label_column]
    )

    empty_text_mask = normalized_dataframe[text_column].eq("")
    empty_label_mask = normalized_dataframe[label_column].eq("")

    empty_text_rows = int(empty_text_mask.sum())
    empty_label_rows = int(empty_label_mask.sum())
    invalid_rows = int((empty_text_mask | empty_label_mask).sum())
    usable_rows = int(len(normalized_dataframe) - invalid_rows)

    if usable_rows == 0:
        raise DatasetValidationError(
            "В датасете нет ни одной корректной строки: обязательные поля текста "
            "или класса заполнены не полностью."
        )

    warning_messages: list[str] = []
    if empty_text_rows > 0:
        warning_messages.append(
            "В части строк отсутствует текст новости. Такие строки нельзя будет "
            "использовать для обучения без дополнительной очистки."
        )
    if empty_label_rows > 0:
        warning_messages.append(
            "В части строк отсутствует метка класса. Такие строки нельзя будет "
            "использовать для обучения без дополнительной очистки."
        )

    rename_map: dict[str, str] = {}
    if text_column != "text":
        rename_map[text_column] = "text"
    if label_column != "label":
        rename_map[label_column] = "label"

    if rename_map:
        normalized_dataframe = normalized_dataframe.rename(columns=rename_map)

    validation_report = DatasetValidationReport(
        text_column=text_column,
        label_column=label_column,
        empty_text_rows=empty_text_rows,
        empty_label_rows=empty_label_rows,
        invalid_rows=invalid_rows,
        usable_rows=usable_rows,
        warning_messages=tuple(warning_messages),
    )

    return normalized_dataframe, validation_report


def _build_dataset_result(
    file_bytes: bytes,
    source_name: str,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> DatasetLoadResult:
    """Формирует итоговый объект загрузки после чтения и проверки датасета."""
    dataframe = _load_dataframe_from_csv_bytes(file_bytes)
    normalized_dataframe, validation_report = _validate_and_normalize_dataframe(
        dataframe
    )
    saved_path = _save_dataset_copy(file_bytes, source_name, project_paths)

    return DatasetLoadResult(
        dataframe=normalized_dataframe,
        validation_report=validation_report,
        source_name=source_name,
        saved_path=saved_path,
        row_count=len(normalized_dataframe),
        column_count=len(normalized_dataframe.columns),
    )


def load_dataset_from_local_file(
    file_path: str | Path,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> DatasetLoadResult:
    """Загружает датасет из локального CSV-файла и сохраняет копию в проект."""
    source_path = Path(file_path).expanduser().resolve()

    try:
        file_bytes = source_path.read_bytes()
    except FileNotFoundError as error:
        raise DatasetValidationError(
            f"Локальный файл не найден: `{source_path}`."
        ) from error

    return _build_dataset_result(file_bytes, source_path.name, project_paths)


def load_dataset_from_uploaded_bytes(
    file_bytes: bytes,
    source_name: str,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> DatasetLoadResult:
    """Загружает датасет из байтов, полученных через GUI, и сохраняет копию в проект."""
    return _build_dataset_result(file_bytes, source_name, project_paths)


def _extract_filename_from_url(url: str) -> str:
    """Извлекает имя файла из URL, чтобы сохранить скачанный датасет в проект."""
    parsed_url = urlparse(url)
    extracted_name = Path(unquote(parsed_url.path)).name
    return extracted_name or "downloaded_dataset.csv"


def load_dataset_from_url(
    url: str,
    project_paths: ProjectPaths = PROJECT_PATHS,
    file_name: str | None = None,
    timeout: int = 30,
) -> DatasetLoadResult:
    """Скачивает CSV по прямой ссылке, читает его и сохраняет копию в проект."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as error:
        raise DatasetValidationError(
            f"Не удалось скачать датасет по ссылке: {error}"
        ) from error

    source_name = file_name or _extract_filename_from_url(url)
    return _build_dataset_result(response.content, source_name, project_paths)
