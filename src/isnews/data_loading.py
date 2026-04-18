"""Загрузка датасетов новостей из локальных файлов и по прямым ссылкам."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

from src.isnews.config import PROJECT_PATHS, ProjectPaths


@dataclass(frozen=True)
class DatasetLoadResult:
    """Хранит результат загрузки датасета и сведения о сохраненной копии."""

    dataframe: pd.DataFrame
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
    return pd.read_csv(BytesIO(file_bytes))


def load_dataset_from_local_file(
    file_path: str | Path,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> DatasetLoadResult:
    """Загружает датасет из локального CSV-файла и сохраняет копию в проект."""
    source_path = Path(file_path).expanduser().resolve()
    file_bytes = source_path.read_bytes()
    dataframe = _load_dataframe_from_csv_bytes(file_bytes)
    saved_path = _save_dataset_copy(file_bytes, source_path.name, project_paths)

    return DatasetLoadResult(
        dataframe=dataframe,
        source_name=source_path.name,
        saved_path=saved_path,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
    )


def load_dataset_from_uploaded_bytes(
    file_bytes: bytes,
    source_name: str,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> DatasetLoadResult:
    """Загружает датасет из байтов, полученных через GUI, и сохраняет копию в проект."""
    dataframe = _load_dataframe_from_csv_bytes(file_bytes)
    saved_path = _save_dataset_copy(file_bytes, source_name, project_paths)

    return DatasetLoadResult(
        dataframe=dataframe,
        source_name=source_name,
        saved_path=saved_path,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
    )


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
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    source_name = file_name or _extract_filename_from_url(url)
    file_bytes = response.content
    dataframe = _load_dataframe_from_csv_bytes(file_bytes)
    saved_path = _save_dataset_copy(file_bytes, source_name, project_paths)

    return DatasetLoadResult(
        dataframe=dataframe,
        source_name=source_name,
        saved_path=saved_path,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
    )
