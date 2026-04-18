"""Предобработка текстов новостных публикаций перед обучением моделей."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.dataset_summary import DatasetSummary, build_dataset_summary
from src.isnews.dataset_summary import get_usable_dataframe


class TextPreprocessingError(ValueError):
    """Ошибка предобработки текстового датасета."""


@dataclass(frozen=True)
class TextPreprocessingConfig:
    """Хранит параметры базовой очистки текстов."""

    lowercase_text: bool = True
    normalize_whitespace: bool = True
    normalize_punctuation_spacing: bool = True
    remove_duplicate_text_label_pairs: bool = True


@dataclass(frozen=True)
class TextPreprocessingReport:
    """Содержит сведения о том, как изменилась таблица после очистки."""

    rows_before: int
    usable_rows_before: int
    rows_after: int
    removed_invalid_rows: int
    removed_empty_after_cleaning_rows: int
    removed_duplicate_rows: int
    changed_text_rows: int


@dataclass(frozen=True)
class TextPreprocessingResult:
    """Возвращает очищенный датасет и пути сохраненных артефактов."""

    dataframe: pd.DataFrame
    summary: DatasetSummary
    config: TextPreprocessingConfig
    report: TextPreprocessingReport
    saved_path: Path
    report_path: Path


def _sanitize_filename(file_name: str) -> str:
    """Очищает имя файла перед сохранением артефакта на диск."""
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
    """Подбирает свободное имя файла, если файл с таким именем уже существует."""
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


def _clean_text(text: str, config: TextPreprocessingConfig) -> str:
    """Нормализует регистр, пробелы и базовое оформление знаков препинания."""
    cleaned_text = text.replace("\u00a0", " ").replace("\t", " ").replace("\n", " ")

    if config.normalize_whitespace:
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    if config.normalize_punctuation_spacing:
        cleaned_text = re.sub(r"\s+([,.;:!?])", r"\1", cleaned_text)

    cleaned_text = cleaned_text.strip()

    if config.lowercase_text:
        cleaned_text = cleaned_text.lower()

    return cleaned_text


def _clean_label(label: str) -> str:
    """Нормализует пробелы и обрезает края в названиях классов."""
    return re.sub(r"\s+", " ", label).strip()


def _save_preprocessed_dataset(
    dataframe: pd.DataFrame,
    source_dataset_path: Path,
    project_paths: ProjectPaths,
) -> Path:
    """Сохраняет очищенный датасет в каталог `data/processed`."""
    project_paths.ensure_directories()

    file_name = _sanitize_filename(f"{source_dataset_path.stem}_processed.csv")
    saved_path = _get_available_path(project_paths.processed_data_dir / file_name)
    dataframe.to_csv(saved_path, index=False, encoding="utf-8")
    return saved_path


def _save_preprocessing_report(
    *,
    report: TextPreprocessingReport,
    summary: DatasetSummary,
    config: TextPreprocessingConfig,
    source_dataset_path: Path,
    saved_dataset_path: Path,
    project_paths: ProjectPaths,
) -> Path:
    """Сохраняет JSON-отчет по результатам предобработки."""
    project_paths.ensure_directories()

    report_path = _get_available_path(
        project_paths.preprocessing_reports_dir
        / f"{saved_dataset_path.stem}_preprocessing.json"
    )

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_dataset_path": str(source_dataset_path),
        "saved_dataset_path": str(saved_dataset_path),
        "config": asdict(config),
        "report": asdict(report),
        "summary": asdict(summary),
    }

    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def preprocess_dataset(
    dataframe: pd.DataFrame,
    *,
    source_dataset_path: Path,
    project_paths: ProjectPaths = PROJECT_PATHS,
    config: TextPreprocessingConfig | None = None,
) -> TextPreprocessingResult:
    """Очищает датасет, удаляет дубликаты и сохраняет результат в проект."""
    resolved_config = config or TextPreprocessingConfig()

    usable_dataframe = get_usable_dataframe(dataframe)
    processed_dataframe = usable_dataframe.copy()

    if processed_dataframe.empty:
        raise TextPreprocessingError(
            "В датасете нет корректных строк для предобработки."
        )

    original_texts = processed_dataframe["text"].astype("string").fillna("")
    cleaned_texts = original_texts.map(
        lambda text: _clean_text(str(text), resolved_config)
    )
    processed_dataframe["text"] = cleaned_texts
    processed_dataframe["label"] = (
        processed_dataframe["label"].astype("string").fillna("").map(_clean_label)
    )

    changed_text_rows = int((original_texts != cleaned_texts).sum())

    empty_after_cleaning_mask = (
        processed_dataframe["text"].astype("string").fillna("").eq("")
        | processed_dataframe["label"].astype("string").fillna("").eq("")
    )
    removed_empty_after_cleaning_rows = int(empty_after_cleaning_mask.sum())
    if removed_empty_after_cleaning_rows > 0:
        processed_dataframe = processed_dataframe.loc[~empty_after_cleaning_mask].copy()

    removed_duplicate_rows = 0
    if resolved_config.remove_duplicate_text_label_pairs:
        duplicate_mask = processed_dataframe.duplicated(
            subset=["text", "label"],
            keep="first",
        )
        removed_duplicate_rows = int(duplicate_mask.sum())
        if removed_duplicate_rows > 0:
            processed_dataframe = processed_dataframe.loc[~duplicate_mask].copy()

    processed_dataframe = processed_dataframe.reset_index(drop=True)

    if processed_dataframe.empty:
        raise TextPreprocessingError(
            "После очистки и удаления дубликатов не осталось строк для обучения."
        )

    report = TextPreprocessingReport(
        rows_before=len(dataframe),
        usable_rows_before=len(usable_dataframe),
        rows_after=len(processed_dataframe),
        removed_invalid_rows=int(len(dataframe) - len(usable_dataframe)),
        removed_empty_after_cleaning_rows=removed_empty_after_cleaning_rows,
        removed_duplicate_rows=removed_duplicate_rows,
        changed_text_rows=changed_text_rows,
    )

    summary = build_dataset_summary(processed_dataframe)
    saved_path = _save_preprocessed_dataset(
        dataframe=processed_dataframe,
        source_dataset_path=source_dataset_path,
        project_paths=project_paths,
    )
    report_path = _save_preprocessing_report(
        report=report,
        summary=summary,
        config=resolved_config,
        source_dataset_path=source_dataset_path,
        saved_dataset_path=saved_path,
        project_paths=project_paths,
    )

    return TextPreprocessingResult(
        dataframe=processed_dataframe,
        summary=summary,
        config=resolved_config,
        report=report,
        saved_path=saved_path,
        report_path=report_path,
    )
