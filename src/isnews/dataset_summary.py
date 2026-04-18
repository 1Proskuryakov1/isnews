"""Формирование сводки по загруженному датасету и сохранение метаданных."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


@dataclass(frozen=True)
class NumericColumnStats:
    """Хранит базовые числовые статистики по одному признаку."""

    minimum: int
    mean: float
    median: float
    maximum: int


@dataclass(frozen=True)
class ClassDistributionItem:
    """Описывает количество объектов определенного класса."""

    label: str
    count: int
    share: float


@dataclass(frozen=True)
class DatasetSummary:
    """Содержит агрегированную сводку по корректным строкам датасета."""

    usable_rows: int
    unique_classes: int
    class_distribution: tuple[ClassDistributionItem, ...]
    text_length_chars: NumericColumnStats
    text_length_words: NumericColumnStats


def _build_numeric_stats(series: pd.Series) -> NumericColumnStats:
    """Рассчитывает минимум, среднее, медиану и максимум по числовому ряду."""
    integer_series = series.astype(int)
    return NumericColumnStats(
        minimum=int(integer_series.min()),
        mean=round(float(integer_series.mean()), 2),
        median=round(float(integer_series.median()), 2),
        maximum=int(integer_series.max()),
    )


def get_usable_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только строки, пригодные для последующего обучения модели."""
    text_mask = dataframe["text"].astype("string").fillna("").str.strip().ne("")
    label_mask = dataframe["label"].astype("string").fillna("").str.strip().ne("")
    return dataframe.loc[text_mask & label_mask].copy()


def build_dataset_summary(dataframe: pd.DataFrame) -> DatasetSummary:
    """Формирует статистическую сводку по валидной части датасета."""
    usable_dataframe = get_usable_dataframe(dataframe)

    class_counts = usable_dataframe["label"].astype("string").value_counts()
    class_distribution = tuple(
        ClassDistributionItem(
            label=str(label),
            count=int(count),
            share=round(float(count / len(usable_dataframe)), 4),
        )
        for label, count in class_counts.items()
    )

    text_lengths_chars = usable_dataframe["text"].astype("string").str.len()
    text_lengths_words = (
        usable_dataframe["text"].astype("string").str.split().str.len()
    )

    return DatasetSummary(
        usable_rows=len(usable_dataframe),
        unique_classes=int(class_counts.shape[0]),
        class_distribution=class_distribution,
        text_length_chars=_build_numeric_stats(text_lengths_chars),
        text_length_words=_build_numeric_stats(text_lengths_words),
    )


def save_dataset_summary(
    *,
    summary: DatasetSummary,
    source_name: str,
    saved_dataset_path: Path,
    row_count: int,
    column_count: int,
    validation_payload: dict[str, object],
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> Path:
    """Сохраняет JSON-файл со сводкой и метаданными датасета."""
    project_paths.ensure_directories()

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_name": source_name,
        "saved_dataset_path": str(saved_dataset_path),
        "row_count": row_count,
        "column_count": column_count,
        "validation": validation_payload,
        "summary": asdict(summary),
    }

    summary_path = project_paths.dataset_reports_dir / f"{saved_dataset_path.stem}_summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path
