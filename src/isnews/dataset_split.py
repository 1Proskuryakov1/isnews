"""Разбиение подготовленного датасета на train, validation и test."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.dataset_summary import DatasetSummary, build_dataset_summary


class DatasetSplitError(ValueError):
    """Ошибка разбиения датасета на подвыборки."""


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Параметры разбиения датасета для последующего обучения модели."""

    train_size: float = 0.7
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    stratify_by_label: bool = True


@dataclass(frozen=True)
class DatasetSplitReport:
    """Содержит итоговую информацию о полученных подвыборках."""

    total_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    stratified_split_used: bool
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSplitPaths:
    """Хранит пути ко всем сохраненным артефактам разбиения."""

    directory: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    report_path: Path


@dataclass(frozen=True)
class DatasetSplitResult:
    """Возвращает сплиты датасета, их сводку и пути сохраненных файлов."""

    train_dataframe: pd.DataFrame
    validation_dataframe: pd.DataFrame
    test_dataframe: pd.DataFrame
    train_summary: DatasetSummary
    validation_summary: DatasetSummary
    test_summary: DatasetSummary
    config: DatasetSplitConfig
    report: DatasetSplitReport
    paths: DatasetSplitPaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла или каталога для артефактов."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "dataset_split"


def _get_available_directory(target_directory: Path) -> Path:
    """Подбирает свободное имя каталога для новой версии разбиения."""
    if not target_directory.exists():
        return target_directory

    counter = 1
    while True:
        candidate = target_directory.with_name(f"{target_directory.name}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def _validate_split_config(config: DatasetSplitConfig) -> None:
    """Проверяет корректность долей разбиения и базовых параметров."""
    ratios_sum = config.train_size + config.validation_size + config.test_size
    if not math.isclose(ratios_sum, 1.0, abs_tol=1e-9):
        raise DatasetSplitError(
            "Сумма долей train, validation и test должна быть равна 1.0."
        )

    if min(config.train_size, config.validation_size, config.test_size) <= 0:
        raise DatasetSplitError(
            "Все доли разбиения должны быть положительными числами."
        )


def _normalize_split_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Сбрасывает индекс после разбиения для удобного сохранения и просмотра."""
    return dataframe.reset_index(drop=True)


def _perform_split(
    dataframe: pd.DataFrame,
    config: DatasetSplitConfig,
    *,
    stratified: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Выполняет двухэтапное разбиение с опциональной стратификацией по классу."""
    temp_size = config.validation_size + config.test_size
    labels = dataframe["label"] if stratified else None

    train_dataframe, temporary_dataframe = train_test_split(
        dataframe,
        test_size=temp_size,
        random_state=config.random_state,
        shuffle=True,
        stratify=labels,
    )

    validation_share_in_temp = config.validation_size / temp_size
    temporary_labels = temporary_dataframe["label"] if stratified else None

    validation_dataframe, test_dataframe = train_test_split(
        temporary_dataframe,
        train_size=validation_share_in_temp,
        random_state=config.random_state,
        shuffle=True,
        stratify=temporary_labels,
    )

    return (
        _normalize_split_dataframe(train_dataframe),
        _normalize_split_dataframe(validation_dataframe),
        _normalize_split_dataframe(test_dataframe),
    )


def _build_split_report(
    *,
    total_rows: int,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    stratified_split_used: bool,
    warning_messages: list[str],
) -> DatasetSplitReport:
    """Формирует финальный отчет по разбиению датасета."""
    return DatasetSplitReport(
        total_rows=total_rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        stratified_split_used=stratified_split_used,
        warning_messages=tuple(warning_messages),
    )


def _save_split_files(
    *,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    source_dataset_path: Path,
    project_paths: ProjectPaths,
) -> DatasetSplitPaths:
    """Сохраняет train, validation и test выборки в отдельный каталог."""
    project_paths.ensure_directories()

    split_directory_name = _sanitize_name(f"{source_dataset_path.stem}_split")
    split_directory = _get_available_directory(
        project_paths.split_data_dir / split_directory_name
    )
    split_directory.mkdir(parents=True, exist_ok=True)

    train_path = split_directory / "train.csv"
    validation_path = split_directory / "validation.csv"
    test_path = split_directory / "test.csv"

    train_dataframe.to_csv(train_path, index=False, encoding="utf-8")
    validation_dataframe.to_csv(validation_path, index=False, encoding="utf-8")
    test_dataframe.to_csv(test_path, index=False, encoding="utf-8")

    report_path = project_paths.split_reports_dir / f"{split_directory.name}_report.json"
    return DatasetSplitPaths(
        directory=split_directory,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        report_path=report_path,
    )


def _save_split_report(
    *,
    split_paths: DatasetSplitPaths,
    source_dataset_path: Path,
    config: DatasetSplitConfig,
    report: DatasetSplitReport,
    train_summary: DatasetSummary,
    validation_summary: DatasetSummary,
    test_summary: DatasetSummary,
) -> None:
    """Сохраняет JSON-файл с метаданными разбиения и сводкой по подвыборкам."""
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_dataset_path": str(source_dataset_path),
        "config": asdict(config),
        "report": asdict(report),
        "paths": {
            "directory": str(split_paths.directory),
            "train_path": str(split_paths.train_path),
            "validation_path": str(split_paths.validation_path),
            "test_path": str(split_paths.test_path),
        },
        "summaries": {
            "train": asdict(train_summary),
            "validation": asdict(validation_summary),
            "test": asdict(test_summary),
        },
    }

    split_paths.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def split_dataset(
    dataframe: pd.DataFrame,
    *,
    source_dataset_path: Path,
    project_paths: ProjectPaths = PROJECT_PATHS,
    config: DatasetSplitConfig | None = None,
) -> DatasetSplitResult:
    """Разбивает подготовленный датасет на train, validation и test."""
    resolved_config = config or DatasetSplitConfig()
    _validate_split_config(resolved_config)

    if len(dataframe) < 4:
        raise DatasetSplitError(
            "Для разбиения на train, validation и test требуется минимум 4 строки."
        )

    warning_messages: list[str] = []
    stratified_split_used = False

    if resolved_config.stratify_by_label:
        try:
            train_dataframe, validation_dataframe, test_dataframe = _perform_split(
                dataframe,
                resolved_config,
                stratified=True,
            )
            stratified_split_used = True
        except ValueError as error:
            warning_messages.append(
                "Стратифицированное разбиение не удалось выполнить для текущего "
                "размера датасета или распределения классов. Использовано "
                "обычное случайное разбиение."
            )
            try:
                train_dataframe, validation_dataframe, test_dataframe = _perform_split(
                    dataframe,
                    resolved_config,
                    stratified=False,
                )
            except ValueError as fallback_error:
                raise DatasetSplitError(
                    f"Не удалось разбить датасет на подвыборки: {fallback_error}"
                ) from fallback_error
    else:
        try:
            train_dataframe, validation_dataframe, test_dataframe = _perform_split(
                dataframe,
                resolved_config,
                stratified=False,
            )
        except ValueError as error:
            raise DatasetSplitError(
                f"Не удалось разбить датасет на подвыборки: {error}"
            ) from error

    if min(len(train_dataframe), len(validation_dataframe), len(test_dataframe)) == 0:
        raise DatasetSplitError(
            "Одна из подвыборок оказалась пустой. Увеличьте размер датасета "
            "или скорректируйте доли разбиения."
        )

    train_summary = build_dataset_summary(train_dataframe)
    validation_summary = build_dataset_summary(validation_dataframe)
    test_summary = build_dataset_summary(test_dataframe)

    report = _build_split_report(
        total_rows=len(dataframe),
        train_rows=len(train_dataframe),
        validation_rows=len(validation_dataframe),
        test_rows=len(test_dataframe),
        stratified_split_used=stratified_split_used,
        warning_messages=warning_messages,
    )

    split_paths = _save_split_files(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        test_dataframe=test_dataframe,
        source_dataset_path=source_dataset_path,
        project_paths=project_paths,
    )
    _save_split_report(
        split_paths=split_paths,
        source_dataset_path=source_dataset_path,
        config=resolved_config,
        report=report,
        train_summary=train_summary,
        validation_summary=validation_summary,
        test_summary=test_summary,
    )

    return DatasetSplitResult(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        test_dataframe=test_dataframe,
        train_summary=train_summary,
        validation_summary=validation_summary,
        test_summary=test_summary,
        config=resolved_config,
        report=report,
        paths=split_paths,
    )
