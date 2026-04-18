"""Сводный реестр экспериментов по transformers-инференсу и оценке."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class TransformersExperimentRegistryError(ValueError):
    """Ошибка построения сводного реестра transformers-экспериментов."""


@dataclass(frozen=True)
class TransformersExperimentRecord:
    """Хранит одну унифицированную запись о запуске transformers-инференса или оценки."""

    record_type: str
    record_id: str
    generated_at: str
    source_name: str
    report_path: str
    predictions_path: str
    confusion_matrix_path: str
    class_count: int | None
    total_rows: int | None
    predicted_rows: int | None
    evaluated_rows: int | None
    skipped_rows: int | None
    accuracy: float | None
    f1_macro: float | None
    warning_count: int


@dataclass(frozen=True)
class TransformersExperimentRegistryPaths:
    """Хранит пути к сводным файлам реестра."""

    csv_path: Path
    json_path: Path


@dataclass(frozen=True)
class TransformersExperimentRegistryResult:
    """Возвращает таблицу сводки экспериментов и пути к сохраненным файлам."""

    dataframe: pd.DataFrame
    paths: TransformersExperimentRegistryPaths


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


def _load_json_payload(file_path: Path) -> dict[str, Any]:
    """Безопасно загружает JSON-файл и приводит ошибки к понятному виду."""
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TransformersExperimentRegistryError(
            f"Не удалось прочитать JSON-отчет `{file_path}`: {error}"
        ) from error


def _safe_float(value: Any) -> float | None:
    """Преобразует значение к float, если это возможно."""
    if value is None or value == "":
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    """Преобразует значение к int, если это возможно."""
    if value is None or value == "":
        return None
    return int(value)


def _build_inference_records(project_paths: ProjectPaths) -> list[TransformersExperimentRecord]:
    """Собирает записи по пакетному transformers-инференсу."""
    records: list[TransformersExperimentRecord] = []
    for report_path in sorted(
        project_paths.inference_reports_dir.glob("*_transformers_batch_report.json")
    ):
        payload = _load_json_payload(report_path)
        report = payload.get("report", {})
        paths = payload.get("paths", {})
        source_name = str(report.get("source_name", report_path.stem))
        records.append(
            TransformersExperimentRecord(
                record_type="transformers_batch_inference",
                record_id=f"inference::{report_path.stem}",
                generated_at=str(payload.get("generated_at", "")),
                source_name=source_name,
                report_path=str(report_path),
                predictions_path=str(paths.get("predictions_path", "")),
                confusion_matrix_path="",
                class_count=_safe_int(len(report.get("class_labels", []))),
                total_rows=_safe_int(report.get("total_rows")),
                predicted_rows=_safe_int(report.get("predicted_rows")),
                evaluated_rows=None,
                skipped_rows=_safe_int(report.get("skipped_empty_rows")),
                accuracy=None,
                f1_macro=None,
                warning_count=len(report.get("warning_messages", [])),
            )
        )
    return records


def _build_evaluation_records(project_paths: ProjectPaths) -> list[TransformersExperimentRecord]:
    """Собирает записи по оценке пакетного transformers-инференса."""
    records: list[TransformersExperimentRecord] = []
    for report_path in sorted(
        project_paths.inference_reports_dir.glob("*_transformers_batch_evaluation.json")
    ):
        payload = _load_json_payload(report_path)
        report = payload.get("report", {})
        paths = payload.get("paths", {})
        source_name = str(report.get("source_name", report_path.stem))
        records.append(
            TransformersExperimentRecord(
                record_type="transformers_batch_evaluation",
                record_id=f"evaluation::{report_path.stem}",
                generated_at=str(payload.get("generated_at", "")),
                source_name=source_name,
                report_path=str(report_path),
                predictions_path="",
                confusion_matrix_path=str(paths.get("confusion_matrix_path", "")),
                class_count=_safe_int(len(report.get("class_labels", []))),
                total_rows=None,
                predicted_rows=None,
                evaluated_rows=_safe_int(report.get("evaluated_rows")),
                skipped_rows=_safe_int(report.get("skipped_rows_without_label")),
                accuracy=_safe_float(report.get("accuracy")),
                f1_macro=_safe_float(report.get("f1_macro")),
                warning_count=len(report.get("warning_messages", [])),
            )
        )
    return records


def _build_dataframe(records: list[TransformersExperimentRecord]) -> pd.DataFrame:
    """Преобразует список записей в таблицу с единым порядком колонок."""
    dataframe = pd.DataFrame(asdict(record) for record in records)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "record_type",
                "record_id",
                "generated_at",
                "source_name",
                "report_path",
                "predictions_path",
                "confusion_matrix_path",
                "class_count",
                "total_rows",
                "predicted_rows",
                "evaluated_rows",
                "skipped_rows",
                "accuracy",
                "f1_macro",
                "warning_count",
            ]
        )

    return dataframe.sort_values(
        by=["generated_at", "record_type", "record_id"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def export_transformers_experiment_registry(
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> TransformersExperimentRegistryResult:
    """Строит и сохраняет единый CSV/JSON-реестр по найденным transformers-экспериментам."""
    project_paths.ensure_directories()

    records = _build_inference_records(project_paths)
    records.extend(_build_evaluation_records(project_paths))
    dataframe = _build_dataframe(records)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    csv_path = _get_available_path(
        project_paths.experiment_reports_dir / "transformers_experiment_registry.csv"
    )
    json_path = _get_available_path(
        project_paths.experiment_reports_dir / "transformers_experiment_registry.json"
    )
    dataframe.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "record_count": int(len(dataframe)),
                "records": dataframe.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return TransformersExperimentRegistryResult(
        dataframe=dataframe,
        paths=TransformersExperimentRegistryPaths(
            csv_path=csv_path,
            json_path=json_path,
        ),
    )
