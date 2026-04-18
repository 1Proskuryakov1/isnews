"""Сводный реестр экспериментов по обучению и тестированию моделей."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class ExperimentRegistryError(ValueError):
    """Ошибка построения сводного реестра экспериментов."""


@dataclass(frozen=True)
class ExperimentRecord:
    """Хранит одну унифицированную запись о запуске обучения или тестирования."""

    record_type: str
    record_id: str
    generated_at: str
    source_name: str
    model_path: str
    vectorizer_path: str
    training_report_path: str
    metrics_report_path: str
    vectorization_report_path: str
    batch_evaluation_report_path: str
    class_count: int | None
    vocabulary_size: int | None
    train_rows: int | None
    validation_rows: int | None
    test_rows: int | None
    training_seconds: float | None
    validation_accuracy: float | None
    validation_f1_macro: float | None
    test_accuracy: float | None
    test_f1_macro: float | None
    batch_accuracy: float | None
    batch_f1_macro: float | None
    warning_count: int


@dataclass(frozen=True)
class ExperimentRegistryPaths:
    """Хранит пути к сводным файлам реестра экспериментов."""

    csv_path: Path
    json_path: Path


@dataclass(frozen=True)
class ExperimentRegistryResult:
    """Возвращает DataFrame сводки экспериментов и пути к сохраненным файлам."""

    dataframe: pd.DataFrame
    paths: ExperimentRegistryPaths


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
        raise ExperimentRegistryError(
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


def _build_metrics_index(project_paths: ProjectPaths) -> dict[str, dict[str, Any]]:
    """Индексирует отчеты по метрикам по пути training report."""
    metrics_index: dict[str, dict[str, Any]] = {}
    for report_path in sorted(project_paths.metrics_reports_dir.glob("*.json")):
        payload = _load_json_payload(report_path)
        payload["__report_path"] = str(report_path)
        training_report_path = str(payload.get("training_report_path", "")).strip()
        if training_report_path:
            metrics_index[training_report_path] = payload
    return metrics_index


def _build_vectorization_index(project_paths: ProjectPaths) -> dict[str, dict[str, Any]]:
    """Индексирует отчеты по векторизации по пути сохраненного векторизатора."""
    vectorization_index: dict[str, dict[str, Any]] = {}
    for report_path in sorted(project_paths.vectorization_reports_dir.glob("*.json")):
        payload = _load_json_payload(report_path)
        payload["__report_path"] = str(report_path)
        vectorizer_path = str(payload.get("paths", {}).get("vectorizer_path", "")).strip()
        if vectorizer_path:
            vectorization_index[vectorizer_path] = payload
    return vectorization_index


def _build_training_records(
    *,
    project_paths: ProjectPaths,
    metrics_index: dict[str, dict[str, Any]],
    vectorization_index: dict[str, dict[str, Any]],
) -> list[ExperimentRecord]:
    """Собирает записи реестра по обучающим экспериментам."""
    records: list[ExperimentRecord] = []
    for training_report_path in sorted(project_paths.training_reports_dir.glob("*.json")):
        payload = _load_json_payload(training_report_path)
        training_report = payload.get("report", {})
        training_config = payload.get("config", {})
        model_path = str(payload.get("paths", {}).get("model_path", "")).strip()
        vectorizer_path = str(payload.get("source_vectorizer_path", "")).strip()
        metrics_payload = metrics_index.get(str(training_report_path.resolve()))
        if metrics_payload is None:
            metrics_payload = metrics_index.get(str(training_report_path))
        vectorization_payload = vectorization_index.get(vectorizer_path)
        metrics_report = metrics_payload.get("metrics", {}) if metrics_payload else {}
        validation_metrics = metrics_report.get("validation_metrics", {})
        test_metrics = metrics_report.get("test_metrics", {})

        vectorization_report = vectorization_payload.get("report", {}) if vectorization_payload else {}
        warning_count = len(training_report.get("warning_messages", []))
        warning_count += len(metrics_report.get("warning_messages", []))
        warning_count += len(vectorization_report.get("warning_messages", []))

        record_id = Path(model_path).stem if model_path else training_report_path.stem
        records.append(
            ExperimentRecord(
                record_type="training_run",
                record_id=record_id,
                generated_at=str(payload.get("generated_at", "")),
                source_name=record_id,
                model_path=model_path,
                vectorizer_path=vectorizer_path,
                training_report_path=str(training_report_path),
                metrics_report_path=str(metrics_payload.get("__report_path", "")) if metrics_payload else "",
                vectorization_report_path=str(vectorization_payload.get("__report_path", "")) if vectorization_payload else "",
                batch_evaluation_report_path="",
                class_count=_safe_int(len(training_report.get("class_labels", []))),
                vocabulary_size=_safe_int(vectorization_report.get("vocabulary_size")),
                train_rows=_safe_int(training_report.get("train_rows")),
                validation_rows=_safe_int(training_report.get("validation_rows")),
                test_rows=_safe_int(training_report.get("test_rows")),
                training_seconds=_safe_float(training_report.get("training_seconds")),
                validation_accuracy=_safe_float(validation_metrics.get("accuracy")),
                validation_f1_macro=_safe_float(validation_metrics.get("f1_macro")),
                test_accuracy=_safe_float(test_metrics.get("accuracy")),
                test_f1_macro=_safe_float(test_metrics.get("f1_macro")),
                batch_accuracy=None,
                batch_f1_macro=None,
                warning_count=warning_count,
            )
        )
    return records


def _build_batch_evaluation_records(project_paths: ProjectPaths) -> list[ExperimentRecord]:
    """Собирает записи реестра по размеченным пакетным тестам."""
    records: list[ExperimentRecord] = []
    for report_path in sorted(project_paths.inference_reports_dir.glob("*_batch_evaluation.json")):
        payload = _load_json_payload(report_path)
        report = payload.get("report", {})
        source_name = str(report.get("source_name", report_path.stem))
        record_id = f"batch::{report_path.stem}"
        records.append(
            ExperimentRecord(
                record_type="batch_evaluation",
                record_id=record_id,
                generated_at=str(payload.get("generated_at", "")),
                source_name=source_name,
                model_path="",
                vectorizer_path="",
                training_report_path="",
                metrics_report_path="",
                vectorization_report_path="",
                batch_evaluation_report_path=str(report_path),
                class_count=_safe_int(len(report.get("class_labels", []))),
                vocabulary_size=None,
                train_rows=None,
                validation_rows=None,
                test_rows=None,
                training_seconds=None,
                validation_accuracy=None,
                validation_f1_macro=None,
                test_accuracy=None,
                test_f1_macro=None,
                batch_accuracy=_safe_float(report.get("accuracy")),
                batch_f1_macro=_safe_float(report.get("f1_macro")),
                warning_count=len(report.get("warning_messages", [])),
            )
        )
    return records


def _build_dataframe(records: list[ExperimentRecord]) -> pd.DataFrame:
    """Преобразует список записей в таблицу с единым порядком колонок."""
    dataframe = pd.DataFrame(asdict(record) for record in records)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "record_type",
                "record_id",
                "generated_at",
                "source_name",
                "model_path",
                "vectorizer_path",
                "training_report_path",
                "metrics_report_path",
                "vectorization_report_path",
                "batch_evaluation_report_path",
                "class_count",
                "vocabulary_size",
                "train_rows",
                "validation_rows",
                "test_rows",
                "training_seconds",
                "validation_accuracy",
                "validation_f1_macro",
                "test_accuracy",
                "test_f1_macro",
                "batch_accuracy",
                "batch_f1_macro",
                "warning_count",
            ]
        )

    return dataframe.sort_values(
        by=["generated_at", "record_type", "record_id"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def export_experiment_registry(
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> ExperimentRegistryResult:
    """Строит и сохраняет единый CSV/JSON-реестр по найденным экспериментам."""
    project_paths.ensure_directories()

    metrics_index = _build_metrics_index(project_paths)
    vectorization_index = _build_vectorization_index(project_paths)
    records = _build_training_records(
        project_paths=project_paths,
        metrics_index=metrics_index,
        vectorization_index=vectorization_index,
    )
    records.extend(_build_batch_evaluation_records(project_paths))

    dataframe = _build_dataframe(records)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    csv_path = _get_available_path(
        project_paths.experiment_reports_dir / "experiment_registry.csv"
    )
    json_path = _get_available_path(
        project_paths.experiment_reports_dir / "experiment_registry.json"
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

    return ExperimentRegistryResult(
        dataframe=dataframe,
        paths=ExperimentRegistryPaths(
            csv_path=csv_path,
            json_path=json_path,
        ),
    )
