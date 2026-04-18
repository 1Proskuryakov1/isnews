"""Сравнение нескольких transformers-экспериментов по сохраненным отчетам."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class TransformersModelComparisonError(ValueError):
    """Ошибка построения сводки сравнения transformers-экспериментов."""


@dataclass(frozen=True)
class TransformersModelComparisonRecord:
    """Хранит одну строку сравнения по transformers-эксперименту."""

    source_name: str
    generated_at: str
    inference_report_path: str
    evaluation_report_path: str
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
class TransformersModelComparisonPaths:
    """Хранит пути к сводным файлам сравнения."""

    csv_path: Path
    json_path: Path


@dataclass(frozen=True)
class TransformersModelComparisonResult:
    """Возвращает таблицу сравнения экспериментов и пути к сохраненным файлам."""

    dataframe: pd.DataFrame
    best_source_name: str | None
    paths: TransformersModelComparisonPaths


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
    """Безопасно загружает JSON-отчет."""
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TransformersModelComparisonError(
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


def _build_evaluation_index(project_paths: ProjectPaths) -> dict[str, dict[str, Any]]:
    """Индексирует отчеты оценки по имени источника."""
    evaluation_index: dict[str, dict[str, Any]] = {}
    for report_path in sorted(
        project_paths.inference_reports_dir.glob("*_transformers_batch_evaluation.json")
    ):
        payload = _load_json_payload(report_path)
        payload["__report_path"] = str(report_path)
        source_name = str(payload.get("report", {}).get("source_name", "")).strip()
        if source_name:
            evaluation_index[source_name] = payload
    return evaluation_index


def _build_record(
    inference_report_path: Path,
    evaluation_index: dict[str, dict[str, Any]],
) -> TransformersModelComparisonRecord:
    """Собирает запись сравнения по одному transformers-запуску."""
    inference_payload = _load_json_payload(inference_report_path)
    inference_report = inference_payload.get("report", {})
    inference_paths = inference_payload.get("paths", {})
    source_name = str(inference_report.get("source_name", inference_report_path.stem))

    evaluation_payload = evaluation_index.get(source_name, {})
    evaluation_report = evaluation_payload.get("report", {})
    evaluation_paths = evaluation_payload.get("paths", {})

    warning_count = len(inference_report.get("warning_messages", []))
    warning_count += len(evaluation_report.get("warning_messages", []))

    return TransformersModelComparisonRecord(
        source_name=source_name,
        generated_at=str(inference_payload.get("generated_at", "")),
        inference_report_path=str(inference_report_path),
        evaluation_report_path=str(evaluation_payload.get("__report_path", "")),
        predictions_path=str(inference_paths.get("predictions_path", "")),
        confusion_matrix_path=str(evaluation_paths.get("confusion_matrix_path", "")),
        class_count=_safe_int(len(inference_report.get("class_labels", []))),
        total_rows=_safe_int(inference_report.get("total_rows")),
        predicted_rows=_safe_int(inference_report.get("predicted_rows")),
        evaluated_rows=_safe_int(evaluation_report.get("evaluated_rows")),
        skipped_rows=_safe_int(
            evaluation_report.get("skipped_rows_without_label", inference_report.get("skipped_empty_rows"))
        ),
        accuracy=_safe_float(evaluation_report.get("accuracy")),
        f1_macro=_safe_float(evaluation_report.get("f1_macro")),
        warning_count=warning_count,
    )


def _build_dataframe(records: list[TransformersModelComparisonRecord]) -> pd.DataFrame:
    """Преобразует записи сравнения в упорядоченную таблицу."""
    dataframe = pd.DataFrame(asdict(record) for record in records)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "source_name",
                "generated_at",
                "inference_report_path",
                "evaluation_report_path",
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
        by=["accuracy", "f1_macro", "predicted_rows", "warning_count"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def compare_transformers_runs(
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> TransformersModelComparisonResult:
    """Строит и сохраняет таблицу сравнения по найденным transformers-экспериментам."""
    project_paths.ensure_directories()

    evaluation_index = _build_evaluation_index(project_paths)
    inference_report_paths = sorted(
        project_paths.inference_reports_dir.glob("*_transformers_batch_report.json")
    )
    records = [
        _build_record(report_path, evaluation_index)
        for report_path in inference_report_paths
    ]
    dataframe = _build_dataframe(records)

    best_source_name = None
    if not dataframe.empty:
        best_source_name = str(dataframe.iloc[0]["source_name"])

    csv_path = _get_available_path(
        project_paths.comparison_reports_dir / "transformers_model_comparison.csv"
    )
    json_path = _get_available_path(
        project_paths.comparison_reports_dir / "transformers_model_comparison.json"
    )
    dataframe.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "record_count": int(len(dataframe)),
                "best_source_name": best_source_name,
                "records": dataframe.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return TransformersModelComparisonResult(
        dataframe=dataframe,
        best_source_name=best_source_name,
        paths=TransformersModelComparisonPaths(
            csv_path=csv_path,
            json_path=json_path,
        ),
    )
