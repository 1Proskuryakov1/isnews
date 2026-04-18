"""Сравнение нескольких обученных моделей по сохраненным отчетам."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class ModelComparisonError(ValueError):
    """Ошибка построения сводки сравнения моделей."""


@dataclass(frozen=True)
class ModelComparisonRecord:
    """Хранит одну строку сравнения по обученной модели."""

    model_name: str
    model_path: str
    training_report_path: str
    metrics_report_path: str
    vectorizer_path: str
    generated_at: str
    class_count: int
    train_rows: int
    validation_rows: int
    test_rows: int
    training_seconds: float | None
    validation_accuracy: float | None
    validation_f1_macro: float | None
    test_accuracy: float | None
    test_f1_macro: float | None
    warning_count: int


@dataclass(frozen=True)
class ModelComparisonPaths:
    """Хранит пути к сводным файлам сравнения."""

    csv_path: Path
    json_path: Path


@dataclass(frozen=True)
class ModelComparisonResult:
    """Возвращает DataFrame сравнения моделей и пути к сохраненным файлам."""

    dataframe: pd.DataFrame
    best_model_name: str | None
    paths: ModelComparisonPaths


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
        raise ModelComparisonError(
            f"Не удалось прочитать JSON-отчет `{file_path}`: {error}"
        ) from error


def _safe_float(value: Any) -> float | None:
    """Преобразует значение к float, если это возможно."""
    if value is None or value == "":
        return None
    return float(value)


def _find_metrics_report_path(project_paths: ProjectPaths, training_report_path: Path) -> str:
    """Ищет JSON-отчет с метриками для конкретного training report."""
    training_report_variants = {
        str(training_report_path),
        str(training_report_path.resolve()),
    }
    for metrics_report_path in sorted(project_paths.metrics_reports_dir.glob("*.json")):
        payload = _load_json_payload(metrics_report_path)
        payload_training_report_path = str(payload.get("training_report_path", "")).strip()
        if payload_training_report_path in training_report_variants:
            return str(metrics_report_path)
    return ""


def _build_record(project_paths: ProjectPaths, training_report_path: Path) -> ModelComparisonRecord:
    """Собирает запись сравнения по одному обучающему запуску."""
    training_payload = _load_json_payload(training_report_path)
    training_report = training_payload.get("report", {})
    metrics_report_path = _find_metrics_report_path(project_paths, training_report_path)
    metrics_payload = _load_json_payload(Path(metrics_report_path)) if metrics_report_path else {}
    metrics_report = metrics_payload.get("metrics", {})
    validation_metrics = metrics_report.get("validation_metrics", {})
    test_metrics = metrics_report.get("test_metrics", {})

    warning_count = len(training_report.get("warning_messages", []))
    warning_count += len(metrics_report.get("warning_messages", []))

    model_name = str(
        training_report.get("model_name")
        or ("LogisticRegression" if "logreg" in training_report_path.stem.lower() else training_report_path.stem)
    )
    class_labels = training_report.get("class_labels", [])

    return ModelComparisonRecord(
        model_name=model_name,
        model_path=str(training_payload.get("paths", {}).get("model_path", "")),
        training_report_path=str(training_report_path),
        metrics_report_path=metrics_report_path,
        vectorizer_path=str(training_payload.get("source_vectorizer_path", "")),
        generated_at=str(training_payload.get("generated_at", "")),
        class_count=int(len(class_labels)),
        train_rows=int(training_report.get("train_rows", 0)),
        validation_rows=int(training_report.get("validation_rows", 0)),
        test_rows=int(training_report.get("test_rows", 0)),
        training_seconds=_safe_float(training_report.get("training_seconds")),
        validation_accuracy=_safe_float(validation_metrics.get("accuracy")),
        validation_f1_macro=_safe_float(validation_metrics.get("f1_macro")),
        test_accuracy=_safe_float(test_metrics.get("accuracy")),
        test_f1_macro=_safe_float(test_metrics.get("f1_macro")),
        warning_count=warning_count,
    )


def _build_dataframe(records: list[ModelComparisonRecord]) -> pd.DataFrame:
    """Преобразует записи сравнения моделей в упорядоченную таблицу."""
    dataframe = pd.DataFrame(asdict(record) for record in records)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "model_path",
                "training_report_path",
                "metrics_report_path",
                "vectorizer_path",
                "generated_at",
                "class_count",
                "train_rows",
                "validation_rows",
                "test_rows",
                "training_seconds",
                "validation_accuracy",
                "validation_f1_macro",
                "test_accuracy",
                "test_f1_macro",
                "warning_count",
            ]
        )

    return dataframe.sort_values(
        by=["validation_accuracy", "test_accuracy", "training_seconds"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def compare_trained_models(
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> ModelComparisonResult:
    """Строит и сохраняет таблицу сравнения по найденным обученным моделям."""
    project_paths.ensure_directories()

    training_report_paths = sorted(project_paths.training_reports_dir.glob("*.json"))
    records = [_build_record(project_paths, report_path) for report_path in training_report_paths]
    dataframe = _build_dataframe(records)

    best_model_name = None
    if not dataframe.empty:
        best_model_name = str(dataframe.iloc[0]["model_name"])

    csv_path = _get_available_path(
        project_paths.comparison_reports_dir / "trained_model_comparison.csv"
    )
    json_path = _get_available_path(
        project_paths.comparison_reports_dir / "trained_model_comparison.json"
    )
    dataframe.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "model_count": int(len(dataframe)),
                "best_model_name": best_model_name,
                "records": dataframe.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return ModelComparisonResult(
        dataframe=dataframe,
        best_model_name=best_model_name,
        paths=ModelComparisonPaths(
            csv_path=csv_path,
            json_path=json_path,
        ),
    )
