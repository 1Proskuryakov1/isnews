"""Анализ уверенности пакетных предсказаний transformers-модели."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.transformers_batch_text_inference import (
    TransformersBatchTextInferenceResult,
)


class TransformersPredictionConfidenceAnalysisError(ValueError):
    """Ошибка анализа уверенности предсказаний."""


@dataclass(frozen=True)
class TransformersPredictionConfidenceAnalysisReport:
    """Содержит сводку по самым уверенным и самым неуверенным предсказаниям."""

    source_name: str
    analyzed_rows: int
    top_n: int
    highest_probability: float
    lowest_probability: float
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class TransformersPredictionConfidenceAnalysisPaths:
    """Хранит пути к CSV и JSON артефактам анализа уверенности."""

    confident_predictions_path: Path
    uncertain_predictions_path: Path
    report_path: Path


@dataclass(frozen=True)
class TransformersPredictionConfidenceAnalysisResult:
    """Возвращает таблицы уверенных и неуверенных предсказаний и пути к файлам."""

    confident_dataframe: pd.DataFrame
    uncertain_dataframe: pd.DataFrame
    report: TransformersPredictionConfidenceAnalysisReport
    paths: TransformersPredictionConfidenceAnalysisPaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла для артефактов анализа."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "transformers_prediction_confidence"


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


def analyze_transformers_prediction_confidence(
    batch_inference_result: TransformersBatchTextInferenceResult,
    *,
    top_n: int = 10,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> TransformersPredictionConfidenceAnalysisResult:
    """Выделяет самые уверенные и самые неуверенные предсказания transformers-модели."""
    if top_n <= 0:
        raise TransformersPredictionConfidenceAnalysisError(
            "Параметр `top_n` должен быть положительным."
        )

    dataframe = batch_inference_result.dataframe.copy()
    if dataframe.empty:
        raise TransformersPredictionConfidenceAnalysisError(
            "Таблица пакетного transformers-инференса пуста."
        )
    if "predicted_probability" not in dataframe.columns:
        raise TransformersPredictionConfidenceAnalysisError(
            "В таблице пакетного transformers-инференса отсутствует колонка `predicted_probability`."
        )

    analyzed_dataframe = dataframe.loc[
        dataframe["predicted_label"].astype("string").fillna("").ne("")
    ].copy()
    analyzed_dataframe["predicted_probability"] = pd.to_numeric(
        analyzed_dataframe["predicted_probability"],
        errors="coerce",
    )
    analyzed_dataframe = analyzed_dataframe.dropna(subset=["predicted_probability"]).copy()

    if analyzed_dataframe.empty:
        raise TransformersPredictionConfidenceAnalysisError(
            "В таблице не найдено корректных строк с вероятностями предсказаний."
        )

    sorted_desc = analyzed_dataframe.sort_values(
        by="predicted_probability",
        ascending=False,
    ).reset_index(drop=True)
    sorted_asc = analyzed_dataframe.sort_values(
        by="predicted_probability",
        ascending=True,
    ).reset_index(drop=True)

    confident_dataframe = sorted_desc.head(top_n).copy()
    uncertain_dataframe = sorted_asc.head(top_n).copy()

    project_paths.ensure_directories()
    base_name = _sanitize_name(batch_inference_result.report.source_name)
    confident_predictions_path = _get_available_path(
        project_paths.confidence_reports_dir
        / f"{base_name}_transformers_top_confident.csv"
    )
    uncertain_predictions_path = _get_available_path(
        project_paths.confidence_reports_dir
        / f"{base_name}_transformers_top_uncertain.csv"
    )
    report_path = _get_available_path(
        project_paths.confidence_reports_dir
        / f"{base_name}_transformers_confidence_report.json"
    )

    confident_dataframe.to_csv(confident_predictions_path, index=False, encoding="utf-8")
    uncertain_dataframe.to_csv(uncertain_predictions_path, index=False, encoding="utf-8")

    warning_messages: list[str] = []
    if len(analyzed_dataframe) < top_n:
        warning_messages.append(
            "Количество предсказаний меньше запрошенного `top_n`, поэтому сохранены все доступные строки."
        )

    report = TransformersPredictionConfidenceAnalysisReport(
        source_name=batch_inference_result.report.source_name,
        analyzed_rows=len(analyzed_dataframe),
        top_n=min(top_n, len(analyzed_dataframe)),
        highest_probability=round(float(sorted_desc.iloc[0]["predicted_probability"]), 6),
        lowest_probability=round(float(sorted_asc.iloc[0]["predicted_probability"]), 6),
        warning_messages=tuple(warning_messages),
    )
    report_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "report": asdict(report),
                "paths": {
                    "confident_predictions_path": str(confident_predictions_path),
                    "uncertain_predictions_path": str(uncertain_predictions_path),
                    "report_path": str(report_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return TransformersPredictionConfidenceAnalysisResult(
        confident_dataframe=confident_dataframe,
        uncertain_dataframe=uncertain_dataframe,
        report=report,
        paths=TransformersPredictionConfidenceAnalysisPaths(
            confident_predictions_path=confident_predictions_path,
            uncertain_predictions_path=uncertain_predictions_path,
            report_path=report_path,
        ),
    )
