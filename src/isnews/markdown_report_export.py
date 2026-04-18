"""Экспорт Markdown-отчета по результатам текущей сессии."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class MarkdownReportExportError(ValueError):
    """Ошибка экспорта Markdown-отчета."""


@dataclass(frozen=True)
class MarkdownReportExportResult:
    """Возвращает путь к Markdown-отчету и список включенных разделов."""

    report_path: Path
    generated_sections: tuple[str, ...]


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


def _safe_text(value: Any) -> str:
    """Преобразует произвольное значение в безопасную строку."""
    if value is None:
        return "нет данных"
    return str(value).replace("\n", " ").strip()


def _render_dataframe(dataframe: pd.DataFrame, *, max_rows: int = 20) -> str:
    """Преобразует DataFrame в Markdown-таблицу."""
    if dataframe.empty:
        return "Данные отсутствуют.\n"
    preview = dataframe.head(max_rows).copy()
    columns = [str(column) for column in preview.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in preview.itertuples(index=False, name=None):
        rows.append(
            "| " + " | ".join(_safe_text(value).replace("|", "\\|") for value in row) + " |"
        )
    return "\n".join([header, separator, *rows]) + "\n"


def _render_key_value_list(rows: list[tuple[str, Any]]) -> str:
    """Формирует Markdown-список из пар ключ-значение."""
    return "\n".join(
        f"- **{_safe_text(key)}:** `{_safe_text(value)}`"
        for key, value in rows
    ) + "\n"


def _build_training_section(training_result: Any) -> str:
    """Строит Markdown-раздел по обучению модели."""
    if training_result is None:
        return "## Обучение модели\n\nРезультат обучения в текущей сессии отсутствует.\n"

    report = training_result.report
    model_name = getattr(report, "model_name", type(training_result.model).__name__)
    rows = [
        ("Модель", model_name),
        ("Количество классов", len(getattr(report, "class_labels", ()))),
        ("Train строк", getattr(report, "train_rows", "")),
        ("Validation строк", getattr(report, "validation_rows", "")),
        ("Test строк", getattr(report, "test_rows", "")),
        ("Время обучения, сек", getattr(report, "training_seconds", "")),
        ("Файл модели", training_result.paths.model_path),
        ("JSON-отчет обучения", training_result.paths.report_path),
    ]
    return "## Обучение модели\n\n" + _render_key_value_list(rows) + "\n"


def _build_evaluation_section(evaluation_result: Any) -> str:
    """Строит Markdown-раздел по метрикам качества."""
    if evaluation_result is None:
        return "## Метрики качества\n\nМетрики в текущей сессии отсутствуют.\n"

    report = evaluation_result.report
    metrics_dataframe = pd.DataFrame(
        [
            {
                "Выборка": "train",
                "Accuracy": report.train_metrics.accuracy,
                "Precision macro": report.train_metrics.precision_macro,
                "Recall macro": report.train_metrics.recall_macro,
                "F1 macro": report.train_metrics.f1_macro,
            },
            {
                "Выборка": "validation",
                "Accuracy": report.validation_metrics.accuracy,
                "Precision macro": report.validation_metrics.precision_macro,
                "Recall macro": report.validation_metrics.recall_macro,
                "F1 macro": report.validation_metrics.f1_macro,
            },
            {
                "Выборка": "test",
                "Accuracy": report.test_metrics.accuracy,
                "Precision macro": report.test_metrics.precision_macro,
                "Recall macro": report.test_metrics.recall_macro,
                "F1 macro": report.test_metrics.f1_macro,
            },
        ]
    )
    section = "## Метрики качества\n\n"
    section += (
        "В таблице ниже приведены основные метрики качества модели на обучающей, "
        "валидационной и тестовой выборках.\n\n"
    )
    section += _render_dataframe(metrics_dataframe)
    section += f"\n- **JSON-отчет:** `{_safe_text(evaluation_result.paths.report_path)}`\n"
    if report.warning_messages:
        section += "\nПредупреждения:\n"
        section += "\n".join(f"- {message}" for message in report.warning_messages) + "\n"
    return section + "\n"


def _build_comparison_section(comparison_result: Any) -> str:
    """Строит Markdown-раздел по сравнению моделей."""
    if comparison_result is None:
        return "## Сравнение моделей\n\nСравнение моделей в текущей сессии отсутствует.\n"

    section = "## Сравнение моделей\n\n"
    section += (
        "Сводная таблица позволяет сопоставить обученные модели по качеству на "
        "валидационной и тестовой выборках.\n\n"
    )
    section += f"- **Лучшая модель:** `{_safe_text(comparison_result.best_model_name or 'нет данных')}`\n"
    section += f"- **CSV-отчет:** `{_safe_text(comparison_result.paths.csv_path)}`\n"
    section += f"- **JSON-отчет:** `{_safe_text(comparison_result.paths.json_path)}`\n\n"
    section += _render_dataframe(comparison_result.dataframe)
    return section + "\n"


def _build_registry_section(registry_result: Any) -> str:
    """Строит Markdown-раздел по реестру экспериментов."""
    if registry_result is None:
        return "## Реестр экспериментов\n\nСводный реестр в текущей сессии отсутствует.\n"

    section = "## Реестр экспериментов\n\n"
    section += (
        "Реестр фиксирует найденные запуски обучения и пакетной оценки, что удобно "
        "для описания итеративной разработки в ВКР.\n\n"
    )
    section += f"- **CSV-реестр:** `{_safe_text(registry_result.paths.csv_path)}`\n"
    section += f"- **JSON-реестр:** `{_safe_text(registry_result.paths.json_path)}`\n\n"
    section += _render_dataframe(registry_result.dataframe)
    return section + "\n"


def _build_error_analysis_section(error_analysis_result: Any) -> str:
    """Строит Markdown-раздел по ошибкам пакетного инференса."""
    if error_analysis_result is None:
        return "## Анализ ошибок\n\nАнализ ошибок в текущей сессии отсутствует.\n"

    report = error_analysis_result.report
    rows = [
        ("Источник", report.source_name),
        ("Проверено строк", report.analyzed_rows),
        ("Ошибок", report.misclassified_rows),
        ("Корректных строк", report.correct_rows),
        ("Доля ошибок", report.error_rate),
        ("CSV ошибок", error_analysis_result.paths.misclassified_rows_path),
        ("JSON-отчет", error_analysis_result.paths.report_path),
    ]
    section = "## Анализ ошибок\n\n"
    section += (
        "Ниже приведены строки, на которых модель допустила ошибку. Этот фрагмент "
        "можно использовать в разделе тестирования модели.\n\n"
    )
    section += _render_key_value_list(rows) + "\n"
    section += _render_dataframe(error_analysis_result.misclassified_dataframe)
    return section + "\n"


def export_session_markdown_report(
    *,
    training_result: Any = None,
    evaluation_result: Any = None,
    comparison_result: Any = None,
    registry_result: Any = None,
    error_analysis_result: Any = None,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> MarkdownReportExportResult:
    """Экспортирует Markdown-отчет по результатам текущей сессии."""
    if all(
        item is None
        for item in (
            training_result,
            evaluation_result,
            comparison_result,
            registry_result,
            error_analysis_result,
        )
    ):
        raise MarkdownReportExportError(
            "Для Markdown-отчета пока нет данных. Сначала выполните хотя бы один этап обучения или анализа."
        )

    project_paths.ensure_directories()
    report_path = _get_available_path(project_paths.markdown_reports_dir / "session_report.md")

    content = [
        "# Отчет по интеллектуальному сервису классификации новостей",
        "",
        f"Сформировано: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "",
        _build_training_section(training_result),
        _build_evaluation_section(evaluation_result),
        _build_comparison_section(comparison_result),
        _build_registry_section(registry_result),
        _build_error_analysis_section(error_analysis_result),
    ]
    report_path.write_text("\n".join(content), encoding="utf-8")

    generated_sections = tuple(
        section_name
        for section_name, result in (
            ("training", training_result),
            ("evaluation", evaluation_result),
            ("comparison", comparison_result),
            ("registry", registry_result),
            ("error_analysis", error_analysis_result),
        )
        if result is not None
    )
    return MarkdownReportExportResult(
        report_path=report_path,
        generated_sections=generated_sections,
    )
