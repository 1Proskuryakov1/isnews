"""Экспорт Markdown-отчета по результатам transformers-экспериментов."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class TransformersMarkdownReportExportError(ValueError):
    """Ошибка экспорта Markdown-отчета по transformers-экспериментам."""


@dataclass(frozen=True)
class TransformersMarkdownReportExportResult:
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


def _build_comparison_section(comparison_result: Any) -> str:
    """Строит Markdown-раздел по сравнению transformers-экспериментов."""
    if comparison_result is None:
        return "## Сравнение transformers-экспериментов\n\nДанные сравнения отсутствуют.\n"

    section = "## Сравнение transformers-экспериментов\n\n"
    section += f"- **Лучший запуск:** `{_safe_text(comparison_result.best_source_name or 'нет данных')}`\n"
    section += f"- **CSV-отчет:** `{_safe_text(comparison_result.paths.csv_path)}`\n"
    section += f"- **JSON-отчет:** `{_safe_text(comparison_result.paths.json_path)}`\n\n"
    section += _render_dataframe(comparison_result.dataframe)
    return section + "\n"


def _build_registry_section(registry_result: Any) -> str:
    """Строит Markdown-раздел по реестру transformers-экспериментов."""
    if registry_result is None:
        return "## Реестр transformers-экспериментов\n\nСводный реестр отсутствует.\n"

    section = "## Реестр transformers-экспериментов\n\n"
    section += f"- **CSV-реестр:** `{_safe_text(registry_result.paths.csv_path)}`\n"
    section += f"- **JSON-реестр:** `{_safe_text(registry_result.paths.json_path)}`\n\n"
    section += _render_dataframe(registry_result.dataframe)
    return section + "\n"


def _build_evaluation_section(evaluation_result: Any) -> str:
    """Строит Markdown-раздел по оценке пакетного transformers-инференса."""
    if evaluation_result is None:
        return "## Метрики пакетного transformers-инференса\n\nМетрики отсутствуют.\n"

    report = evaluation_result.report
    rows = [
        ("Источник", report.source_name),
        ("Колонка меток", report.label_column),
        ("Оценено строк", report.evaluated_rows),
        ("Пропущено строк", report.skipped_rows_without_label),
        ("Accuracy", report.accuracy),
        ("Precision macro", report.precision_macro),
        ("Recall macro", report.recall_macro),
        ("F1 macro", report.f1_macro),
        ("CSV матрицы ошибок", evaluation_result.paths.confusion_matrix_path),
        ("JSON-отчет", evaluation_result.paths.report_path),
    ]
    section = "## Метрики пакетного transformers-инференса\n\n"
    section += _render_key_value_list(rows) + "\n"
    section += _render_dataframe(evaluation_result.confusion_matrix_dataframe.reset_index())
    return section + "\n"


def _build_error_analysis_section(error_analysis_result: Any) -> str:
    """Строит Markdown-раздел по ошибкам пакетного transformers-инференса."""
    if error_analysis_result is None:
        return "## Анализ ошибок transformers-модели\n\nАнализ ошибок отсутствует.\n"

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
    section = "## Анализ ошибок transformers-модели\n\n"
    section += _render_key_value_list(rows) + "\n"
    section += _render_dataframe(error_analysis_result.misclassified_dataframe)
    return section + "\n"


def export_transformers_markdown_report(
    *,
    comparison_result: Any = None,
    registry_result: Any = None,
    evaluation_result: Any = None,
    error_analysis_result: Any = None,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> TransformersMarkdownReportExportResult:
    """Экспортирует Markdown-отчет по результатам transformers-экспериментов."""
    if all(
        item is None
        for item in (
            comparison_result,
            registry_result,
            evaluation_result,
            error_analysis_result,
        )
    ):
        raise TransformersMarkdownReportExportError(
            "Для Markdown-отчета по transformers-экспериментам пока нет данных."
        )

    project_paths.ensure_directories()
    report_path = _get_available_path(
        project_paths.markdown_reports_dir / "transformers_session_report.md"
    )

    content = [
        "# Отчет по transformers-экспериментам классификации новостей",
        "",
        f"Сформировано: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "",
        _build_comparison_section(comparison_result),
        _build_registry_section(registry_result),
        _build_evaluation_section(evaluation_result),
        _build_error_analysis_section(error_analysis_result),
    ]
    report_path.write_text("\n".join(content), encoding="utf-8")

    generated_sections = tuple(
        section_name
        for section_name, result in (
            ("comparison", comparison_result),
            ("registry", registry_result),
            ("evaluation", evaluation_result),
            ("error_analysis", error_analysis_result),
        )
        if result is not None
    )
    return TransformersMarkdownReportExportResult(
        report_path=report_path,
        generated_sections=generated_sections,
    )
