"""Экспорт HTML-отчета по результатам transformers-экспериментов."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class TransformersHtmlReportExportError(ValueError):
    """Ошибка экспорта HTML-отчета по transformers-экспериментам."""


@dataclass(frozen=True)
class TransformersHtmlReportExportResult:
    """Возвращает путь к HTML-отчету и список включенных разделов."""

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


def _safe_html(value: Any) -> str:
    """Экранирует произвольное значение для HTML."""
    if value is None:
        return "нет данных"
    return escape(str(value))


def _render_key_value_table(rows: list[tuple[str, Any]]) -> str:
    """Строит HTML-таблицу из пар ключ-значение."""
    body = "\n".join(
        f"<tr><th>{_safe_html(key)}</th><td>{_safe_html(value)}</td></tr>"
        for key, value in rows
    )
    return f"<table class='kv-table'>{body}</table>"


def _render_dataframe(dataframe: pd.DataFrame, *, max_rows: int = 20) -> str:
    """Безопасно преобразует DataFrame в HTML-представление."""
    if dataframe.empty:
        return "<p class='empty-block'>Данные отсутствуют.</p>"
    preview = dataframe.head(max_rows).copy()
    return preview.to_html(index=False, escape=True, classes="data-table", border=0)


def _build_comparison_section(comparison_result: Any) -> str:
    """Формирует HTML-блок по сравнению transformers-экспериментов."""
    if comparison_result is None:
        return "<section><h2>Сравнение transformers-экспериментов</h2><p class='empty-block'>Данные сравнения отсутствуют.</p></section>"

    best_source_name = comparison_result.best_source_name or "нет данных"
    return (
        "<section><h2>Сравнение transformers-экспериментов</h2>"
        + f"<p><strong>Лучший запуск:</strong> {_safe_html(best_source_name)}</p>"
        + _render_dataframe(comparison_result.dataframe)
        + "<p><strong>CSV:</strong> "
        + _safe_html(comparison_result.paths.csv_path)
        + "<br><strong>JSON:</strong> "
        + _safe_html(comparison_result.paths.json_path)
        + "</p></section>"
    )


def _build_registry_section(registry_result: Any) -> str:
    """Формирует HTML-блок по реестру transformers-экспериментов."""
    if registry_result is None:
        return "<section><h2>Реестр transformers-экспериментов</h2><p class='empty-block'>Сводный реестр отсутствует.</p></section>"

    return (
        "<section><h2>Реестр transformers-экспериментов</h2>"
        + _render_dataframe(registry_result.dataframe)
        + "<p><strong>CSV:</strong> "
        + _safe_html(registry_result.paths.csv_path)
        + "<br><strong>JSON:</strong> "
        + _safe_html(registry_result.paths.json_path)
        + "</p></section>"
    )


def _build_evaluation_section(evaluation_result: Any) -> str:
    """Формирует HTML-блок по метрикам пакетного transformers-инференса."""
    if evaluation_result is None:
        return "<section><h2>Метрики пакетного transformers-инференса</h2><p class='empty-block'>Метрики отсутствуют.</p></section>"

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
    return (
        "<section><h2>Метрики пакетного transformers-инференса</h2>"
        + _render_key_value_table(rows)
        + _render_dataframe(evaluation_result.confusion_matrix_dataframe.reset_index())
        + "</section>"
    )


def _build_error_analysis_section(error_analysis_result: Any) -> str:
    """Формирует HTML-блок по ошибкам пакетного transformers-инференса."""
    if error_analysis_result is None:
        return "<section><h2>Анализ ошибок transformers-модели</h2><p class='empty-block'>Анализ ошибок отсутствует.</p></section>"

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
    return (
        "<section><h2>Анализ ошибок transformers-модели</h2>"
        + _render_key_value_table(rows)
        + _render_dataframe(error_analysis_result.misclassified_dataframe)
        + "</section>"
    )


def _build_html_document(*, title: str, sections: list[str]) -> str:
    """Собирает полный HTML-документ."""
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{
      font-family: "Segoe UI", Tahoma, sans-serif;
      margin: 0;
      padding: 32px;
      background: #f5f0e8;
      color: #1f2933;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      background: #fffdf9;
      border-radius: 20px;
      padding: 32px;
      box-shadow: 0 18px 40px rgba(31, 41, 51, 0.12);
    }}
    h1, h2 {{
      font-family: Georgia, "Times New Roman", serif;
      color: #17405d;
    }}
    section {{
      margin-top: 28px;
      padding-top: 16px;
      border-top: 1px solid #d8d2c5;
    }}
    .meta {{
      color: #52606d;
      margin-bottom: 24px;
    }}
    .kv-table, .data-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }}
    .kv-table th, .kv-table td, .data-table th, .data-table td {{
      border: 1px solid #d8d2c5;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    .kv-table th, .data-table th {{
      background: #f1ece2;
      width: 28%;
    }}
    .empty-block {{
      padding: 14px 16px;
      background: #f6f7f9;
      border-radius: 10px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <p class="meta">Сформировано: {escape(generated_at)}</p>
    {''.join(sections)}
  </main>
</body>
</html>"""


def export_transformers_html_report(
    *,
    comparison_result: Any = None,
    registry_result: Any = None,
    evaluation_result: Any = None,
    error_analysis_result: Any = None,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> TransformersHtmlReportExportResult:
    """Экспортирует HTML-отчет по результатам transformers-экспериментов."""
    if all(
        item is None
        for item in (
            comparison_result,
            registry_result,
            evaluation_result,
            error_analysis_result,
        )
    ):
        raise TransformersHtmlReportExportError(
            "Для HTML-отчета по transformers-экспериментам пока нет данных."
        )

    project_paths.ensure_directories()
    report_path = _get_available_path(
        project_paths.html_reports_dir / "transformers_session_report.html"
    )

    sections = [
        _build_comparison_section(comparison_result),
        _build_registry_section(registry_result),
        _build_evaluation_section(evaluation_result),
        _build_error_analysis_section(error_analysis_result),
    ]
    report_path.write_text(
        _build_html_document(
            title="Отчет по transformers-экспериментам классификации новостей",
            sections=sections,
        ),
        encoding="utf-8",
    )

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
    return TransformersHtmlReportExportResult(
        report_path=report_path,
        generated_sections=generated_sections,
    )
