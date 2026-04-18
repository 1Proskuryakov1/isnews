"""Экспорт краткого HTML-отчета по результатам текущей сессии."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class HtmlReportExportError(ValueError):
    """Ошибка экспорта HTML-отчета."""


@dataclass(frozen=True)
class HtmlReportExportResult:
    """Возвращает путь к HTML-отчету и краткую сводку по его наполнению."""

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


def _build_training_section(training_result: Any) -> str:
    """Формирует HTML-блок по последнему обучению модели."""
    if training_result is None:
        return "<section><h2>Обучение модели</h2><p class='empty-block'>Результат обучения в текущей сессии отсутствует.</p></section>"

    report = training_result.report
    model_name = getattr(report, "model_name", type(training_result.model).__name__)
    rows = [
        ("Модель", model_name),
        ("Классов", len(getattr(report, "class_labels", ()))),
        ("Train строк", getattr(report, "train_rows", "")),
        ("Validation строк", getattr(report, "validation_rows", "")),
        ("Test строк", getattr(report, "test_rows", "")),
        ("Время обучения, сек", getattr(report, "training_seconds", "")),
        ("Файл модели", training_result.paths.model_path),
        ("JSON-отчет обучения", training_result.paths.report_path),
    ]
    return (
        "<section><h2>Обучение модели</h2>"
        + _render_key_value_table(rows)
        + "</section>"
    )


def _build_evaluation_section(evaluation_result: Any) -> str:
    """Формирует HTML-блок по основным метрикам качества."""
    if evaluation_result is None:
        return "<section><h2>Метрики качества</h2><p class='empty-block'>Метрики в текущей сессии отсутствуют.</p></section>"

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
    warning_block = ""
    if report.warning_messages:
        warning_block = "<ul>" + "".join(
            f"<li>{_safe_html(message)}</li>" for message in report.warning_messages
        ) + "</ul>"
    return (
        "<section><h2>Метрики качества</h2>"
        + _render_dataframe(metrics_dataframe)
        + f"<p><strong>JSON-отчет:</strong> {_safe_html(evaluation_result.paths.report_path)}</p>"
        + warning_block
        + "</section>"
    )


def _build_registry_section(registry_result: Any) -> str:
    """Формирует HTML-блок по сводному реестру экспериментов."""
    if registry_result is None:
        return "<section><h2>Реестр экспериментов</h2><p class='empty-block'>Сводный реестр в текущей сессии отсутствует.</p></section>"

    return (
        "<section><h2>Реестр экспериментов</h2>"
        + _render_dataframe(registry_result.dataframe)
        + "<p><strong>CSV:</strong> "
        + _safe_html(registry_result.paths.csv_path)
        + "<br><strong>JSON:</strong> "
        + _safe_html(registry_result.paths.json_path)
        + "</p></section>"
    )


def _build_comparison_section(comparison_result: Any) -> str:
    """Формирует HTML-блок по сравнению моделей."""
    if comparison_result is None:
        return "<section><h2>Сравнение моделей</h2><p class='empty-block'>Сравнение моделей в текущей сессии отсутствует.</p></section>"

    best_model_name = comparison_result.best_model_name or "нет данных"
    return (
        "<section><h2>Сравнение моделей</h2>"
        + f"<p><strong>Лучшая модель:</strong> {_safe_html(best_model_name)}</p>"
        + _render_dataframe(comparison_result.dataframe)
        + "<p><strong>CSV:</strong> "
        + _safe_html(comparison_result.paths.csv_path)
        + "<br><strong>JSON:</strong> "
        + _safe_html(comparison_result.paths.json_path)
        + "</p></section>"
    )


def _build_error_analysis_section(error_analysis_result: Any) -> str:
    """Формирует HTML-блок по ошибкам пакетного инференса."""
    if error_analysis_result is None:
        return "<section><h2>Анализ ошибок</h2><p class='empty-block'>Анализ ошибок в текущей сессии отсутствует.</p></section>"

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
        "<section><h2>Анализ ошибок</h2>"
        + _render_key_value_table(rows)
        + _render_dataframe(error_analysis_result.misclassified_dataframe)
        + "</section>"
    )


def _build_html_document(
    *,
    title: str,
    sections: list[str],
) -> str:
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
      background: #f4f1ea;
      color: #1f2933;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      background: #fffdf8;
      border-radius: 20px;
      padding: 32px;
      box-shadow: 0 18px 40px rgba(31, 41, 51, 0.12);
    }}
    h1, h2 {{
      font-family: Georgia, "Times New Roman", serif;
      color: #1c3d5a;
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


def export_session_html_report(
    *,
    training_result: Any = None,
    evaluation_result: Any = None,
    comparison_result: Any = None,
    registry_result: Any = None,
    error_analysis_result: Any = None,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> HtmlReportExportResult:
    """Экспортирует краткий HTML-отчет по результатам текущей сессии."""
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
        raise HtmlReportExportError(
            "Для HTML-отчета пока нет данных. Сначала выполните хотя бы один этап обучения или анализа."
        )

    project_paths.ensure_directories()
    report_path = _get_available_path(project_paths.html_reports_dir / "session_report.html")

    sections = [
        _build_training_section(training_result),
        _build_evaluation_section(evaluation_result),
        _build_comparison_section(comparison_result),
        _build_registry_section(registry_result),
        _build_error_analysis_section(error_analysis_result),
    ]
    report_path.write_text(
        _build_html_document(
            title="Краткий отчет по интеллектуальному сервису классификации новостей",
            sections=sections,
        ),
        encoding="utf-8",
    )

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
    return HtmlReportExportResult(
        report_path=report_path,
        generated_sections=generated_sections,
    )
