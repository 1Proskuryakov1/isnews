"""Экспорт человекочитаемой инструкции по развёртыванию сервиса."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.deployment_manifest import (
    DeploymentManifestResult,
    build_deployment_manifest,
)


class DeploymentGuideExportError(ValueError):
    """Ошибка формирования Markdown-инструкции по deployment."""


@dataclass(frozen=True)
class DeploymentGuideExportResult:
    """Возвращает путь к Markdown-инструкции и связанный deployment-манифест."""

    guide_path: Path
    manifest_result: DeploymentManifestResult


def _get_available_path(target_path: Path) -> Path:
    """Подбирает свободное имя файла, если инструкция уже существует."""
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


def _build_artifacts_section(manifest_result: DeploymentManifestResult) -> str:
    """Строит Markdown-блок по найденным и отсутствующим deployment-артефактам."""
    lines = ["## Проверка артефактов", ""]
    for artifact in manifest_result.artifacts:
        status = "найден" if artifact.exists else "отсутствует"
        lines.append(
            f"- `{artifact.relative_path}`: {status} ({artifact.artifact_type})."
        )
    if manifest_result.report.missing_artifacts:
        lines.extend(
            [
                "",
                "Отсутствующие обязательные элементы:",
                f"- `{', '.join(manifest_result.report.missing_artifacts)}`.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _build_launch_steps(
    deployment_target: str,
    manifest_result: DeploymentManifestResult,
) -> str:
    """Строит пошаговую инструкцию запуска сервиса."""
    model_note = (
        "В проекте найдены сохранённые модели, их можно использовать без переобучения."
        if (
            manifest_result.report.detected_model_count > 0
            or manifest_result.report.detected_transformers_model_count > 0
        )
        else "Сохранённые модели не обнаружены, перед демонстрацией потребуется обучить и сохранить хотя бы одну модель."
    )

    lines = [
        "## Шаги развёртывания",
        "",
        "1. Создать и активировать чистое виртуальное окружение Python 3.11+.",
        "2. Установить зависимости командой `pip install -r requirements.txt`.",
        "3. Убедиться, что в репозитории присутствуют каталоги `models`, `reports` и `data`.",
        "4. Запустить приложение командой `python main.py`.",
        "5. Открыть локальный адрес Streamlit в браузере и проверить загрузку интерфейса.",
        f"6. Проверить сценарий deployment `{deployment_target}` на целевой машине или в облаке.",
        "",
        model_note,
        "",
    ]
    return "\n".join(lines)


def _build_notes_section(manifest_result: DeploymentManifestResult) -> str:
    """Строит Markdown-блок с замечаниями по готовности к deployment."""
    lines = ["## Замечания", ""]
    if manifest_result.report.notes:
        lines.extend(f"- {note}" for note in manifest_result.report.notes)
    else:
        lines.append("- Дополнительных замечаний нет.")
    lines.append("")
    return "\n".join(lines)


def export_deployment_guide(
    *,
    deployment_target: str = "streamlit_local_or_cloud",
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> DeploymentGuideExportResult:
    """Формирует Markdown-инструкцию по развёртыванию проекта."""
    project_paths.ensure_directories()
    manifest_result = build_deployment_manifest(
        deployment_target=deployment_target,
        project_paths=project_paths,
    )
    guide_path = _get_available_path(
        project_paths.deployment_reports_dir / "deployment_guide.md"
    )

    content = [
        "# Инструкция по развёртыванию isnews",
        "",
        f"- Целевой сценарий: `{manifest_result.report.deployment_target}`.",
        f"- JSON-манифест: `{manifest_result.manifest_path}`.",
        "",
        _build_artifacts_section(manifest_result),
        _build_launch_steps(deployment_target, manifest_result),
        _build_notes_section(manifest_result),
    ]
    guide_path.write_text("\n".join(content), encoding="utf-8")

    return DeploymentGuideExportResult(
        guide_path=guide_path,
        manifest_result=manifest_result,
    )
