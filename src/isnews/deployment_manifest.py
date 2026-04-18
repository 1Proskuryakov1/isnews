"""Подготовка манифеста развёртывания демонстрационного сервиса."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class DeploymentManifestError(ValueError):
    """Ошибка подготовки манифеста развёртывания проекта."""


@dataclass(frozen=True)
class DeploymentArtifactInfo:
    """Описывает один файл или каталог, который нужен для развёртывания сервиса."""

    artifact_name: str
    relative_path: str
    exists: bool
    artifact_type: str


@dataclass(frozen=True)
class DeploymentManifestReport:
    """Хранит итоговую сводку по готовности проекта к развёртыванию."""

    generated_at: str
    deployment_target: str
    required_artifacts_ready: bool
    detected_model_count: int
    detected_vectorizer_count: int
    detected_transformers_model_count: int
    missing_artifacts: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class DeploymentManifestResult:
    """Возвращает путь к JSON-манифесту и собранную сводку по deployment."""

    report: DeploymentManifestReport
    artifacts: tuple[DeploymentArtifactInfo, ...]
    manifest_path: Path


def _get_available_path(target_path: Path) -> Path:
    """Подбирает свободный путь к манифесту, если файл уже существует."""
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


def _collect_transformers_directories(models_dir: Path) -> list[Path]:
    """Ищет каталоги с сохранёнными transformers-артефактами."""
    candidate_directories: list[Path] = []
    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "config.json").exists() and (
            (child / "tokenizer_config.json").exists()
            or (child / "special_tokens_map.json").exists()
        ):
            candidate_directories.append(child)
    return sorted(candidate_directories)


def _build_artifacts(project_paths: ProjectPaths) -> tuple[DeploymentArtifactInfo, ...]:
    """Собирает список обязательных и вспомогательных deployment-артефактов."""
    root = project_paths.root
    artifacts = [
        DeploymentArtifactInfo(
            artifact_name="main_entrypoint",
            relative_path="main.py",
            exists=(root / "main.py").exists(),
            artifact_type="file",
        ),
        DeploymentArtifactInfo(
            artifact_name="requirements",
            relative_path="requirements.txt",
            exists=(root / "requirements.txt").exists(),
            artifact_type="file",
        ),
        DeploymentArtifactInfo(
            artifact_name="source_package",
            relative_path="src/isnews",
            exists=(project_paths.src_dir / "isnews").exists(),
            artifact_type="directory",
        ),
        DeploymentArtifactInfo(
            artifact_name="models_directory",
            relative_path="models",
            exists=project_paths.models_dir.exists(),
            artifact_type="directory",
        ),
        DeploymentArtifactInfo(
            artifact_name="reports_directory",
            relative_path="reports",
            exists=project_paths.reports_dir.exists(),
            artifact_type="directory",
        ),
    ]
    return tuple(artifacts)


def build_deployment_manifest(
    *,
    deployment_target: str = "streamlit_local_or_cloud",
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> DeploymentManifestResult:
    """Строит JSON-манифест развёртывания для проверки готовности проекта."""
    project_paths.ensure_directories()

    artifacts = _build_artifacts(project_paths)
    missing_artifacts = tuple(
        artifact.artifact_name for artifact in artifacts if not artifact.exists
    )

    classifier_files = sorted(project_paths.classifiers_dir.glob("*.joblib"))
    vectorizer_files = sorted(project_paths.vectorizers_dir.glob("*.joblib"))
    transformers_directories = _collect_transformers_directories(project_paths.models_dir)

    notes: list[str] = []
    if not classifier_files and not transformers_directories:
        notes.append(
            "В репозитории пока не найдено сохранённых моделей для демонстрации после развёртывания."
        )
    if classifier_files and not vectorizer_files:
        notes.append(
            "Найдены sklearn-модели, но не найден ни один TF-IDF-векторизатор."
        )
    if not missing_artifacts:
        notes.append(
            "Базовые файлы запуска присутствуют: main.py, requirements.txt и пакет src/isnews."
        )

    report = DeploymentManifestReport(
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        deployment_target=deployment_target,
        required_artifacts_ready=len(missing_artifacts) == 0,
        detected_model_count=len(classifier_files),
        detected_vectorizer_count=len(vectorizer_files),
        detected_transformers_model_count=len(transformers_directories),
        missing_artifacts=missing_artifacts,
        notes=tuple(notes),
    )

    manifest_path = _get_available_path(
        project_paths.deployment_reports_dir / "deployment_manifest.json"
    )
    manifest_payload = {
        "report": asdict(report),
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "detected_paths": {
            "classifier_files": [
                str(path.relative_to(project_paths.root)) for path in classifier_files
            ],
            "vectorizer_files": [
                str(path.relative_to(project_paths.root)) for path in vectorizer_files
            ],
            "transformers_directories": [
                str(path.relative_to(project_paths.root))
                for path in transformers_directories
            ],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return DeploymentManifestResult(
        report=report,
        artifacts=artifacts,
        manifest_path=manifest_path,
    )
