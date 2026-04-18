"""Загрузка сохраненной transformers-модели и токенизатора из локальных артефактов."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class TransformersArtifactsLoadingError(ValueError):
    """Ошибка загрузки локальных артефактов нейросетевой модели."""


@dataclass(frozen=True)
class TransformersArtifactsLoadingReport:
    """Содержит сводку по загруженной transformers-модели и токенизатору."""

    model_type: str
    tokenizer_type: str
    config_model_type: str
    base_model_name: str
    num_labels: int
    vocabulary_size: int
    max_position_embeddings: int
    id2label: dict[str, str]
    state_dict_loaded: bool
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class TransformersArtifactsLoadingPaths:
    """Хранит пути к каталогам и отчету по загрузке."""

    model_directory_path: Path
    tokenizer_directory_path: Path
    state_dict_path: Path | None
    report_path: Path


@dataclass(frozen=True)
class TransformersArtifactsLoadingResult:
    """Возвращает загруженные объекты и сводку по их проверке."""

    model: Any
    tokenizer: Any
    report: TransformersArtifactsLoadingReport
    paths: TransformersArtifactsLoadingPaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя файла отчета."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "transformers_loading_report"


def _get_available_path(target_path: Path) -> Path:
    """Подбирает свободный путь к файлу, если такой уже существует."""
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


def _resolve_existing_directory(path_like: str | Path, artifact_label: str) -> Path:
    """Проверяет, что путь к артефакту существует и указывает на каталог."""
    resolved_path = Path(path_like).expanduser().resolve()
    if not resolved_path.exists():
        raise TransformersArtifactsLoadingError(
            f"Каталог `{artifact_label}` не найден: `{resolved_path}`."
        )
    if not resolved_path.is_dir():
        raise TransformersArtifactsLoadingError(
            f"Путь `{resolved_path}` для `{artifact_label}` не является каталогом."
        )
    return resolved_path


def _resolve_existing_file(path_like: str | Path, artifact_label: str) -> Path:
    """Проверяет, что путь к артефакту существует и указывает на файл."""
    resolved_path = Path(path_like).expanduser().resolve()
    if not resolved_path.exists():
        raise TransformersArtifactsLoadingError(
            f"Файл `{artifact_label}` не найден: `{resolved_path}`."
        )
    if not resolved_path.is_file():
        raise TransformersArtifactsLoadingError(
            f"Путь `{resolved_path}` для `{artifact_label}` не является файлом."
        )
    return resolved_path


def _import_transformers_dependencies() -> tuple[Any, Any, Any, Any]:
    """Лениво импортирует зависимости transformers и torch."""
    try:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except Exception as error:  # pragma: no cover - зависит от окружения
        raise TransformersArtifactsLoadingError(
            "Для загрузки нейросетевых артефактов нужны библиотеки `torch` и `transformers`."
        ) from error

    return AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, torch


def _validate_model_directory(model_directory_path: Path) -> None:
    """Проверяет наличие минимального набора файлов save_pretrained-модели."""
    required_files = ("config.json",)
    missing_files = [
        file_name
        for file_name in required_files
        if not (model_directory_path / file_name).exists()
    ]
    if missing_files:
        raise TransformersArtifactsLoadingError(
            "Каталог модели не похож на результат `save_pretrained`: "
            f"не найдены файлы {', '.join(f'`{name}`' for name in missing_files)}."
        )


def _validate_tokenizer_directory(tokenizer_directory_path: Path) -> None:
    """Проверяет наличие минимального набора файлов save_pretrained-токенизатора."""
    if not (tokenizer_directory_path / "tokenizer_config.json").exists():
        raise TransformersArtifactsLoadingError(
            "Каталог токенизатора не похож на результат `save_pretrained`: "
            "не найден файл `tokenizer_config.json`."
        )


def _save_loading_report(
    *,
    report: TransformersArtifactsLoadingReport,
    paths: TransformersArtifactsLoadingPaths,
    project_paths: ProjectPaths,
) -> None:
    """Сохраняет JSON-отчет по загрузке нейросетевых артефактов."""
    project_paths.ensure_directories()

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "report": asdict(report),
        "paths": {
            "model_directory_path": str(paths.model_directory_path),
            "tokenizer_directory_path": str(paths.tokenizer_directory_path),
            "state_dict_path": str(paths.state_dict_path) if paths.state_dict_path else None,
            "report_path": str(paths.report_path),
        },
    }

    paths.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_transformers_artifacts(
    model_directory_path: str | Path,
    tokenizer_directory_path: str | Path,
    *,
    state_dict_path: str | Path | None = None,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> TransformersArtifactsLoadingResult:
    """Загружает локальные артефакты transformers-модели и токенизатора."""
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, torch = (
        _import_transformers_dependencies()
    )

    resolved_model_directory = _resolve_existing_directory(
        model_directory_path,
        "каталог модели",
    )
    resolved_tokenizer_directory = _resolve_existing_directory(
        tokenizer_directory_path,
        "каталог токенизатора",
    )
    resolved_state_dict_path = (
        _resolve_existing_file(state_dict_path, "state dict")
        if state_dict_path is not None
        else None
    )

    _validate_model_directory(resolved_model_directory)
    _validate_tokenizer_directory(resolved_tokenizer_directory)

    try:
        config = AutoConfig.from_pretrained(
            str(resolved_model_directory),
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(resolved_tokenizer_directory),
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(resolved_model_directory),
            config=config,
            local_files_only=True,
        )
    except Exception as error:  # pragma: no cover - зависит от внешних артефактов
        raise TransformersArtifactsLoadingError(
            f"Не удалось загрузить локальные transformers-артефакты: {error}"
        ) from error

    warning_messages: list[str] = []
    if resolved_state_dict_path is not None:
        try:
            loaded_state_dict = torch.load(
                resolved_state_dict_path,
                map_location="cpu",
            )
            incompatible_keys = model.load_state_dict(loaded_state_dict, strict=False)
        except Exception as error:  # pragma: no cover - зависит от файла state dict
            raise TransformersArtifactsLoadingError(
                f"Не удалось загрузить файл весов `{resolved_state_dict_path}`: {error}"
            ) from error

        if incompatible_keys.missing_keys:
            warning_messages.append(
                "После загрузки `state_dict` у модели остались незаполненные веса: "
                f"{len(incompatible_keys.missing_keys)}."
            )
        if incompatible_keys.unexpected_keys:
            warning_messages.append(
                "В `state_dict` найдены лишние веса, не используемые текущей архитектурой: "
                f"{len(incompatible_keys.unexpected_keys)}."
            )

    num_labels = int(getattr(model.config, "num_labels", 0))
    if num_labels < 2:
        raise TransformersArtifactsLoadingError(
            "Загруженная transformers-модель содержит меньше двух классов."
        )

    vocabulary_size = int(len(tokenizer))
    if vocabulary_size <= 0:
        raise TransformersArtifactsLoadingError(
            "У загруженного токенизатора пустой словарь."
        )

    max_position_embeddings = int(
        getattr(model.config, "max_position_embeddings", 0) or 0
    )
    if max_position_embeddings <= 0:
        warning_messages.append(
            "У конфигурации модели отсутствует или не задано `max_position_embeddings`."
        )

    id2label = {
        str(label_id): str(label_name)
        for label_id, label_name in getattr(model.config, "id2label", {}).items()
    }
    if len(id2label) != num_labels:
        warning_messages.append(
            "Словарь `id2label` в конфигурации модели заполнен не полностью."
        )

    report = TransformersArtifactsLoadingReport(
        model_type=type(model).__name__,
        tokenizer_type=type(tokenizer).__name__,
        config_model_type=str(getattr(model.config, "model_type", "")),
        base_model_name=str(getattr(model.config, "_name_or_path", "")),
        num_labels=num_labels,
        vocabulary_size=vocabulary_size,
        max_position_embeddings=max_position_embeddings,
        id2label=id2label,
        state_dict_loaded=resolved_state_dict_path is not None,
        warning_messages=tuple(warning_messages),
    )

    report_name = _sanitize_name(
        f"{resolved_model_directory.name}__{resolved_tokenizer_directory.name}_transformers_loading.json"
    )
    paths = TransformersArtifactsLoadingPaths(
        model_directory_path=resolved_model_directory,
        tokenizer_directory_path=resolved_tokenizer_directory,
        state_dict_path=resolved_state_dict_path,
        report_path=_get_available_path(project_paths.loading_reports_dir / report_name),
    )
    _save_loading_report(
        report=report,
        paths=paths,
        project_paths=project_paths,
    )

    return TransformersArtifactsLoadingResult(
        model=model,
        tokenizer=tokenizer,
        report=report,
        paths=paths,
    )
