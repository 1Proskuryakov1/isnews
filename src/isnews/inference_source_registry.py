"""Единый реестр доступных источников инференса для интерфейса приложения."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceSourceDescriptor:
    """Описывает один доступный источник модели для инференса."""

    display_name: str
    source_name: str
    source_kind: str
    model: Any
    vectorizer: Any | None = None
    tokenizer: Any | None = None


def collect_inference_sources(
    *,
    training_result: Any = None,
    vectorization_result: Any = None,
    loaded_artifacts_result: Any = None,
    loaded_transformers_result: Any = None,
) -> dict[str, InferenceSourceDescriptor]:
    """Собирает единый набор доступных источников для одиночного и пакетного инференса."""
    available_sources: dict[str, InferenceSourceDescriptor] = {}

    if (
        training_result is not None
        and vectorization_result is not None
        and hasattr(training_result.model, "predict_proba")
        and hasattr(training_result.model, "classes_")
    ):
        available_sources["Модель текущей сессии"] = InferenceSourceDescriptor(
            display_name="Модель текущей сессии",
            source_name=f"session::{training_result.paths.model_path.name}",
            source_kind="sklearn",
            model=training_result.model,
            vectorizer=vectorization_result.vectorizer,
        )

    if loaded_artifacts_result is not None:
        available_sources["Загруженные артефакты"] = InferenceSourceDescriptor(
            display_name="Загруженные артефакты",
            source_name=f"loaded::{loaded_artifacts_result.paths.model_path.name}",
            source_kind="sklearn",
            model=loaded_artifacts_result.model,
            vectorizer=loaded_artifacts_result.vectorizer,
        )

    if loaded_transformers_result is not None:
        available_sources["Загруженная transformers-модель"] = InferenceSourceDescriptor(
            display_name="Загруженная transformers-модель",
            source_name=(
                f"transformers::{loaded_transformers_result.paths.model_directory_path.name}"
            ),
            source_kind="transformers",
            model=loaded_transformers_result.model,
            tokenizer=loaded_transformers_result.tokenizer,
        )

    return available_sources
