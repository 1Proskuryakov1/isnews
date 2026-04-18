"""Экспорт тепловых карт матриц ошибок в PNG."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.isnews.config import PROJECT_PATHS, ProjectPaths


class ConfusionHeatmapExportError(ValueError):
    """Ошибка экспорта тепловых карт матриц ошибок."""


@dataclass(frozen=True)
class ConfusionHeatmapExportPaths:
    """Хранит пути к PNG-файлам и manifest-файлу."""

    validation_heatmap_path: Path | None
    test_heatmap_path: Path | None
    batch_heatmap_path: Path | None
    manifest_path: Path


@dataclass(frozen=True)
class ConfusionHeatmapExportResult:
    """Возвращает список созданных тепловых карт и пути к ним."""

    exported_heatmap_names: tuple[str, ...]
    paths: ConfusionHeatmapExportPaths


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


def _save_heatmap(
    dataframe: pd.DataFrame,
    *,
    title: str,
    target_path: Path,
) -> None:
    """Сохраняет одну тепловую карту матрицы ошибок."""
    figure, axis = plt.subplots(figsize=(7, 5.5))
    image = axis.imshow(dataframe.to_numpy(), cmap="Blues", aspect="auto")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(dataframe.columns)))
    axis.set_yticks(range(len(dataframe.index)))
    axis.set_xticklabels([str(label) for label in dataframe.columns], rotation=45, ha="right")
    axis.set_yticklabels([str(label) for label in dataframe.index])
    for row_index in range(len(dataframe.index)):
        for column_index in range(len(dataframe.columns)):
            axis.text(
                column_index,
                row_index,
                str(dataframe.iat[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )
    axis.set_title(title)
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    figure.tight_layout()
    figure.savefig(target_path, dpi=180)
    plt.close(figure)


def export_confusion_heatmaps(
    *,
    detailed_evaluation_result: Any = None,
    batch_evaluation_result: Any = None,
    project_paths: ProjectPaths = PROJECT_PATHS,
) -> ConfusionHeatmapExportResult:
    """Экспортирует тепловые карты матриц ошибок по доступным результатам оценки."""
    if detailed_evaluation_result is None and batch_evaluation_result is None:
        raise ConfusionHeatmapExportError(
            "Для экспорта тепловых карт пока нет данных. Сначала выполните подробную оценку модели или пакетную оценку на размеченном CSV."
        )

    project_paths.ensure_directories()

    exported_heatmap_names: list[str] = []
    validation_heatmap_path = None
    test_heatmap_path = None
    batch_heatmap_path = None

    if detailed_evaluation_result is not None:
        validation_dataframe = pd.DataFrame(
            detailed_evaluation_result.report.validation.confusion_matrix,
            index=detailed_evaluation_result.report.validation.class_labels,
            columns=detailed_evaluation_result.report.validation.class_labels,
        )
        test_dataframe = pd.DataFrame(
            detailed_evaluation_result.report.test.confusion_matrix,
            index=detailed_evaluation_result.report.test.class_labels,
            columns=detailed_evaluation_result.report.test.class_labels,
        )

        validation_heatmap_path = _get_available_path(
            project_paths.heatmaps_reports_dir / "validation_confusion_heatmap.png"
        )
        test_heatmap_path = _get_available_path(
            project_paths.heatmaps_reports_dir / "test_confusion_heatmap.png"
        )
        _save_heatmap(
            validation_dataframe,
            title="Матрица ошибок validation",
            target_path=validation_heatmap_path,
        )
        _save_heatmap(
            test_dataframe,
            title="Матрица ошибок test",
            target_path=test_heatmap_path,
        )
        exported_heatmap_names.extend(["validation", "test"])

    if batch_evaluation_result is not None:
        batch_heatmap_path = _get_available_path(
            project_paths.heatmaps_reports_dir / "batch_confusion_heatmap.png"
        )
        _save_heatmap(
            batch_evaluation_result.confusion_matrix_dataframe,
            title="Матрица ошибок пакетной оценки",
            target_path=batch_heatmap_path,
        )
        exported_heatmap_names.append("batch")

    manifest_path = _get_available_path(
        project_paths.heatmaps_reports_dir / "heatmaps_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "exported_heatmap_names": exported_heatmap_names,
                "paths": {
                    "validation_heatmap_path": str(validation_heatmap_path) if validation_heatmap_path else "",
                    "test_heatmap_path": str(test_heatmap_path) if test_heatmap_path else "",
                    "batch_heatmap_path": str(batch_heatmap_path) if batch_heatmap_path else "",
                    "manifest_path": str(manifest_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return ConfusionHeatmapExportResult(
        exported_heatmap_names=tuple(exported_heatmap_names),
        paths=ConfusionHeatmapExportPaths(
            validation_heatmap_path=validation_heatmap_path,
            test_heatmap_path=test_heatmap_path,
            batch_heatmap_path=batch_heatmap_path,
            manifest_path=manifest_path,
        ),
    )
