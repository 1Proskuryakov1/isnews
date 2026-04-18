# -*- coding: utf-8 -*-
"""Проверяет готовность проекта IsNews к демонстрации и сдаче.

Скрипт не изменяет обученные модели и датасеты. Он выполняет контрольные проверки:
наличие обязательных файлов, валидность ноутбуков, закрепление зависимостей,
объем кода, загрузку сохраненных моделей и расчет Accuracy на test-выборке.
"""

from __future__ import annotations

import compileall
import importlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

REQUIRED_FILES = (
    "main.py",
    "requirements.txt",
    "README.md",
    "data/raw/news_demo_dataset.csv",
    "data/processed/news_demo_dataset_processed.csv",
    "data/splits/news_demo_dataset/train.csv",
    "data/splits/news_demo_dataset/validation.csv",
    "data/splits/news_demo_dataset/test.csv",
    "data/features/news_demo_dataset_tfidf/train_features.npz",
    "data/features/news_demo_dataset_tfidf/validation_features.npz",
    "data/features/news_demo_dataset_tfidf/test_features.npz",
    "models/classifiers/model.joblib",
    "models/classifiers/model1.joblib",
    "models/vectorizers/model_vectorizer.joblib",
    "models/model_manifest.json",
    "reports/training/final_demo_training_summary.json",
    "reports/comparisons/final_model_comparison.csv",
    "notebooks/colab_baseline_training.ipynb",
    "notebooks/colab_transformers_training.ipynb",
    "notebooks/colab_batch_inference_evaluation.ipynb",
    "docs/LIBRARY_JUSTIFICATION.md",
    "docs/MODEL_MATH.md",
    "docs/TEST_PLAN.md",
    "docs/TEST_CASES.md",
    "docs/BUG_REPORTS.md",
    "docs/APP_DESIGN.md",
    "docs/FINAL_READINESS_CHECKLIST.md",
)

REQUIRED_IMPORTS = (
    "streamlit",
    "pandas",
    "numpy",
    "requests",
    "joblib",
    "docx",
    "sklearn",
    "scipy",
    "torch",
    "transformers",
    "datasets",
    "safetensors",
    "matplotlib",
    "seaborn",
    "plotly",
)


@dataclass(frozen=True)
class VerificationItem:
    """Хранит результат одной контрольной проверки."""

    name: str
    passed: bool
    details: str


def _relative(path: Path) -> str:
    """Преобразует путь в читаемый относительный формат."""
    return str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")


def _check_required_files() -> VerificationItem:
    """Проверяет наличие обязательных файлов сдаваемого комплекта."""
    missing_files = [
        relative_path
        for relative_path in REQUIRED_FILES
        if not (REPOSITORY_ROOT / relative_path).is_file()
    ]
    if missing_files:
        return VerificationItem(
            name="required_files",
            passed=False,
            details="Отсутствуют файлы: " + ", ".join(missing_files),
        )
    return VerificationItem(
        name="required_files",
        passed=True,
        details=f"Найдены все обязательные файлы: {len(REQUIRED_FILES)}.",
    )


def _check_requirements_pinned() -> VerificationItem:
    """Проверяет, что зависимости закреплены через оператор ==."""
    requirements_path = REPOSITORY_ROOT / "requirements.txt"
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    dependency_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]
    unpinned = [line for line in dependency_lines if "==" not in line]
    if unpinned:
        return VerificationItem(
            name="requirements_pinned",
            passed=False,
            details="Не закреплены версии: " + ", ".join(unpinned),
        )
    return VerificationItem(
        name="requirements_pinned",
        passed=True,
        details=f"Закреплены версии зависимостей: {len(dependency_lines)}.",
    )


def _check_imports() -> VerificationItem:
    """Проверяет импорт библиотек, перечисленных в requirements.txt."""
    failed_imports: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as error:
            failed_imports.append(f"{module_name}: {error}")

    if failed_imports:
        return VerificationItem(
            name="dependency_imports",
            passed=False,
            details="Ошибки импорта: " + "; ".join(failed_imports),
        )
    return VerificationItem(
        name="dependency_imports",
        passed=True,
        details=f"Все библиотеки импортируются: {len(REQUIRED_IMPORTS)}.",
    )


def _check_python_compilation() -> VerificationItem:
    """Компилирует Python-код проекта без выполнения приложения."""
    targets = [
        REPOSITORY_ROOT / "main.py",
        REPOSITORY_ROOT / "src",
        REPOSITORY_ROOT / "scripts",
    ]
    results: list[bool] = []
    for target in targets:
        if target.is_file():
            results.append(compileall.compile_file(str(target), quiet=1))
        else:
            results.append(compileall.compile_dir(str(target), quiet=1))

    if not all(results):
        return VerificationItem(
            name="python_compilation",
            passed=False,
            details="Не все Python-файлы прошли компиляцию.",
        )
    return VerificationItem(
        name="python_compilation",
        passed=True,
        details="main.py, src/ и scripts/ компилируются без ошибок.",
    )


def _check_notebooks() -> VerificationItem:
    """Проверяет, что ipynb-файлы являются корректным JSON notebook v4."""
    notebook_paths = sorted((REPOSITORY_ROOT / "notebooks").glob("*.ipynb"))
    invalid_notebooks: list[str] = []
    for notebook_path in notebook_paths:
        try:
            payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            invalid_notebooks.append(f"{_relative(notebook_path)}: {error}")
            continue

        if payload.get("nbformat") != 4 or not isinstance(payload.get("cells"), list):
            invalid_notebooks.append(f"{_relative(notebook_path)}: некорректный формат notebook")

    if invalid_notebooks:
        return VerificationItem(
            name="notebooks",
            passed=False,
            details="; ".join(invalid_notebooks),
        )
    return VerificationItem(
        name="notebooks",
        passed=True,
        details=f"Корректные notebook-файлы: {len(notebook_paths)}.",
    )


def _logical_python_lines() -> int:
    """Считает логические строки Python-кода без пустых строк и комментариев."""
    ignored_parts = {".venv", "__pycache__", ".git"}
    total = 0
    for python_file in REPOSITORY_ROOT.rglob("*.py"):
        if ignored_parts.intersection(python_file.parts):
            continue
        for line in python_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


def _check_code_volume() -> VerificationItem:
    """Проверяет требование по объему собственноручно разработанного кода."""
    line_count = _logical_python_lines()
    return VerificationItem(
        name="logical_code_lines",
        passed=line_count >= 2000,
        details=f"Логических строк Python-кода: {line_count}; требуется не менее 2000.",
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Загружает JSON-файл с UTF-8 кодировкой."""
    return json.loads(path.read_text(encoding="utf-8"))


def _check_models_and_metrics() -> VerificationItem:
    """Загружает сохраненные модели и оценивает их на сохраненной test-выборке."""
    model_path = REPOSITORY_ROOT / "models" / "classifiers" / "model.joblib"
    model1_path = REPOSITORY_ROOT / "models" / "classifiers" / "model1.joblib"
    vectorizer_path = REPOSITORY_ROOT / "models" / "vectorizers" / "model_vectorizer.joblib"
    test_path = REPOSITORY_ROOT / "data" / "splits" / "news_demo_dataset" / "test.csv"
    summary_path = REPOSITORY_ROOT / "reports" / "training" / "final_demo_training_summary.json"

    model = joblib.load(model_path)
    model1 = joblib.load(model1_path)
    vectorizer = joblib.load(vectorizer_path)
    test_dataframe = pd.read_csv(test_path)
    test_matrix = vectorizer.transform(test_dataframe["text"].astype(str))

    model_accuracy = float(accuracy_score(test_dataframe["label"], model.predict(test_matrix)))
    model1_accuracy = float(accuracy_score(test_dataframe["label"], model1.predict(test_matrix)))
    summary = _load_json(summary_path)

    passed = (
        model_accuracy >= 0.7
        and model1_accuracy >= 0.7
        and int(summary["dataset"]["test_rows"]) == len(test_dataframe)
    )
    return VerificationItem(
        name="models_and_metrics",
        passed=passed,
        details=(
            f"model.joblib Accuracy={model_accuracy:.4f}; "
            f"model1.joblib Accuracy={model1_accuracy:.4f}; "
            f"test rows={len(test_dataframe)}."
        ),
    )


def _check_training_dataset() -> VerificationItem:
    """Проверяет размер и структуру сохраненного обучающего датасета."""
    raw_path = REPOSITORY_ROOT / "data" / "raw" / "news_demo_dataset.csv"
    processed_path = REPOSITORY_ROOT / "data" / "processed" / "news_demo_dataset_processed.csv"
    raw_dataframe = pd.read_csv(raw_path)
    processed_dataframe = pd.read_csv(processed_path)
    required_columns = {"text", "label"}

    passed = (
        required_columns.issubset(raw_dataframe.columns)
        and required_columns.issubset(processed_dataframe.columns)
        and len(raw_dataframe) == 180
        and len(processed_dataframe) == 180
        and processed_dataframe["label"].nunique() == 5
    )
    return VerificationItem(
        name="training_dataset",
        passed=passed,
        details=(
            f"raw rows={len(raw_dataframe)}; processed rows={len(processed_dataframe)}; "
            f"classes={processed_dataframe['label'].nunique()}."
        ),
    )


def _check_git_commit_count() -> VerificationItem:
    """Проверяет количество коммитов, если проект открыт внутри git-репозитория."""
    try:
        completed = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        commit_count = int(completed.stdout.strip())
    except Exception as error:
        return VerificationItem(
            name="git_commits",
            passed=False,
            details=f"Не удалось получить количество коммитов: {error}",
        )

    return VerificationItem(
        name="git_commits",
        passed=commit_count >= 50,
        details=f"Коммитов в текущей ветке: {commit_count}; требуется не менее 50.",
    )


def run_verification() -> list[VerificationItem]:
    """Выполняет все контрольные проверки проекта."""
    return [
        _check_required_files(),
        _check_requirements_pinned(),
        _check_imports(),
        _check_python_compilation(),
        _check_notebooks(),
        _check_code_volume(),
        _check_training_dataset(),
        _check_models_and_metrics(),
        _check_git_commit_count(),
    ]


def main() -> int:
    """Печатает JSON-отчет проверки и возвращает код ошибки при провале."""
    items = run_verification()
    payload = {
        "passed": all(item.passed for item in items),
        "checks": [asdict(item) for item in items],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
