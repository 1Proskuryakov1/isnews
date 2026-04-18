"""TF-IDF-векторизация подготовленных подвыборок датасета."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
from scipy import sparse
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.dataset_split import DatasetSplitResult


class TfidfVectorizationError(ValueError):
    """Ошибка построения TF-IDF-признаков."""


@dataclass(frozen=True)
class TfidfVectorizationConfig:
    """Параметры построения TF-IDF-признаков."""

    max_features: int = 20000
    min_df: int = 1
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)
    lowercase: bool = False
    sublinear_tf: bool = True


@dataclass(frozen=True)
class TfidfVectorizationReport:
    """Содержит краткие сведения о результатах векторизации."""

    vocabulary_size: int
    train_shape: tuple[int, int]
    validation_shape: tuple[int, int]
    test_shape: tuple[int, int]
    train_density: float
    validation_density: float
    test_density: float
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class TfidfVectorizationPaths:
    """Хранит пути ко всем артефактам этапа TF-IDF."""

    feature_directory: Path
    train_matrix_path: Path
    validation_matrix_path: Path
    test_matrix_path: Path
    vectorizer_path: Path
    report_path: Path


@dataclass(frozen=True)
class TfidfVectorizationResult:
    """Возвращает матрицы признаков, векторизатор и сохраненные артефакты."""

    train_matrix: spmatrix
    validation_matrix: spmatrix
    test_matrix: spmatrix
    vectorizer: TfidfVectorizer
    config: TfidfVectorizationConfig
    report: TfidfVectorizationReport
    paths: TfidfVectorizationPaths


def _sanitize_name(name: str) -> str:
    """Подготавливает безопасное имя каталога или файла для артефактов."""
    cleaned_name = "".join(
        symbol if symbol.isalnum() or symbol in {"-", "_", "."} else "_"
        for symbol in name.strip()
    )
    return cleaned_name or "tfidf_artifact"


def _get_available_path(target_path: Path) -> Path:
    """Подбирает свободный путь к файлу или каталогу."""
    if not target_path.exists():
        return target_path

    counter = 1
    while True:
        candidate = target_path.with_name(f"{target_path.stem}_{counter}{target_path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _get_available_directory(target_directory: Path) -> Path:
    """Подбирает свободный каталог для новой версии TF-IDF артефактов."""
    if not target_directory.exists():
        return target_directory

    counter = 1
    while True:
        candidate = target_directory.with_name(f"{target_directory.name}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def _matrix_density(matrix: spmatrix) -> float:
    """Рассчитывает долю ненулевых элементов sparse-матрицы."""
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return 0.0
    return round(float(matrix.nnz / (rows * columns)), 6)


def _build_vectorizer(config: TfidfVectorizationConfig) -> TfidfVectorizer:
    """Создает объект TF-IDF-векторизатора из конфигурации."""
    return TfidfVectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        ngram_range=config.ngram_range,
        lowercase=config.lowercase,
        sublinear_tf=config.sublinear_tf,
    )


def _validate_split_result(split_result: DatasetSplitResult) -> None:
    """Проверяет, что сплиты содержат необходимые колонки и не пусты."""
    required_columns = {"text", "label"}
    for split_name, dataframe in (
        ("train", split_result.train_dataframe),
        ("validation", split_result.validation_dataframe),
        ("test", split_result.test_dataframe),
    ):
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise TfidfVectorizationError(
                f"В подвыборке `{split_name}` отсутствуют колонки: "
                f"{', '.join(sorted(missing_columns))}."
            )
        if dataframe.empty:
            raise TfidfVectorizationError(
                f"Подвыборка `{split_name}` пуста. Векторизация невозможна."
            )


def _save_feature_matrices(
    *,
    train_matrix: spmatrix,
    validation_matrix: spmatrix,
    test_matrix: spmatrix,
    split_result: DatasetSplitResult,
    project_paths: ProjectPaths,
) -> tuple[Path, Path, Path, Path]:
    """Сохраняет sparse-матрицы признаков в каталог `data/features`."""
    project_paths.ensure_directories()

    directory_name = _sanitize_name(f"{split_result.paths.directory.name}_tfidf")
    feature_directory = _get_available_directory(
        project_paths.feature_data_dir / directory_name
    )
    feature_directory.mkdir(parents=True, exist_ok=True)

    train_matrix_path = feature_directory / "train_features.npz"
    validation_matrix_path = feature_directory / "validation_features.npz"
    test_matrix_path = feature_directory / "test_features.npz"

    sparse.save_npz(train_matrix_path, train_matrix)
    sparse.save_npz(validation_matrix_path, validation_matrix)
    sparse.save_npz(test_matrix_path, test_matrix)

    return (
        feature_directory,
        train_matrix_path,
        validation_matrix_path,
        test_matrix_path,
    )


def _save_vectorizer(
    *,
    vectorizer: TfidfVectorizer,
    split_result: DatasetSplitResult,
    project_paths: ProjectPaths,
) -> Path:
    """Сохраняет TF-IDF-векторизатор в каталог `models/vectorizers`."""
    project_paths.ensure_directories()

    vectorizer_name = _sanitize_name(
        f"{split_result.paths.directory.name}_tfidf_vectorizer.joblib"
    )
    vectorizer_path = _get_available_path(project_paths.vectorizers_dir / vectorizer_name)
    joblib.dump(vectorizer, vectorizer_path)
    return vectorizer_path


def _save_vectorization_report(
    *,
    report: TfidfVectorizationReport,
    config: TfidfVectorizationConfig,
    split_result: DatasetSplitResult,
    paths: TfidfVectorizationPaths,
    project_paths: ProjectPaths,
) -> None:
    """Сохраняет JSON-отчет по результатам TF-IDF-векторизации."""
    project_paths.ensure_directories()

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": asdict(config),
        "report": asdict(report),
        "source_split_directory": str(split_result.paths.directory),
        "paths": {
            "feature_directory": str(paths.feature_directory),
            "train_matrix_path": str(paths.train_matrix_path),
            "validation_matrix_path": str(paths.validation_matrix_path),
            "test_matrix_path": str(paths.test_matrix_path),
            "vectorizer_path": str(paths.vectorizer_path),
        },
    }

    paths.report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def vectorize_split_result(
    split_result: DatasetSplitResult,
    *,
    project_paths: ProjectPaths = PROJECT_PATHS,
    config: TfidfVectorizationConfig | None = None,
) -> TfidfVectorizationResult:
    """Строит TF-IDF-признаки по train-части и применяет их к validation и test."""
    resolved_config = config or TfidfVectorizationConfig()
    _validate_split_result(split_result)

    vectorizer = _build_vectorizer(resolved_config)
    warning_messages: list[str] = []

    train_texts = split_result.train_dataframe["text"].astype("string")
    validation_texts = split_result.validation_dataframe["text"].astype("string")
    test_texts = split_result.test_dataframe["text"].astype("string")

    try:
        train_matrix = vectorizer.fit_transform(train_texts)
        validation_matrix = vectorizer.transform(validation_texts)
        test_matrix = vectorizer.transform(test_texts)
    except ValueError as error:
        raise TfidfVectorizationError(
            f"Не удалось построить TF-IDF-признаки: {error}"
        ) from error

    vocabulary_size = len(vectorizer.vocabulary_)
    if vocabulary_size == 0:
        raise TfidfVectorizationError(
            "TF-IDF-векторизатор построил пустой словарь признаков."
        )

    if train_matrix.shape[1] < 20:
        warning_messages.append(
            "Словарь TF-IDF получился очень маленьким. Для обучения модели "
            "может понадобиться больший датасет."
        )

    (
        feature_directory,
        train_matrix_path,
        validation_matrix_path,
        test_matrix_path,
    ) = _save_feature_matrices(
        train_matrix=train_matrix,
        validation_matrix=validation_matrix,
        test_matrix=test_matrix,
        split_result=split_result,
        project_paths=project_paths,
    )
    vectorizer_path = _save_vectorizer(
        vectorizer=vectorizer,
        split_result=split_result,
        project_paths=project_paths,
    )
    report_path = _get_available_path(
        project_paths.vectorization_reports_dir
        / f"{split_result.paths.directory.name}_tfidf_report.json"
    )

    report = TfidfVectorizationReport(
        vocabulary_size=vocabulary_size,
        train_shape=(int(train_matrix.shape[0]), int(train_matrix.shape[1])),
        validation_shape=(int(validation_matrix.shape[0]), int(validation_matrix.shape[1])),
        test_shape=(int(test_matrix.shape[0]), int(test_matrix.shape[1])),
        train_density=_matrix_density(train_matrix),
        validation_density=_matrix_density(validation_matrix),
        test_density=_matrix_density(test_matrix),
        warning_messages=tuple(warning_messages),
    )

    paths = TfidfVectorizationPaths(
        feature_directory=feature_directory,
        train_matrix_path=train_matrix_path,
        validation_matrix_path=validation_matrix_path,
        test_matrix_path=test_matrix_path,
        vectorizer_path=vectorizer_path,
        report_path=report_path,
    )
    _save_vectorization_report(
        report=report,
        config=resolved_config,
        split_result=split_result,
        paths=paths,
        project_paths=project_paths,
    )

    return TfidfVectorizationResult(
        train_matrix=train_matrix,
        validation_matrix=validation_matrix,
        test_matrix=test_matrix,
        vectorizer=vectorizer,
        config=resolved_config,
        report=report,
        paths=paths,
    )
