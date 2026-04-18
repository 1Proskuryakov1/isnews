# -*- coding: utf-8 -*-
"""Обучает демонстрационные модели и сохраняет сдаваемые артефакты проекта IsNews.

Скрипт нужен для финальной комплектации репозитория: он создает локальный учебный
датасет новостных публикаций, запускает штатные этапы пайплайна проекта и
сохраняет модели с именами, ожидаемыми в требованиях ВКР: model.joblib и
model1.joblib.
"""

from __future__ import annotations

import json
import sys
from tempfile import TemporaryDirectory
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.isnews.config import PROJECT_PATHS, ProjectPaths
from src.isnews.dataset_split import DatasetSplitConfig, split_dataset
from src.isnews.logistic_regression_training import train_logistic_regression
from src.isnews.model_evaluation import evaluate_trained_model
from src.isnews.multinomial_nb_training import train_multinomial_nb
from src.isnews.text_preprocessing import preprocess_dataset
from src.isnews.tfidf_vectorization import TfidfVectorizationConfig, vectorize_split_result


RAW_DATASET_PATH = PROJECT_PATHS.raw_data_dir / "news_demo_dataset.csv"
PROCESSED_DATASET_PATH = PROJECT_PATHS.processed_data_dir / "news_demo_dataset_processed.csv"
SPLIT_DIRECTORY = PROJECT_PATHS.split_data_dir / "news_demo_dataset"
FEATURE_DIRECTORY = PROJECT_PATHS.feature_data_dir / "news_demo_dataset_tfidf"
MODEL_PATH = PROJECT_PATHS.classifiers_dir / "model.joblib"
MODEL1_PATH = PROJECT_PATHS.classifiers_dir / "model1.joblib"
VECTORIZER_PATH = PROJECT_PATHS.vectorizers_dir / "model_vectorizer.joblib"
MODEL_MANIFEST_PATH = PROJECT_PATHS.models_dir / "model_manifest.json"
FINAL_SUMMARY_PATH = PROJECT_PATHS.training_reports_dir / "final_demo_training_summary.json"
FINAL_COMPARISON_CSV_PATH = PROJECT_PATHS.comparison_reports_dir / "final_model_comparison.csv"
FINAL_COMPARISON_JSON_PATH = PROJECT_PATHS.comparison_reports_dir / "final_model_comparison.json"


CATEGORY_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "политика": {
        "subjects": (
            "правительство",
            "парламент",
            "избирательная комиссия",
            "региональная администрация",
            "министерство",
            "общественный совет",
        ),
        "actions": (
            "обсудило законопроект",
            "утвердило новую программу",
            "провело заседание",
            "подготовило поправки",
            "представило доклад",
            "согласовало бюджетные решения",
        ),
        "details": (
            "о муниципальном управлении",
            "о развитии регионов",
            "о поддержке гражданских инициатив",
            "о прозрачности выборов",
            "о работе ведомств",
            "о парламентском контроле",
        ),
        "markers": (
            "депутаты",
            "выборы",
            "закон",
            "кабинет министров",
            "фракция",
            "губернатор",
        ),
    },
    "экономика": {
        "subjects": (
            "центральный банк",
            "фондовый рынок",
            "промышленная компания",
            "министерство финансов",
            "инвесторы",
            "налоговая служба",
        ),
        "actions": (
            "сообщил о росте показателей",
            "пересмотрел прогноз",
            "запустил инвестиционный проект",
            "зафиксировал увеличение спроса",
            "опубликовал финансовый отчет",
            "оценил динамику инфляции",
        ),
        "details": (
            "на рынке облигаций",
            "в производственном секторе",
            "в банковской системе",
            "по итогам квартала",
            "в сфере малого бизнеса",
            "на валютном рынке",
        ),
        "markers": (
            "рубль",
            "инфляция",
            "акции",
            "кредит",
            "экспорт",
            "прибыль",
        ),
    },
    "спорт": {
        "subjects": (
            "футбольный клуб",
            "хоккейная команда",
            "олимпийская сборная",
            "теннисист",
            "тренерский штаб",
            "баскетбольная лига",
        ),
        "actions": (
            "выиграл важный матч",
            "объявил состав на турнир",
            "провел контрольную тренировку",
            "завоевал медаль",
            "подписал нового игрока",
            "обновил рекорд сезона",
        ),
        "details": (
            "в национальном чемпионате",
            "перед решающим финалом",
            "после серии пенальти",
            "на международных соревнованиях",
            "в домашнем матче",
            "по итогам плей-офф",
        ),
        "markers": (
            "гол",
            "тренировка",
            "турнир",
            "болельщики",
            "счет",
            "чемпионат",
        ),
    },
    "технологии": {
        "subjects": (
            "IT-компания",
            "исследовательская лаборатория",
            "разработчики",
            "производитель смартфонов",
            "центр кибербезопасности",
            "стартап",
        ),
        "actions": (
            "представил новую платформу",
            "обновил алгоритм",
            "запустил облачный сервис",
            "показал прототип устройства",
            "усилил защиту данных",
            "открыл доступ к приложению",
        ),
        "details": (
            "на базе машинного обучения",
            "для анализа больших данных",
            "с поддержкой нейронных сетей",
            "для мобильных пользователей",
            "после тестирования безопасности",
            "в рамках цифровой трансформации",
        ),
        "markers": (
            "алгоритм",
            "нейросеть",
            "сервер",
            "приложение",
            "данные",
            "робот",
        ),
    },
    "культура": {
        "subjects": (
            "музей",
            "театр",
            "кинорежиссер",
            "оркестр",
            "литературный фестиваль",
            "художественная галерея",
        ),
        "actions": (
            "открыл новую выставку",
            "представил премьеру",
            "получил профессиональную награду",
            "подготовил концертную программу",
            "провел творческую встречу",
            "анонсировал культурный сезон",
        ),
        "details": (
            "для широкой аудитории",
            "после реставрации коллекции",
            "на международном фестивале",
            "с участием молодых авторов",
            "в историческом здании",
            "в рамках городской программы",
        ),
        "markers": (
            "выставка",
            "спектакль",
            "фильм",
            "музыка",
            "книга",
            "картина",
        ),
    },
}


def _pick(values: tuple[str, ...], index: int, multiplier: int = 1) -> str:
    """Возвращает элемент по циклическому индексу, чтобы тексты были разнообразными."""
    return values[(index * multiplier) % len(values)]


def build_demo_dataset(rows_per_category: int = 36) -> pd.DataFrame:
    """Формирует учебный датасет из коротких, но реалистичных новостных текстов."""
    rows: list[dict[str, str]] = []
    templates = (
        "Выпуск {serial}: {subject} {action} {detail}. В сообщении отдельно отмечены темы: {marker1}, {marker2}.",
        "Новость {serial}: по данным редакции, {subject} {action} {detail}; ключевые слова выпуска: {marker1} и {marker2}.",
        "Материал {serial}: сегодня {subject} {action} {detail}. Эксперты связывают публикацию с темами {marker1} и {marker2}.",
        "Публикация {serial}: {subject} {action} {detail}. В центре внимания редакции: {marker1}, {marker2}.",
    )

    for label, spec in CATEGORY_SPECS.items():
        for index in range(rows_per_category):
            text = _pick(templates, index).format(
                serial=f"{label}-{index + 1:02d}",
                subject=_pick(spec["subjects"], index),
                action=_pick(spec["actions"], index, multiplier=2),
                detail=_pick(spec["details"], index, multiplier=3),
                marker1=_pick(spec["markers"], index, multiplier=4),
                marker2=_pick(spec["markers"], index + 1, multiplier=5),
            )
            rows.append({"text": text, "label": label})

    # Перемешивание фиксировано, чтобы обучение и метрики воспроизводились при каждом запуске.
    return pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)


def _write_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    """Сохраняет таблицу в UTF-8 CSV и заранее создает родительскую директорию."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, encoding="utf-8")


def _write_split_files(split_result) -> None:
    """Сохраняет итоговые train, validation и test выборки в стабильный каталог."""
    SPLIT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_dataframe(split_result.train_dataframe, SPLIT_DIRECTORY / "train.csv")
    _write_dataframe(split_result.validation_dataframe, SPLIT_DIRECTORY / "validation.csv")
    _write_dataframe(split_result.test_dataframe, SPLIT_DIRECTORY / "test.csv")


def _write_feature_files(vectorization_result) -> None:
    """Сохраняет TF-IDF матрицы с признаками в стабильный каталог проекта."""
    FEATURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(FEATURE_DIRECTORY / "train_features.npz", vectorization_result.train_matrix)
    sparse.save_npz(
        FEATURE_DIRECTORY / "validation_features.npz",
        vectorization_result.validation_matrix,
    )
    sparse.save_npz(FEATURE_DIRECTORY / "test_features.npz", vectorization_result.test_matrix)


def _evaluation_record(model_name: str, model_path: Path, training_result, evaluation_result) -> dict[str, object]:
    """Собирает компактную строку сравнения по одной обученной модели."""
    report = evaluation_result.report
    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "final_summary_path": str(FINAL_SUMMARY_PATH),
        "class_count": len(report.class_labels),
        "train_rows": report.train_metrics.support,
        "validation_rows": report.validation_metrics.support,
        "test_rows": report.test_metrics.support,
        "training_seconds": training_result.report.training_seconds,
        "train_accuracy": report.train_metrics.accuracy,
        "validation_accuracy": report.validation_metrics.accuracy,
        "test_accuracy": report.test_metrics.accuracy,
        "validation_f1_macro": report.validation_metrics.f1_macro,
        "test_f1_macro": report.test_metrics.f1_macro,
        "warning_messages": list(report.warning_messages),
    }


def _write_final_reports(
    *,
    processed_result,
    split_result,
    vectorization_result,
    logreg_result,
    nb_result,
    logreg_evaluation,
    nb_evaluation,
) -> None:
    """Сохраняет итоговый manifest, JSON-отчет и таблицу сравнения моделей."""
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    records = [
        _evaluation_record("LogisticRegression", MODEL_PATH, logreg_result, logreg_evaluation),
        _evaluation_record("MultinomialNB", MODEL1_PATH, nb_result, nb_evaluation),
    ]
    comparison_dataframe = pd.DataFrame(records).sort_values(
        by=["validation_accuracy", "test_accuracy", "training_seconds"],
        ascending=[False, False, True],
    )

    FINAL_COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_dataframe.to_csv(FINAL_COMPARISON_CSV_PATH, index=False, encoding="utf-8")
    FINAL_COMPARISON_JSON_PATH.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "model_count": len(records),
                "records": comparison_dataframe.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_payload = {
        "generated_at": generated_at,
        "task": "Классификация новостных публикаций по тематическим категориям",
        "dataset": {
            "raw_dataset_path": str(RAW_DATASET_PATH),
            "processed_dataset_path": str(PROCESSED_DATASET_PATH),
            "split_directory": str(SPLIT_DIRECTORY),
            "rows_after_preprocessing": len(processed_result.dataframe),
            "class_labels": sorted(processed_result.dataframe["label"].unique().tolist()),
            "train_rows": len(split_result.train_dataframe),
            "validation_rows": len(split_result.validation_dataframe),
            "test_rows": len(split_result.test_dataframe),
        },
        "features": {
            "feature_directory": str(FEATURE_DIRECTORY),
            "vectorizer_path": str(VECTORIZER_PATH),
            "vocabulary_size": vectorization_result.report.vocabulary_size,
            "train_shape": vectorization_result.report.train_shape,
            "validation_shape": vectorization_result.report.validation_shape,
            "test_shape": vectorization_result.report.test_shape,
        },
        "models": records,
    }

    FINAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_SUMMARY_PATH.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    MODEL_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "main_model": str(MODEL_PATH),
                "additional_model": str(MODEL1_PATH),
                "vectorizer": str(VECTORIZER_PATH),
                "training_dataset": str(RAW_DATASET_PATH),
                "processed_dataset": str(PROCESSED_DATASET_PATH),
                "split_directory": str(SPLIT_DIRECTORY),
                "comparison_report": str(FINAL_COMPARISON_CSV_PATH),
                "metrics_summary": str(FINAL_SUMMARY_PATH),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    """Запускает полный цикл создания данных, обучения, оценки и сохранения моделей."""
    PROJECT_PATHS.ensure_directories()

    dataset = build_demo_dataset()
    _write_dataframe(dataset, RAW_DATASET_PATH)

    # Штатные функции проекта сохраняют много промежуточных файлов. Для финальной
    # комплектации репозитория они запускаются во временной папке, а в проект
    # копируются только стабильные артефакты, которые нужны для проверки ВКР.
    with TemporaryDirectory(prefix="isnews_training_") as temporary_root:
        training_paths = ProjectPaths.from_root(Path(temporary_root))
        training_paths.ensure_directories()

        processed_result = preprocess_dataset(
            dataset,
            source_dataset_path=RAW_DATASET_PATH,
            project_paths=training_paths,
        )
        _write_dataframe(processed_result.dataframe, PROCESSED_DATASET_PATH)

        split_result = split_dataset(
            processed_result.dataframe,
            source_dataset_path=PROCESSED_DATASET_PATH,
            project_paths=training_paths,
            config=DatasetSplitConfig(
                train_size=0.7,
                validation_size=0.15,
                test_size=0.15,
                random_state=42,
                stratify_by_label=True,
            ),
        )
        _write_split_files(split_result)

        vectorization_result = vectorize_split_result(
            split_result,
            project_paths=training_paths,
            config=TfidfVectorizationConfig(
                max_features=8000,
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 2),
                lowercase=False,
                sublinear_tf=True,
            ),
        )
        _write_feature_files(vectorization_result)

        logreg_result = train_logistic_regression(
            split_result,
            vectorization_result,
            project_paths=training_paths,
        )
        nb_result = train_multinomial_nb(
            split_result,
            vectorization_result,
            project_paths=training_paths,
        )

        logreg_evaluation = evaluate_trained_model(
            split_result,
            vectorization_result,
            logreg_result,
            project_paths=training_paths,
        )
        nb_evaluation = evaluate_trained_model(
            split_result,
            vectorization_result,
            nb_result,
            project_paths=training_paths,
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(logreg_result.model, MODEL_PATH)
        joblib.dump(nb_result.model, MODEL1_PATH)
        joblib.dump(vectorization_result.vectorizer, VECTORIZER_PATH)

        _write_final_reports(
            processed_result=processed_result,
            split_result=split_result,
            vectorization_result=vectorization_result,
            logreg_result=logreg_result,
            nb_result=nb_result,
            logreg_evaluation=logreg_evaluation,
            nb_evaluation=nb_evaluation,
        )

        print(json.dumps(asdict(logreg_evaluation.report.train_metrics), ensure_ascii=False))
        print(json.dumps(asdict(logreg_evaluation.report.validation_metrics), ensure_ascii=False))
        print(json.dumps(asdict(logreg_evaluation.report.test_metrics), ensure_ascii=False))
    print(f"saved_model={MODEL_PATH}")
    print(f"saved_model1={MODEL1_PATH}")
    print(f"saved_vectorizer={VECTORIZER_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
