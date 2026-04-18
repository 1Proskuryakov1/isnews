"""Минимальный пользовательский интерфейс демонстрационного приложения."""

from __future__ import annotations

import pandas as pd

from src.isnews.config import PROJECT_PATHS
from src.isnews.data_loading import DatasetLoadResult, DatasetValidationError
from src.isnews.detailed_model_evaluation import (
    DetailedModelEvaluationError,
    DetailedModelEvaluationResult,
    evaluate_model_in_detail,
)
from src.isnews.data_loading import load_dataset_from_uploaded_bytes, load_dataset_from_url
from src.isnews.dataset_split import DatasetSplitError, DatasetSplitResult, split_dataset
from src.isnews.logistic_regression_training import (
    LogisticRegressionTrainingError,
    LogisticRegressionTrainingResult,
    train_logistic_regression,
)
from src.isnews.model_evaluation import (
    ModelEvaluationError,
    ModelEvaluationResult,
    evaluate_trained_model,
)
from src.isnews.text_preprocessing import TextPreprocessingError, TextPreprocessingResult
from src.isnews.text_preprocessing import preprocess_dataset
from src.isnews.tfidf_vectorization import (
    TfidfVectorizationError,
    TfidfVectorizationResult,
    vectorize_split_result,
)


def _render_dataset_statistics(dataset_result: DatasetLoadResult) -> None:
    """Показывает статистику по классам и длине текстов для корректных строк."""
    import streamlit as st

    dataset_summary = dataset_result.dataset_summary
    class_distribution_table = pd.DataFrame(
        [
            {
                "Класс": item.label,
                "Количество": item.count,
                "Доля": round(item.share * 100, 2),
            }
            for item in dataset_summary.class_distribution
        ]
    )

    st.subheader("Сводка по датасету")
    metric_column_1, metric_column_2, metric_column_3 = st.columns(3)
    metric_column_1.metric("Уникальных классов", dataset_summary.unique_classes)
    metric_column_2.metric(
        "Средняя длина текста, символов",
        dataset_summary.text_length_chars.mean,
    )
    metric_column_3.metric(
        "Средняя длина текста, слов",
        dataset_summary.text_length_words.mean,
    )

    st.write("Распределение классов по корректным строкам:")
    st.dataframe(class_distribution_table, use_container_width=True)

    st.write("Статистика длины текстов:")
    st.markdown(
        "\n".join(
            [
                f"- символы: min `{dataset_summary.text_length_chars.minimum}`, "
                f"median `{dataset_summary.text_length_chars.median}`, "
                f"max `{dataset_summary.text_length_chars.maximum}`;",
                f"- слова: min `{dataset_summary.text_length_words.minimum}`, "
                f"median `{dataset_summary.text_length_words.median}`, "
                f"max `{dataset_summary.text_length_words.maximum}`;",
                f"- JSON-сводка сохранена в `{dataset_result.summary_path}`.",
            ]
        )
    )


def _render_preprocessing_preview(
    preprocessing_result: TextPreprocessingResult,
) -> None:
    """Показывает результаты очистки текста и сохранения обработанного датасета."""
    import streamlit as st

    preprocessing_report = preprocessing_result.report
    preprocessing_summary = preprocessing_result.summary

    st.success(
        "Очищенный датасет сохранен: "
        f"`{preprocessing_result.saved_path}`"
    )

    metric_column_1, metric_column_2, metric_column_3, metric_column_4 = st.columns(4)
    metric_column_1.metric("Строк до очистки", preprocessing_report.rows_before)
    metric_column_2.metric("Строк после очистки", preprocessing_report.rows_after)
    metric_column_3.metric(
        "Удалено дубликатов",
        preprocessing_report.removed_duplicate_rows,
    )
    metric_column_4.metric(
        "Изменено текстов",
        preprocessing_report.changed_text_rows,
    )

    st.markdown(
        "\n".join(
            [
                f"- удалено неполных строк до очистки: "
                f"`{preprocessing_report.removed_invalid_rows}`;",
                f"- удалено пустых строк после очистки: "
                f"`{preprocessing_report.removed_empty_after_cleaning_rows}`;",
                f"- уникальных классов после очистки: "
                f"`{preprocessing_summary.unique_classes}`;",
                f"- средняя длина текста после очистки, символов: "
                f"`{preprocessing_summary.text_length_chars.mean}`;",
                f"- отчет сохранен в `{preprocessing_result.report_path}`.",
            ]
        )
    )

    st.write("Первые 10 строк очищенного датасета:")
    st.dataframe(preprocessing_result.dataframe.head(10), use_container_width=True)


def _render_split_preview(split_result: DatasetSplitResult) -> None:
    """Показывает результат разбиения очищенного датасета на три подвыборки."""
    import streamlit as st

    split_report = split_result.report
    split_overview_table = pd.DataFrame(
        [
            {
                "Подвыборка": "train",
                "Строк": split_report.train_rows,
                "Уникальных классов": split_result.train_summary.unique_classes,
                "Средняя длина текста, слов": split_result.train_summary.text_length_words.mean,
            },
            {
                "Подвыборка": "validation",
                "Строк": split_report.validation_rows,
                "Уникальных классов": split_result.validation_summary.unique_classes,
                "Средняя длина текста, слов": split_result.validation_summary.text_length_words.mean,
            },
            {
                "Подвыборка": "test",
                "Строк": split_report.test_rows,
                "Уникальных классов": split_result.test_summary.unique_classes,
                "Средняя длина текста, слов": split_result.test_summary.text_length_words.mean,
            },
        ]
    )

    st.success(
        "Разбиение датасета сохранено в каталог: "
        f"`{split_result.paths.directory}`"
    )

    metric_column_1, metric_column_2, metric_column_3, metric_column_4 = st.columns(4)
    metric_column_1.metric("Train", split_report.train_rows)
    metric_column_2.metric("Validation", split_report.validation_rows)
    metric_column_3.metric("Test", split_report.test_rows)
    metric_column_4.metric(
        "Стратификация",
        "да" if split_report.stratified_split_used else "нет",
    )

    for warning_message in split_report.warning_messages:
        st.warning(warning_message)

    st.markdown(
        "\n".join(
            [
                f"- `train.csv`: `{split_result.paths.train_path}`;",
                f"- `validation.csv`: `{split_result.paths.validation_path}`;",
                f"- `test.csv`: `{split_result.paths.test_path}`;",
                f"- JSON-отчет: `{split_result.paths.report_path}`.",
            ]
        )
    )

    st.write("Сводка по подвыборкам:")
    st.dataframe(split_overview_table, use_container_width=True)

    st.write("Первые 5 строк train-выборки:")
    st.dataframe(split_result.train_dataframe.head(5), use_container_width=True)


def _render_vectorization_preview(
    vectorization_result: TfidfVectorizationResult,
) -> None:
    """Показывает результат TF-IDF-векторизации и пути к сохраненным артефактам."""
    import streamlit as st

    vectorization_report = vectorization_result.report
    matrix_table = pd.DataFrame(
        [
            {
                "Матрица": "train",
                "Форма": f"{vectorization_report.train_shape[0]} x {vectorization_report.train_shape[1]}",
                "Плотность": vectorization_report.train_density,
                "Файл": str(vectorization_result.paths.train_matrix_path),
            },
            {
                "Матрица": "validation",
                "Форма": f"{vectorization_report.validation_shape[0]} x {vectorization_report.validation_shape[1]}",
                "Плотность": vectorization_report.validation_density,
                "Файл": str(vectorization_result.paths.validation_matrix_path),
            },
            {
                "Матрица": "test",
                "Форма": f"{vectorization_report.test_shape[0]} x {vectorization_report.test_shape[1]}",
                "Плотность": vectorization_report.test_density,
                "Файл": str(vectorization_result.paths.test_matrix_path),
            },
        ]
    )

    st.success(
        "TF-IDF-признаки и векторизатор успешно сохранены."
    )

    metric_column_1, metric_column_2, metric_column_3 = st.columns(3)
    metric_column_1.metric("Размер словаря", vectorization_report.vocabulary_size)
    metric_column_2.metric(
        "Признаков в train",
        vectorization_report.train_shape[1],
    )
    metric_column_3.metric(
        "Плотность train",
        vectorization_report.train_density,
    )

    for warning_message in vectorization_report.warning_messages:
        st.warning(warning_message)

    st.markdown(
        "\n".join(
            [
                f"- каталог sparse-матриц: `{vectorization_result.paths.feature_directory}`;",
                f"- сохраненный векторизатор: `{vectorization_result.paths.vectorizer_path}`;",
                f"- JSON-отчет: `{vectorization_result.paths.report_path}`.",
            ]
        )
    )

    st.write("Сводка по матрицам признаков:")
    st.dataframe(matrix_table, use_container_width=True)


def _render_training_preview(
    training_result: LogisticRegressionTrainingResult,
) -> None:
    """Показывает результат обучения базовой модели Logistic Regression."""
    import streamlit as st

    training_report = training_result.report

    st.success(
        "Базовая модель Logistic Regression обучена и сохранена."
    )

    metric_column_1, metric_column_2, metric_column_3, metric_column_4 = st.columns(4)
    metric_column_1.metric("Классов", len(training_report.class_labels))
    metric_column_2.metric("Время обучения, сек", training_report.training_seconds)
    metric_column_3.metric(
        "Размер coef_",
        f"{training_report.coefficient_shape[0]} x {training_report.coefficient_shape[1]}",
    )
    metric_column_4.metric(
        "Итераций",
        ", ".join(str(value) for value in training_report.iterations_per_class),
    )

    for warning_message in training_report.warning_messages:
        st.warning(warning_message)

    st.markdown(
        "\n".join(
            [
                f"- классы модели: `{', '.join(training_report.class_labels)}`;",
                f"- форма `intercept_`: `{training_report.intercept_shape}`;",
                f"- сохраненная модель: `{training_result.paths.model_path}`;",
                f"- JSON-отчет: `{training_result.paths.report_path}`.",
            ]
        )
    )


def _render_evaluation_preview(
    evaluation_result: ModelEvaluationResult,
) -> None:
    """Показывает итоговые метрики качества для train, validation и test."""
    import streamlit as st

    evaluation_report = evaluation_result.report
    metrics_table = pd.DataFrame(
        [
            {
                "Выборка": "train",
                "Accuracy": evaluation_report.train_metrics.accuracy,
                "Precision macro": evaluation_report.train_metrics.precision_macro,
                "Recall macro": evaluation_report.train_metrics.recall_macro,
                "F1 macro": evaluation_report.train_metrics.f1_macro,
                "Support": evaluation_report.train_metrics.support,
            },
            {
                "Выборка": "validation",
                "Accuracy": evaluation_report.validation_metrics.accuracy,
                "Precision macro": evaluation_report.validation_metrics.precision_macro,
                "Recall macro": evaluation_report.validation_metrics.recall_macro,
                "F1 macro": evaluation_report.validation_metrics.f1_macro,
                "Support": evaluation_report.validation_metrics.support,
            },
            {
                "Выборка": "test",
                "Accuracy": evaluation_report.test_metrics.accuracy,
                "Precision macro": evaluation_report.test_metrics.precision_macro,
                "Recall macro": evaluation_report.test_metrics.recall_macro,
                "F1 macro": evaluation_report.test_metrics.f1_macro,
                "Support": evaluation_report.test_metrics.support,
            },
        ]
    )

    st.success("Метрики качества модели рассчитаны и сохранены.")

    metric_column_1, metric_column_2, metric_column_3 = st.columns(3)
    metric_column_1.metric(
        "Validation Accuracy",
        evaluation_report.validation_metrics.accuracy,
    )
    metric_column_2.metric(
        "Test Accuracy",
        evaluation_report.test_metrics.accuracy,
    )
    metric_column_3.metric(
        "Validation F1",
        evaluation_report.validation_metrics.f1_macro,
    )

    for warning_message in evaluation_report.warning_messages:
        st.warning(warning_message)

    st.markdown(
        "\n".join(
            [
                f"- классы модели: `{', '.join(evaluation_report.class_labels)}`;",
                f"- JSON-отчет по метрикам: `{evaluation_result.paths.report_path}`.",
            ]
        )
    )

    st.write("Сводка метрик:")
    st.dataframe(metrics_table, use_container_width=True)


def _render_detailed_evaluation_preview(
    detailed_evaluation_result: DetailedModelEvaluationResult,
) -> None:
    """Показывает поклассовые метрики и матрицы ошибок для validation и test."""
    import streamlit as st

    validation_metrics_table = pd.DataFrame(
        [
            {
                "Класс": metrics.label,
                "Precision": metrics.precision,
                "Recall": metrics.recall,
                "F1-score": metrics.f1_score,
                "Support": metrics.support,
            }
            for metrics in detailed_evaluation_result.report.validation.per_class_metrics
        ]
    )
    test_metrics_table = pd.DataFrame(
        [
            {
                "Класс": metrics.label,
                "Precision": metrics.precision,
                "Recall": metrics.recall,
                "F1-score": metrics.f1_score,
                "Support": metrics.support,
            }
            for metrics in detailed_evaluation_result.report.test.per_class_metrics
        ]
    )
    validation_confusion_matrix = pd.DataFrame(
        detailed_evaluation_result.report.validation.confusion_matrix,
        index=detailed_evaluation_result.report.validation.class_labels,
        columns=detailed_evaluation_result.report.validation.class_labels,
    )
    test_confusion_matrix = pd.DataFrame(
        detailed_evaluation_result.report.test.confusion_matrix,
        index=detailed_evaluation_result.report.test.class_labels,
        columns=detailed_evaluation_result.report.test.class_labels,
    )

    st.success("Подробный отчет по классам и матрицы ошибок сохранены.")

    for warning_message in detailed_evaluation_result.report.validation.warning_messages:
        st.warning(warning_message)
    for warning_message in detailed_evaluation_result.report.test.warning_messages:
        st.warning(warning_message)

    st.markdown(
        "\n".join(
            [
                f"- JSON-отчет: `{detailed_evaluation_result.paths.report_path}`;",
                f"- CSV матрицы ошибок validation: "
                f"`{detailed_evaluation_result.paths.validation_confusion_matrix_path}`;",
                f"- CSV матрицы ошибок test: "
                f"`{detailed_evaluation_result.paths.test_confusion_matrix_path}`.",
            ]
        )
    )

    validation_column, test_column = st.columns(2)
    with validation_column:
        st.write("Поклассовые метрики validation:")
        st.dataframe(validation_metrics_table, use_container_width=True)
        st.write("Матрица ошибок validation:")
        st.dataframe(validation_confusion_matrix, use_container_width=True)

    with test_column:
        st.write("Поклассовые метрики test:")
        st.dataframe(test_metrics_table, use_container_width=True)
        st.write("Матрица ошибок test:")
        st.dataframe(test_confusion_matrix, use_container_width=True)


def _reset_detailed_evaluation_state() -> None:
    """Сбрасывает состояние подробного отчета при смене базовой оценки."""
    import streamlit as st

    st.session_state.detailed_evaluation_result = None
    st.session_state.detailed_evaluation_source_key = None


def _reset_evaluation_state() -> None:
    """Сбрасывает результат оценки качества при смене модели."""
    import streamlit as st

    st.session_state.evaluation_result = None
    st.session_state.evaluation_source_key = None
    _reset_detailed_evaluation_state()


def _reset_training_state() -> None:
    """Сбрасывает состояние обучения при смене признаков или сплита."""
    import streamlit as st

    st.session_state.training_result = None
    st.session_state.training_source_key = None
    _reset_evaluation_state()


def _reset_vectorization_state() -> None:
    """Сбрасывает результат TF-IDF-векторизации при смене сплита."""
    import streamlit as st

    st.session_state.vectorization_result = None
    st.session_state.vectorization_source_key = None
    _reset_training_state()


def _reset_split_state() -> None:
    """Сбрасывает состояние разбиения при смене подготовленного датасета."""
    import streamlit as st

    st.session_state.split_result = None
    st.session_state.split_source_key = None
    _reset_vectorization_state()


def _reset_preprocessing_state() -> None:
    """Сбрасывает результат предобработки при смене загруженного датасета."""
    import streamlit as st

    st.session_state.preprocessing_result = None
    st.session_state.preprocessing_dataset_key = None
    _reset_split_state()


def _render_preprocessing_section(dataset_result: DatasetLoadResult) -> None:
    """Отрисовывает блок запуска предобработки для уже загруженного датасета."""
    import streamlit as st

    current_dataset_key = str(dataset_result.saved_path)
    if "preprocessing_result" not in st.session_state:
        st.session_state.preprocessing_result = None
    if "preprocessing_dataset_key" not in st.session_state:
        st.session_state.preprocessing_dataset_key = None

    if st.session_state.preprocessing_dataset_key != current_dataset_key:
        st.session_state.preprocessing_result = None
        st.session_state.preprocessing_dataset_key = current_dataset_key

    st.subheader("Предобработка текста")
    st.caption(
        "На этом этапе выполняются перевод текста в нижний регистр, нормализация "
        "пробелов и базовой пунктуации, а также удаление дубликатов по паре "
        "`text + label`."
    )

    if st.button(
        "Очистить текст и сохранить подготовленный датасет",
        use_container_width=True,
    ):
        try:
            _reset_split_state()
            st.session_state.preprocessing_result = preprocess_dataset(
                dataset_result.dataframe,
                source_dataset_path=dataset_result.saved_path,
            )
            st.session_state.preprocessing_dataset_key = current_dataset_key
        except TextPreprocessingError as error:
            st.session_state.preprocessing_result = None
            _reset_split_state()
            st.error(str(error))

    preprocessing_result = st.session_state.preprocessing_result
    if preprocessing_result is None:
        st.info(
            "После запуска предобработки здесь появятся сведения об очистке текста, "
            "удаленных дубликатах и путь к сохраненному CSV."
        )
        return

    _render_preprocessing_preview(preprocessing_result)


def _render_split_section(preprocessing_result: TextPreprocessingResult) -> None:
    """Отрисовывает блок разбиения очищенного датасета на train, validation и test."""
    import streamlit as st

    current_source_key = str(preprocessing_result.saved_path)
    if "split_result" not in st.session_state:
        st.session_state.split_result = None
    if "split_source_key" not in st.session_state:
        st.session_state.split_source_key = None

    if st.session_state.split_source_key != current_source_key:
        st.session_state.split_result = None
        st.session_state.split_source_key = current_source_key

    st.subheader("Разбиение датасета")
    st.caption(
        "Подготовленный датасет разбивается на `train`, `validation` и `test` "
        "в долях 70% / 15% / 15%. По возможности используется стратификация по метке класса."
    )

    if st.button(
        "Сформировать train / validation / test",
        use_container_width=True,
    ):
        try:
            _reset_split_state()
            st.session_state.split_result = split_dataset(
                preprocessing_result.dataframe,
                source_dataset_path=preprocessing_result.saved_path,
            )
            st.session_state.split_source_key = current_source_key
        except DatasetSplitError as error:
            st.session_state.split_result = None
            _reset_vectorization_state()
            st.error(str(error))

    split_result = st.session_state.split_result
    if split_result is None:
        st.info(
            "После запуска разбиения здесь появятся размеры подвыборок, пути к CSV "
            "и отчет по train / validation / test."
        )
        return

    _render_split_preview(split_result)


def _render_vectorization_section(split_result: DatasetSplitResult) -> None:
    """Отрисовывает блок TF-IDF-векторизации поверх сохраненных сплитов."""
    import streamlit as st

    current_source_key = str(split_result.paths.directory)
    if "vectorization_result" not in st.session_state:
        st.session_state.vectorization_result = None
    if "vectorization_source_key" not in st.session_state:
        st.session_state.vectorization_source_key = None

    if st.session_state.vectorization_source_key != current_source_key:
        st.session_state.vectorization_result = None
        st.session_state.vectorization_source_key = current_source_key

    st.subheader("TF-IDF-векторизация")
    st.caption(
        "Векторизатор обучается только на `train`-части, после чего применяется "
        "к `validation` и `test`. Результат сохраняется как sparse-матрицы и `.joblib`."
    )

    if st.button(
        "Построить TF-IDF признаки и сохранить векторизатор",
        use_container_width=True,
    ):
        try:
            _reset_vectorization_state()
            st.session_state.vectorization_result = vectorize_split_result(split_result)
            st.session_state.vectorization_source_key = current_source_key
        except TfidfVectorizationError as error:
            st.session_state.vectorization_result = None
            st.error(str(error))

    vectorization_result = st.session_state.vectorization_result
    if vectorization_result is None:
        st.info(
            "После запуска векторизации здесь появятся формы матриц признаков, "
            "размер словаря и пути к сохраненным артефактам."
        )
        return

    _render_vectorization_preview(vectorization_result)


def _render_training_section(
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
) -> None:
    """Отрисовывает блок обучения базовой модели на TF-IDF признаках."""
    import streamlit as st

    current_source_key = str(vectorization_result.paths.vectorizer_path)
    if "training_result" not in st.session_state:
        st.session_state.training_result = None
    if "training_source_key" not in st.session_state:
        st.session_state.training_source_key = None

    if st.session_state.training_source_key != current_source_key:
        st.session_state.training_result = None
        st.session_state.training_source_key = current_source_key

    st.subheader("Обучение Logistic Regression")
    st.caption(
        "На этом этапе базовый линейный классификатор обучается на `train`-матрице "
        "TF-IDF признаков и сохраняется в формате `.joblib`."
    )

    if st.button(
        "Обучить базовую модель и сохранить классификатор",
        use_container_width=True,
    ):
        try:
            _reset_training_state()
            st.session_state.training_result = train_logistic_regression(
                split_result,
                vectorization_result,
            )
            st.session_state.training_source_key = current_source_key
        except LogisticRegressionTrainingError as error:
            st.session_state.training_result = None
            st.error(str(error))

    training_result = st.session_state.training_result
    if training_result is None:
        st.info(
            "После запуска обучения здесь появятся путь к сохраненной модели, "
            "время обучения и краткая сводка по параметрам классификатора."
        )
        return

    _render_training_preview(training_result)


def _render_evaluation_section(
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
    training_result: LogisticRegressionTrainingResult,
) -> None:
    """Отрисовывает блок расчета метрик качества для обученной модели."""
    import streamlit as st

    current_source_key = str(training_result.paths.model_path)
    if "evaluation_result" not in st.session_state:
        st.session_state.evaluation_result = None
    if "evaluation_source_key" not in st.session_state:
        st.session_state.evaluation_source_key = None

    if st.session_state.evaluation_source_key != current_source_key:
        st.session_state.evaluation_result = None
        st.session_state.evaluation_source_key = current_source_key

    st.subheader("Оценка качества")
    st.caption(
        "На этом этапе рассчитываются `Accuracy`, `Precision`, `Recall` и `F1-score` "
        "для `train`, `validation` и `test`."
    )

    if st.button(
        "Рассчитать метрики качества модели",
        use_container_width=True,
    ):
        try:
            _reset_evaluation_state()
            st.session_state.evaluation_result = evaluate_trained_model(
                split_result,
                vectorization_result,
                training_result,
            )
            st.session_state.evaluation_source_key = current_source_key
        except ModelEvaluationError as error:
            st.session_state.evaluation_result = None
            st.error(str(error))

    evaluation_result = st.session_state.evaluation_result
    if evaluation_result is None:
        st.info(
            "После расчета метрик здесь появятся показатели качества на train, "
            "validation и test, а также путь к JSON-отчету."
        )
        return

    _render_evaluation_preview(evaluation_result)


def _render_detailed_evaluation_section(
    split_result: DatasetSplitResult,
    vectorization_result: TfidfVectorizationResult,
    training_result: LogisticRegressionTrainingResult,
    evaluation_result: ModelEvaluationResult,
) -> None:
    """Отрисовывает блок поклассового отчета и матриц ошибок."""
    import streamlit as st

    current_source_key = str(evaluation_result.paths.report_path)
    if "detailed_evaluation_result" not in st.session_state:
        st.session_state.detailed_evaluation_result = None
    if "detailed_evaluation_source_key" not in st.session_state:
        st.session_state.detailed_evaluation_source_key = None

    if st.session_state.detailed_evaluation_source_key != current_source_key:
        st.session_state.detailed_evaluation_result = None
        st.session_state.detailed_evaluation_source_key = current_source_key

    st.subheader("Подробный отчет по классам")
    st.caption(
        "На этом этапе строятся поклассовые `Precision`, `Recall`, `F1-score` "
        "и матрицы ошибок для `validation` и `test`."
    )

    if st.button(
        "Сформировать подробный отчет и матрицы ошибок",
        use_container_width=True,
    ):
        try:
            _reset_detailed_evaluation_state()
            st.session_state.detailed_evaluation_result = evaluate_model_in_detail(
                split_result,
                vectorization_result,
                training_result,
            )
            st.session_state.detailed_evaluation_source_key = current_source_key
        except DetailedModelEvaluationError as error:
            st.session_state.detailed_evaluation_result = None
            st.error(str(error))

    detailed_evaluation_result = st.session_state.detailed_evaluation_result
    if detailed_evaluation_result is None:
        st.info(
            "После запуска этого этапа здесь появятся поклассовые таблицы и "
            "матрицы ошибок для validation и test."
        )
        return

    _render_detailed_evaluation_preview(detailed_evaluation_result)


def _render_dataset_preview(dataset_result: DatasetLoadResult) -> None:
    """Показывает краткую сводку о загруженном датасете и первые строки таблицы."""
    import streamlit as st

    validation_report = dataset_result.validation_report

    st.success(
        "Датасет успешно загружен и сохранен в проект: "
        f"`{dataset_result.saved_path}`"
    )

    metric_column_1, metric_column_2, metric_column_3, metric_column_4 = st.columns(4)
    metric_column_1.metric("Строк", dataset_result.row_count)
    metric_column_2.metric("Столбцов", dataset_result.column_count)
    metric_column_3.metric("Годных строк", validation_report.usable_rows)
    metric_column_4.metric("Строк с пропусками", validation_report.invalid_rows)

    st.write("Результат проверки структуры датасета:")
    st.markdown(
        "\n".join(
            [
                f"- распознана колонка текста: `{validation_report.text_column}` -> `text`;",
                f"- распознана колонка класса: `{validation_report.label_column}` -> `label`;",
                f"- пустых значений в тексте: `{validation_report.empty_text_rows}`;",
                f"- пустых значений в метках класса: `{validation_report.empty_label_rows}`;",
                f"- исходный файл: `{dataset_result.source_name}`.",
            ]
        )
    )

    for warning_message in validation_report.warning_messages:
        st.warning(warning_message)

    _render_dataset_statistics(dataset_result)

    st.write("Названия колонок:")
    st.code(", ".join(str(column) for column in dataset_result.dataframe.columns))

    st.write("Первые 10 строк датасета:")
    st.dataframe(dataset_result.dataframe.head(10), use_container_width=True)


def _render_dataset_loading_section() -> None:
    """Отрисовывает блок загрузки датасета из локального файла или по URL."""
    import streamlit as st

    if "dataset_result" not in st.session_state:
        st.session_state.dataset_result = None
    if "loaded_local_file_key" not in st.session_state:
        st.session_state.loaded_local_file_key = None

    st.subheader("Загрузка датасета")
    st.caption(
        "На текущем этапе датасет должен содержать колонку с текстом новости и "
        "колонку с меткой класса. Поддерживаются алиасы вроде `text`, `content`, "
        "`title`, `label`, `category`, `topic`, а также русскоязычные варианты."
    )
    local_tab, url_tab = st.tabs(["Локальный CSV", "CSV по прямой ссылке"])

    with local_tab:
        uploaded_file = st.file_uploader(
            "Выберите CSV-файл с новостными публикациями",
            type=["csv"],
            help=(
                "На этом этапе поддерживается базовая загрузка CSV. "
                "В следующих итерациях будет добавлена проверка структуры датасета."
            ),
        )

        if uploaded_file is not None:
            uploaded_file_key = (uploaded_file.name, uploaded_file.size)
            if st.session_state.loaded_local_file_key != uploaded_file_key:
                try:
                    st.session_state.dataset_result = load_dataset_from_uploaded_bytes(
                        file_bytes=uploaded_file.getvalue(),
                        source_name=uploaded_file.name,
                    )
                    st.session_state.loaded_local_file_key = uploaded_file_key
                    _reset_preprocessing_state()
                except DatasetValidationError as error:
                    st.session_state.dataset_result = None
                    st.session_state.loaded_local_file_key = None
                    _reset_preprocessing_state()
                    st.error(str(error))
        else:
            st.session_state.loaded_local_file_key = None

    with url_tab:
        dataset_url = st.text_input(
            "Прямая ссылка на CSV-файл",
            placeholder="https://example.com/news_dataset.csv",
        )

        if st.button("Скачать и сохранить датасет", use_container_width=True):
            if not dataset_url.strip():
                st.warning("Укажите прямую ссылку на CSV-файл.")
            else:
                try:
                    st.session_state.dataset_result = load_dataset_from_url(dataset_url)
                    st.session_state.loaded_local_file_key = None
                    _reset_preprocessing_state()
                except DatasetValidationError as error:
                    st.session_state.dataset_result = None
                    _reset_preprocessing_state()
                    st.error(str(error))

    dataset_result = st.session_state.dataset_result
    if dataset_result is None:
        st.info(
            "После загрузки файла здесь появятся размер датасета, названия колонок "
            "и первые строки таблицы."
        )
        return

    _render_dataset_preview(dataset_result)
    _render_preprocessing_section(dataset_result)
    preprocessing_result = st.session_state.get("preprocessing_result")
    if preprocessing_result is not None:
        _render_split_section(preprocessing_result)
    split_result = st.session_state.get("split_result")
    if split_result is not None:
        _render_vectorization_section(split_result)
    vectorization_result = st.session_state.get("vectorization_result")
    if split_result is not None and vectorization_result is not None:
        _render_training_section(split_result, vectorization_result)
    training_result = st.session_state.get("training_result")
    if (
        split_result is not None
        and vectorization_result is not None
        and training_result is not None
    ):
        _render_evaluation_section(
            split_result,
            vectorization_result,
            training_result,
        )
    evaluation_result = st.session_state.get("evaluation_result")
    if (
        split_result is not None
        and vectorization_result is not None
        and training_result is not None
        and evaluation_result is not None
    ):
        _render_detailed_evaluation_section(
            split_result,
            vectorization_result,
            training_result,
            evaluation_result,
        )


def render_main_page() -> None:
    """Отрисовывает стартовую страницу интерфейса проекта."""
    import streamlit as st

    PROJECT_PATHS.ensure_directories()

    st.set_page_config(
        page_title="isnews",
        layout="wide",
    )

    st.title("Интеллектуальный сервис классификации новостей")
    st.caption(
        "Стартовый каркас проекта ВКР. На следующих этапах сюда будет добавлена "
        "загрузка датасетов, обучение моделей, сохранение артефактов и инференс."
    )

    st.subheader("Назначение системы")
    st.write(
        "Приложение предназначено для экспериментов с моделями классификации "
        "новостных публикаций и для демонстрации полного жизненного цикла: "
        "от загрузки данных до применения сохраненной модели."
    )

    st.subheader("Планируемые возможности")
    st.markdown(
        """
        - загрузка датасета из локального файла или по прямой ссылке;
        - предобработка и анализ новостных текстов;
        - обучение нескольких моделей;
        - сохранение и загрузка обученных моделей;
        - классификация новых новостей;
        - сравнение метрик качества.
        """
    )

    _render_dataset_loading_section()

    st.subheader("Базовые директории проекта")
    st.code(
        "\n".join(
            [
                f"Корень проекта: {PROJECT_PATHS.root}",
                f"Каталог документации: {PROJECT_PATHS.docs_dir}",
                f"Каталог данных: {PROJECT_PATHS.data_dir}",
                f"Каталог исходных датасетов: {PROJECT_PATHS.raw_data_dir}",
                f"Каталог очищенных датасетов: {PROJECT_PATHS.processed_data_dir}",
                f"Каталог выборок train/validation/test: {PROJECT_PATHS.split_data_dir}",
                f"Каталог матриц признаков: {PROJECT_PATHS.feature_data_dir}",
                f"Каталог моделей: {PROJECT_PATHS.models_dir}",
                f"Каталог векторизаторов: {PROJECT_PATHS.vectorizers_dir}",
                f"Каталог классификаторов: {PROJECT_PATHS.classifiers_dir}",
                f"Каталог ноутбуков: {PROJECT_PATHS.notebooks_dir}",
                f"Каталог отчетов: {PROJECT_PATHS.reports_dir}",
                f"Каталог JSON-сводок: {PROJECT_PATHS.dataset_reports_dir}",
                f"Каталог отчетов предобработки: {PROJECT_PATHS.preprocessing_reports_dir}",
                f"Каталог отчетов по сплитам: {PROJECT_PATHS.split_reports_dir}",
                f"Каталог отчетов по векторизации: {PROJECT_PATHS.vectorization_reports_dir}",
                f"Каталог отчетов по обучению: {PROJECT_PATHS.training_reports_dir}",
                f"Каталог отчетов по метрикам: {PROJECT_PATHS.metrics_reports_dir}",
                f"Каталог подробных отчетов: {PROJECT_PATHS.detailed_metrics_reports_dir}",
            ]
        ),
        language="text",
    )

    st.info(
        "Текущая версия приложения является стартовой. "
        "Функциональные вкладки будут добавляться постепенно, отдельными коммитами."
    )
