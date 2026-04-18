"""Минимальный пользовательский интерфейс демонстрационного приложения."""

from __future__ import annotations

import pandas as pd

from src.isnews.batch_text_inference import (
    BatchTextInferenceError,
    BatchTextInferenceResult,
    predict_batch_news,
)
from src.isnews.batch_inference_evaluation import (
    BatchInferenceEvaluationError,
    BatchInferenceEvaluationResult,
    evaluate_batch_inference,
)
from src.isnews.confusion_heatmap_export import (
    ConfusionHeatmapExportError,
    ConfusionHeatmapExportResult,
    export_confusion_heatmaps,
)
from src.isnews.batch_error_analysis import (
    BatchErrorAnalysisError,
    BatchErrorAnalysisResult,
    analyze_batch_errors,
)
from src.isnews.config import PROJECT_PATHS
from src.isnews.data_loading import DatasetLoadResult, DatasetValidationError
from src.isnews.detailed_model_evaluation import (
    DetailedModelEvaluationError,
    DetailedModelEvaluationResult,
    evaluate_model_in_detail,
)
from src.isnews.data_loading import load_dataset_from_uploaded_bytes, load_dataset_from_url
from src.isnews.dataset_split import DatasetSplitError, DatasetSplitResult, split_dataset
from src.isnews.experiment_registry import (
    ExperimentRegistryError,
    ExperimentRegistryResult,
    export_experiment_registry,
)
from src.isnews.html_report_export import (
    HtmlReportExportError,
    HtmlReportExportResult,
    export_session_html_report,
)
from src.isnews.docx_report_export import (
    DocxReportExportError,
    DocxReportExportResult,
    export_session_docx_report,
)
from src.isnews.logistic_regression_training import (
    LogisticRegressionTrainingError,
    LogisticRegressionTrainingResult,
    train_logistic_regression,
)
from src.isnews.multinomial_nb_training import (
    MultinomialNBTrainingError,
    MultinomialNBTrainingResult,
    train_multinomial_nb,
)
from src.isnews.model_evaluation import (
    ModelEvaluationError,
    ModelEvaluationResult,
    evaluate_trained_model,
)
from src.isnews.model_comparison import (
    ModelComparisonError,
    ModelComparisonResult,
    compare_trained_models,
)
from src.isnews.markdown_report_export import (
    MarkdownReportExportError,
    MarkdownReportExportResult,
    export_session_markdown_report,
)
from src.isnews.prediction_confidence_analysis import (
    PredictionConfidenceAnalysisError,
    PredictionConfidenceAnalysisResult,
    analyze_prediction_confidence,
)
from src.isnews.plot_export import (
    PlotExportError,
    PlotExportResult,
    export_plots,
)
from src.isnews.saved_artifacts_loading import (
    SavedArtifactsLoadingError,
    SavedArtifactsLoadingResult,
    load_saved_artifacts,
)
from src.isnews.transformers_artifacts_loading import (
    TransformersArtifactsLoadingError,
    TransformersArtifactsLoadingResult,
    load_transformers_artifacts,
)
from src.isnews.single_text_inference import (
    SingleTextInferenceError,
    SingleTextInferenceResult,
    predict_single_news,
)
from src.isnews.thesis_tables_export import (
    ThesisTablesExportError,
    ThesisTablesExportResult,
    export_thesis_tables,
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
    training_result: LogisticRegressionTrainingResult | MultinomialNBTrainingResult,
) -> None:
    """Показывает результат обучения базовой модели и ключевые параметры классификатора."""
    import streamlit as st

    training_report = training_result.report
    model_name = getattr(training_report, "model_name", type(training_result.model).__name__)

    st.success(
        f"Базовая модель `{model_name}` обучена и сохранена."
    )

    metric_column_1, metric_column_2, metric_column_3, metric_column_4 = st.columns(4)
    metric_column_1.metric("Классов", len(training_report.class_labels))
    metric_column_2.metric("Время обучения, сек", training_report.training_seconds)
    if hasattr(training_report, "coefficient_shape"):
        metric_column_3.metric(
            "Размер coef_",
            f"{training_report.coefficient_shape[0]} x {training_report.coefficient_shape[1]}",
        )
    else:
        metric_column_3.metric("Признаков", training_report.feature_count)
    if hasattr(training_report, "iterations_per_class"):
        metric_column_4.metric(
            "Итераций",
            ", ".join(str(value) for value in training_report.iterations_per_class),
        )
    else:
        metric_column_4.metric("Alpha", training_report.alpha)

    for warning_message in training_report.warning_messages:
        st.warning(warning_message)

    details = [
        f"- тип модели: `{model_name}`;",
        f"- классы модели: `{', '.join(training_report.class_labels)}`;",
        f"- сохраненная модель: `{training_result.paths.model_path}`;",
        f"- JSON-отчет: `{training_result.paths.report_path}`.",
    ]
    if hasattr(training_report, "intercept_shape"):
        details.insert(2, f"- форма `intercept_`: `{training_report.intercept_shape}`;")
    if hasattr(training_report, "class_log_prior_shape"):
        details.insert(2, f"- форма `class_log_prior_`: `{training_report.class_log_prior_shape}`;")
    st.markdown("\n".join(details))


def _render_loaded_artifacts_preview(
    loaded_artifacts_result: SavedArtifactsLoadingResult,
) -> None:
    """Показывает результат загрузки сохраненных артефактов модели и векторизатора."""
    import streamlit as st

    loading_report = loaded_artifacts_result.report

    st.success(
        "Сохраненные артефакты успешно загружены и проверены на совместимость."
    )

    metric_column_1, metric_column_2, metric_column_3, metric_column_4 = st.columns(4)
    metric_column_1.metric("Классов", loading_report.class_count)
    metric_column_2.metric("Размер словаря", loading_report.vocabulary_size)
    metric_column_3.metric("Признаков в модели", loading_report.feature_count)
    metric_column_4.metric(
        "Размер coef_",
        f"{loading_report.coefficient_shape[0]} x {loading_report.coefficient_shape[1]}",
    )

    for warning_message in loading_report.warning_messages:
        st.warning(warning_message)

    st.markdown(
        "\n".join(
            [
                f"- тип модели: `{loading_report.model_type}`;",
                f"- тип векторизатора: `{loading_report.vectorizer_type}`;",
                f"- классы модели: `{', '.join(loading_report.class_labels)}`;",
                f"- загруженная модель: `{loaded_artifacts_result.paths.model_path}`;",
                f"- загруженный векторизатор: `{loaded_artifacts_result.paths.vectorizer_path}`;",
                f"- JSON-отчет о загрузке: `{loaded_artifacts_result.paths.report_path}`.",
            ]
        )
    )


def _render_loaded_transformers_artifacts_preview(
    loaded_transformers_result: TransformersArtifactsLoadingResult,
) -> None:
    """Показывает результат загрузки сохраненной transformers-модели и токенизатора."""
    import streamlit as st

    loading_report = loaded_transformers_result.report

    st.success("Локальные transformers-артефакты успешно загружены.")

    metric_column_1, metric_column_2, metric_column_3, metric_column_4 = st.columns(4)
    metric_column_1.metric("Классов", loading_report.num_labels)
    metric_column_2.metric("Размер словаря", loading_report.vocabulary_size)
    metric_column_3.metric("Позиции", loading_report.max_position_embeddings)
    metric_column_4.metric(
        "State dict",
        "да" if loading_report.state_dict_loaded else "нет",
    )

    for warning_message in loading_report.warning_messages:
        st.warning(warning_message)

    id2label_text = ", ".join(
        f"{label_id}:{label_name}"
        for label_id, label_name in loading_report.id2label.items()
    ) or "не задан"
    st.markdown(
        "\n".join(
            [
                f"- тип модели: `{loading_report.model_type}`;",
                f"- тип токенизатора: `{loading_report.tokenizer_type}`;",
                f"- тип конфигурации: `{loading_report.config_model_type}`;",
                f"- базовая модель: `{loading_report.base_model_name}`;",
                f"- `id2label`: `{id2label_text}`;",
                f"- каталог модели: `{loaded_transformers_result.paths.model_directory_path}`;",
                f"- каталог токенизатора: `{loaded_transformers_result.paths.tokenizer_directory_path}`;",
                f"- файл `state_dict`: `{loaded_transformers_result.paths.state_dict_path}`;",
                f"- JSON-отчет о загрузке: `{loaded_transformers_result.paths.report_path}`.",
            ]
        )
    )


def _render_single_inference_preview(
    inference_result: SingleTextInferenceResult,
) -> None:
    """Показывает результат классификации одной новости и вероятности по классам."""
    import streamlit as st

    inference_report = inference_result.report
    probabilities_table = pd.DataFrame(
        [
            {
                "Класс": item.label,
                "Вероятность": item.probability,
            }
            for item in inference_report.class_probabilities
        ]
    )

    st.success("Текст новости успешно классифицирован.")

    metric_column_1, metric_column_2, metric_column_3 = st.columns(3)
    metric_column_1.metric("Предсказанный класс", inference_report.predicted_label)
    metric_column_2.metric(
        "Вероятность класса",
        inference_report.predicted_probability,
    )
    metric_column_3.metric("Длина текста, символов", inference_report.text_length_chars)

    for warning_message in inference_report.warning_messages:
        st.warning(warning_message)

    st.markdown(
        "\n".join(
            [
                f"- источник модели: `{inference_report.source_name}`;",
                f"- тип модели: `{inference_report.model_type}`;",
                f"- тип векторизатора: `{inference_report.vectorizer_type}`;",
                f"- JSON-отчет по инференсу: `{inference_result.paths.report_path}`.",
            ]
        )
    )

    st.write("Очищенный текст, поданный в модель:")
    st.code(inference_result.cleaned_text, language="text")

    st.write("Вероятности по классам:")
    st.dataframe(probabilities_table, use_container_width=True)


def _render_batch_inference_preview(
    batch_inference_result: BatchTextInferenceResult,
) -> None:
    """Показывает результат пакетной классификации CSV и предпросмотр таблицы."""
    import streamlit as st

    batch_report = batch_inference_result.report

    st.success("CSV-файл успешно классифицирован пакетно.")

    metric_column_1, metric_column_2, metric_column_3 = st.columns(3)
    metric_column_1.metric("Всего строк", batch_report.total_rows)
    metric_column_2.metric("Классифицировано", batch_report.predicted_rows)
    metric_column_3.metric("Пропущено", batch_report.skipped_empty_rows)

    for warning_message in batch_report.warning_messages:
        st.warning(warning_message)

    st.markdown(
        "\n".join(
            [
                f"- источник модели: `{batch_report.source_name}`;",
                f"- колонка с текстом: `{batch_report.text_column}`;",
                f"- CSV с предсказаниями: `{batch_inference_result.paths.predictions_path}`;",
                f"- JSON-отчет: `{batch_inference_result.paths.report_path}`.",
            ]
        )
    )

    st.write("Первые 10 строк таблицы предсказаний:")
    st.dataframe(batch_inference_result.dataframe.head(10), use_container_width=True)


def _render_batch_inference_evaluation_preview(
    evaluation_result: BatchInferenceEvaluationResult,
) -> None:
    """Показывает метрики и матрицу ошибок для размеченного пакетного инференса."""
    import streamlit as st

    evaluation_report = evaluation_result.report

    st.success("Качество пакетного инференса на размеченном CSV рассчитано.")

    metric_column_1, metric_column_2, metric_column_3, metric_column_4 = st.columns(4)
    metric_column_1.metric("Accuracy", evaluation_report.accuracy)
    metric_column_2.metric("Precision macro", evaluation_report.precision_macro)
    metric_column_3.metric("Recall macro", evaluation_report.recall_macro)
    metric_column_4.metric("F1 macro", evaluation_report.f1_macro)

    for warning_message in evaluation_report.warning_messages:
        st.warning(warning_message)

    st.markdown(
        "\n".join(
            [
                f"- колонка истинного класса: `{evaluation_report.label_column}`;",
                f"- оценено строк: `{evaluation_report.evaluated_rows}`;",
                f"- пропущено строк: `{evaluation_report.skipped_rows_without_label}`;",
                f"- JSON-отчет: `{evaluation_result.paths.report_path}`;",
                f"- CSV матрицы ошибок: `{evaluation_result.paths.confusion_matrix_path}`.",
            ]
        )
    )

    st.write("Матрица ошибок:")
    st.dataframe(evaluation_result.confusion_matrix_dataframe, use_container_width=True)


def _render_prediction_confidence_preview(
    confidence_result: PredictionConfidenceAnalysisResult,
) -> None:
    """Показывает top-N самых уверенных и самых неуверенных предсказаний модели."""
    import streamlit as st

    confidence_report = confidence_result.report

    st.success("Анализ уверенности предсказаний успешно сформирован.")

    metric_column_1, metric_column_2, metric_column_3, metric_column_4 = st.columns(4)
    metric_column_1.metric("Проанализировано строк", confidence_report.analyzed_rows)
    metric_column_2.metric("Top N", confidence_report.top_n)
    metric_column_3.metric("Макс. вероятность", confidence_report.highest_probability)
    metric_column_4.metric("Мин. вероятность", confidence_report.lowest_probability)

    for warning_message in confidence_report.warning_messages:
        st.warning(warning_message)

    st.markdown(
        "\n".join(
            [
                f"- источник предсказаний: `{confidence_report.source_name}`;",
                f"- CSV уверенных предсказаний: `{confidence_result.paths.confident_predictions_path}`;",
                f"- CSV неуверенных предсказаний: `{confidence_result.paths.uncertain_predictions_path}`;",
                f"- JSON-отчет: `{confidence_result.paths.report_path}`.",
            ]
        )
    )

    confident_column, uncertain_column = st.columns(2)
    with confident_column:
        st.write("Самые уверенные предсказания:")
        st.dataframe(confidence_result.confident_dataframe, use_container_width=True)
    with uncertain_column:
        st.write("Самые неуверенные предсказания:")
        st.dataframe(confidence_result.uncertain_dataframe, use_container_width=True)


def _render_batch_error_analysis_preview(
    error_analysis_result: BatchErrorAnalysisResult,
) -> None:
    """Показывает таблицу неверно классифицированных строк."""
    import streamlit as st

    error_report = error_analysis_result.report

    st.success("Анализ ошибок по размеченному CSV успешно сформирован.")

    metric_column_1, metric_column_2, metric_column_3 = st.columns(3)
    metric_column_1.metric("Проверено строк", error_report.analyzed_rows)
    metric_column_2.metric("Ошибок", error_report.misclassified_rows)
    metric_column_3.metric("Доля ошибок", error_report.error_rate)

    for warning_message in error_report.warning_messages:
        st.warning(warning_message)

    st.markdown(
        "\n".join(
            [
                f"- колонка истинного класса: `{error_report.label_column}`;",
                f"- CSV с ошибками: `{error_analysis_result.paths.misclassified_rows_path}`;",
                f"- JSON-отчет: `{error_analysis_result.paths.report_path}`.",
            ]
        )
    )

    if error_analysis_result.misclassified_dataframe.empty:
        st.info("Ошибочных классификаций не найдено.")
        return

    st.write("Неверно классифицированные строки:")
    st.dataframe(error_analysis_result.misclassified_dataframe, use_container_width=True)


def _render_experiment_registry_preview(
    registry_result: ExperimentRegistryResult,
) -> None:
    """Показывает сводную таблицу по найденным экспериментам и пути к экспортам."""
    import streamlit as st

    registry_dataframe = registry_result.dataframe

    st.success("Сводный реестр экспериментов успешно сформирован.")

    metric_column_1, metric_column_2, metric_column_3 = st.columns(3)
    metric_column_1.metric("Всего записей", len(registry_dataframe))
    metric_column_2.metric(
        "Запусков обучения",
        int((registry_dataframe.get("record_type", pd.Series(dtype="string")) == "training_run").sum()),
    )
    metric_column_3.metric(
        "Пакетных оценок",
        int((registry_dataframe.get("record_type", pd.Series(dtype="string")) == "batch_evaluation").sum()),
    )

    st.markdown(
        "\n".join(
            [
                f"- CSV-реестр: `{registry_result.paths.csv_path}`;",
                f"- JSON-реестр: `{registry_result.paths.json_path}`.",
            ]
        )
    )

    if registry_dataframe.empty:
        st.info(
            "Пока не найдено сохраненных отчетов экспериментов. После обучения модели и расчета метрик здесь появятся записи."
        )
        return

    st.write("Первые 20 записей реестра:")
    st.dataframe(registry_dataframe.head(20), use_container_width=True)


def _render_html_report_preview(html_report_result: HtmlReportExportResult) -> None:
    """Показывает сведения о сформированном HTML-отчете."""
    import streamlit as st

    st.success("Краткий HTML-отчет успешно сформирован.")
    st.markdown(
        "\n".join(
            [
                f"- HTML-файл: `{html_report_result.report_path}`;",
                f"- включенные разделы: `{', '.join(html_report_result.generated_sections)}`.",
            ]
        )
    )


def _render_docx_report_preview(docx_report_result: DocxReportExportResult) -> None:
    """Показывает сведения о сформированном DOCX-отчете."""
    import streamlit as st

    st.success("DOCX-отчет успешно сформирован.")
    st.markdown(
        "\n".join(
            [
                f"- DOCX-файл: `{docx_report_result.report_path}`;",
                f"- включенные разделы: `{', '.join(docx_report_result.generated_sections)}`.",
            ]
        )
    )


def _render_markdown_report_preview(
    markdown_report_result: MarkdownReportExportResult,
) -> None:
    """Показывает сведения о сформированном Markdown-отчете."""
    import streamlit as st

    st.success("Markdown-отчет успешно сформирован.")
    st.markdown(
        "\n".join(
            [
                f"- Markdown-файл: `{markdown_report_result.report_path}`;",
                f"- включенные разделы: `{', '.join(markdown_report_result.generated_sections)}`.",
            ]
        )
    )


def _render_thesis_tables_preview(
    thesis_tables_result: ThesisTablesExportResult,
) -> None:
    """Показывает сведения о выгруженных CSV-таблицах для ВКР."""
    import streamlit as st

    st.success("CSV-таблицы для ВКР успешно сформированы.")
    rows = [
        f"- выгруженные таблицы: `{', '.join(thesis_tables_result.exported_table_names)}`;",
        f"- manifest: `{thesis_tables_result.paths.manifest_path}`;",
    ]
    if thesis_tables_result.paths.metrics_table_path is not None:
        rows.append(f"- таблица метрик: `{thesis_tables_result.paths.metrics_table_path}`;")
    if thesis_tables_result.paths.comparison_table_path is not None:
        rows.append(f"- таблица сравнения моделей: `{thesis_tables_result.paths.comparison_table_path}`;")
    if thesis_tables_result.paths.error_table_path is not None:
        rows.append(f"- таблица ошибок: `{thesis_tables_result.paths.error_table_path}`;")
    st.markdown("\n".join(rows))


def _render_plot_export_preview(plot_export_result: PlotExportResult) -> None:
    """Показывает сведения о выгруженных PNG-графиках."""
    import streamlit as st

    st.success("PNG-графики успешно сформированы.")
    rows = [
        f"- выгруженные графики: `{', '.join(plot_export_result.exported_plot_names)}`;",
        f"- manifest: `{plot_export_result.paths.manifest_path}`;",
    ]
    if plot_export_result.paths.metrics_plot_path is not None:
        rows.append(f"- график метрик: `{plot_export_result.paths.metrics_plot_path}`;")
    if plot_export_result.paths.comparison_plot_path is not None:
        rows.append(f"- график сравнения моделей: `{plot_export_result.paths.comparison_plot_path}`;")
    st.markdown("\n".join(rows))


def _render_confusion_heatmap_preview(
    heatmap_result: ConfusionHeatmapExportResult,
) -> None:
    """Показывает сведения о выгруженных тепловых картах матриц ошибок."""
    import streamlit as st

    st.success("PNG-тепловые карты матриц ошибок успешно сформированы.")
    rows = [
        f"- выгруженные тепловые карты: `{', '.join(heatmap_result.exported_heatmap_names)}`;",
        f"- manifest: `{heatmap_result.paths.manifest_path}`;",
    ]
    if heatmap_result.paths.validation_heatmap_path is not None:
        rows.append(f"- validation heatmap: `{heatmap_result.paths.validation_heatmap_path}`;")
    if heatmap_result.paths.test_heatmap_path is not None:
        rows.append(f"- test heatmap: `{heatmap_result.paths.test_heatmap_path}`;")
    if heatmap_result.paths.batch_heatmap_path is not None:
        rows.append(f"- batch heatmap: `{heatmap_result.paths.batch_heatmap_path}`;")
    st.markdown("\n".join(rows))


def _render_model_comparison_preview(
    comparison_result: ModelComparisonResult,
) -> None:
    """Показывает сводную таблицу сравнения обученных моделей."""
    import streamlit as st

    comparison_dataframe = comparison_result.dataframe

    st.success("Сравнение обученных моделей успешно сформировано.")

    metric_column_1, metric_column_2, metric_column_3 = st.columns(3)
    metric_column_1.metric("Моделей в таблице", len(comparison_dataframe))
    metric_column_2.metric(
        "Лучшая модель",
        comparison_result.best_model_name or "нет данных",
    )
    metric_column_3.metric(
        "Лучший Validation Accuracy",
        comparison_dataframe.iloc[0]["validation_accuracy"] if not comparison_dataframe.empty else "нет данных",
    )

    st.markdown(
        "\n".join(
            [
                f"- CSV-сравнение: `{comparison_result.paths.csv_path}`;",
                f"- JSON-сравнение: `{comparison_result.paths.json_path}`.",
            ]
        )
    )

    if comparison_dataframe.empty:
        st.info(
            "Пока не найдено сохраненных обучающих запусков. После обучения моделей здесь появится таблица сравнения."
        )
        return

    st.write("Сравнение моделей по основным метрикам:")
    st.dataframe(comparison_dataframe, use_container_width=True)


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


def _reset_saved_artifacts_state() -> None:
    """Сбрасывает состояние загрузки сохраненных артефактов при смене выбранных файлов."""
    import streamlit as st

    st.session_state.loaded_artifacts_result = None
    st.session_state.loaded_artifacts_selection_key = None


def _reset_transformers_artifacts_state() -> None:
    """Сбрасывает состояние загрузки нейросетевых артефактов при смене выбранных путей."""
    import streamlit as st

    st.session_state.loaded_transformers_artifacts_result = None
    st.session_state.loaded_transformers_artifacts_selection_key = None


def _reset_single_inference_state() -> None:
    """Сбрасывает состояние одиночного инференса при смене текста или источника модели."""
    import streamlit as st

    st.session_state.single_inference_result = None
    st.session_state.single_inference_request_key = None


def _reset_batch_inference_state() -> None:
    """Сбрасывает состояние пакетного инференса при смене файла или источника модели."""
    import streamlit as st

    st.session_state.batch_inference_result = None
    st.session_state.batch_inference_request_key = None
    _reset_prediction_confidence_state()
    _reset_batch_inference_evaluation_state()


def _reset_prediction_confidence_state() -> None:
    """Сбрасывает анализ уверенности при смене результатов пакетного инференса."""
    import streamlit as st

    st.session_state.batch_confidence_analysis_result = None
    st.session_state.batch_confidence_analysis_source_key = None


def _reset_batch_inference_evaluation_state() -> None:
    """Сбрасывает состояние оценки пакетного инференса при смене CSV с предсказаниями."""
    import streamlit as st

    st.session_state.batch_inference_evaluation_result = None
    st.session_state.batch_inference_evaluation_source_key = None
    _reset_batch_error_analysis_state()


def _reset_batch_error_analysis_state() -> None:
    """Сбрасывает анализ ошибочных классификаций при смене оценки."""
    import streamlit as st

    st.session_state.batch_error_analysis_result = None
    st.session_state.batch_error_analysis_source_key = None


def _get_available_inference_sources() -> dict[str, dict[str, object]]:
    """Собирает доступные источники модели для одиночного и пакетного инференса."""
    import streamlit as st

    training_result = st.session_state.get("training_result")
    vectorization_result = st.session_state.get("vectorization_result")
    loaded_artifacts_result = st.session_state.get("loaded_artifacts_result")

    available_sources: dict[str, dict[str, object]] = {}
    if (
        training_result is not None
        and vectorization_result is not None
        and hasattr(training_result.model, "predict_proba")
        and hasattr(training_result.model, "classes_")
    ):
        available_sources["Модель текущей сессии"] = {
            "model": training_result.model,
            "vectorizer": vectorization_result.vectorizer,
            "source_name": f"session::{training_result.paths.model_path.name}",
        }
    if loaded_artifacts_result is not None:
        available_sources["Загруженные артефакты"] = {
            "model": loaded_artifacts_result.model,
            "vectorizer": loaded_artifacts_result.vectorizer,
            "source_name": f"loaded::{loaded_artifacts_result.paths.model_path.name}",
        }
    return available_sources


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

    selected_model_name = st.radio(
        "Выберите базовую модель",
        options=["Logistic Regression", "MultinomialNB"],
        horizontal=True,
        key="training_model_choice",
    )
    current_source_key = f"{vectorization_result.paths.vectorizer_path}|{selected_model_name}"
    if "training_result" not in st.session_state:
        st.session_state.training_result = None
    if "training_source_key" not in st.session_state:
        st.session_state.training_source_key = None

    if st.session_state.training_source_key != current_source_key:
        st.session_state.training_result = None
        st.session_state.training_source_key = current_source_key

    st.subheader("Обучение базовой модели")
    st.caption(
        "На этом этапе можно обучить и сравнить две базовые модели на одной и той же "
        "`train`-матрице TF-IDF признаков: `Logistic Regression` и `MultinomialNB`."
    )
    st.markdown(f"- выбранная модель: `{selected_model_name}`;")

    if st.button(
        "Обучить базовую модель и сохранить классификатор",
        use_container_width=True,
    ):
        try:
            _reset_training_state()
            if selected_model_name == "Logistic Regression":
                st.session_state.training_result = train_logistic_regression(
                    split_result,
                    vectorization_result,
                )
            else:
                st.session_state.training_result = train_multinomial_nb(
                    split_result,
                    vectorization_result,
                )
            st.session_state.training_source_key = current_source_key
        except (LogisticRegressionTrainingError, MultinomialNBTrainingError) as error:
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


def _render_saved_artifacts_loading_section() -> None:
    """Отрисовывает блок загрузки сохраненных `.joblib`-артефактов без переобучения."""
    import streamlit as st

    if "loaded_artifacts_result" not in st.session_state:
        st.session_state.loaded_artifacts_result = None
    if "loaded_artifacts_selection_key" not in st.session_state:
        st.session_state.loaded_artifacts_selection_key = None

    model_paths = sorted(PROJECT_PATHS.classifiers_dir.glob("*.joblib"))
    vectorizer_paths = sorted(PROJECT_PATHS.vectorizers_dir.glob("*.joblib"))

    st.subheader("Загрузка сохраненных артефактов")
    st.caption(
        "Этот блок позволяет без переобучения выбрать ранее сохраненные `.joblib`-файлы "
        "модели и TF-IDF-векторизатора, загрузить их и проверить совместимость по числу признаков."
    )

    if not model_paths or not vectorizer_paths:
        _reset_saved_artifacts_state()
        st.info(
            "В каталогах `models/classifiers` и `models/vectorizers` пока нет полного набора "
            "артефактов. Сначала обучите модель или добавьте сохраненные `.joblib`-файлы в проект."
        )
        return

    model_options = {path.name: path for path in model_paths}
    vectorizer_options = {path.name: path for path in vectorizer_paths}

    selection_column_1, selection_column_2 = st.columns(2)
    with selection_column_1:
        selected_model_name = st.selectbox(
            "Файл классификатора",
            options=list(model_options.keys()),
            key="saved_model_filename",
        )
    with selection_column_2:
        selected_vectorizer_name = st.selectbox(
            "Файл векторизатора",
            options=list(vectorizer_options.keys()),
            key="saved_vectorizer_filename",
        )

    current_selection_key = f"{selected_model_name}|{selected_vectorizer_name}"
    if st.session_state.loaded_artifacts_selection_key != current_selection_key:
        st.session_state.loaded_artifacts_result = None
        st.session_state.loaded_artifacts_selection_key = current_selection_key

    st.markdown(
        "\n".join(
            [
                f"- выбранная модель: `{model_options[selected_model_name]}`;",
                f"- выбранный векторизатор: `{vectorizer_options[selected_vectorizer_name]}`;",
            ]
        )
    )

    if st.button(
        "Загрузить модель и векторизатор из файлов",
        use_container_width=True,
    ):
        try:
            _reset_saved_artifacts_state()
            st.session_state.loaded_artifacts_result = load_saved_artifacts(
                model_options[selected_model_name],
                vectorizer_options[selected_vectorizer_name],
            )
            st.session_state.loaded_artifacts_selection_key = current_selection_key
        except SavedArtifactsLoadingError as error:
            st.session_state.loaded_artifacts_result = None
            st.session_state.loaded_artifacts_selection_key = current_selection_key
            st.error(str(error))

    loaded_artifacts_result = st.session_state.loaded_artifacts_result
    if loaded_artifacts_result is None:
        st.info(
            "После загрузки здесь появятся сведения о типе артефактов, количестве классов, "
            "размере словаря и пути к JSON-отчету о проверке."
        )
        return

    _render_loaded_artifacts_preview(loaded_artifacts_result)


def _render_transformers_artifacts_loading_section() -> None:
    """Отрисовывает блок загрузки локальной transformers-модели без переобучения."""
    import streamlit as st

    if "loaded_transformers_artifacts_result" not in st.session_state:
        st.session_state.loaded_transformers_artifacts_result = None
    if "loaded_transformers_artifacts_selection_key" not in st.session_state:
        st.session_state.loaded_transformers_artifacts_selection_key = None

    model_directories = sorted(
        path
        for path in PROJECT_PATHS.models_dir.iterdir()
        if path.is_dir() and (path / "config.json").exists()
    )
    tokenizer_directories = sorted(
        path
        for path in PROJECT_PATHS.models_dir.iterdir()
        if path.is_dir() and (path / "tokenizer_config.json").exists()
    )
    state_dict_paths = sorted(PROJECT_PATHS.models_dir.glob("model*.pt"))

    st.subheader("Загрузка transformers-модели")
    st.caption(
        "Этот блок позволяет выбрать локальные каталоги `save_pretrained` для модели и "
        "токенизатора, а также при необходимости подключить файл весов `model*.pt`."
    )

    if not model_directories or not tokenizer_directories:
        _reset_transformers_artifacts_state()
        st.info(
            "В каталоге `models` пока нет полного набора локальных transformers-артефактов. "
            "Сначала выгрузите их из Google Colab или добавьте в проект."
        )
        return

    model_options = {path.name: path for path in model_directories}
    tokenizer_options = {path.name: path for path in tokenizer_directories}
    state_dict_options = {"Без файла .pt": None}
    state_dict_options.update({path.name: path for path in state_dict_paths})

    selection_column_1, selection_column_2, selection_column_3 = st.columns(3)
    with selection_column_1:
        selected_model_name = st.selectbox(
            "Каталог модели",
            options=list(model_options.keys()),
            key="saved_transformers_model_directory",
        )
    with selection_column_2:
        selected_tokenizer_name = st.selectbox(
            "Каталог токенизатора",
            options=list(tokenizer_options.keys()),
            key="saved_transformers_tokenizer_directory",
        )
    with selection_column_3:
        selected_state_dict_name = st.selectbox(
            "Файл весов",
            options=list(state_dict_options.keys()),
            key="saved_transformers_state_dict",
        )

    current_selection_key = (
        f"{selected_model_name}|{selected_tokenizer_name}|{selected_state_dict_name}"
    )
    if (
        st.session_state.loaded_transformers_artifacts_selection_key
        != current_selection_key
    ):
        st.session_state.loaded_transformers_artifacts_result = None
        st.session_state.loaded_transformers_artifacts_selection_key = (
            current_selection_key
        )

    selected_state_dict_path = state_dict_options[selected_state_dict_name]
    st.markdown(
        "\n".join(
            [
                f"- выбранный каталог модели: `{model_options[selected_model_name]}`;",
                f"- выбранный каталог токенизатора: `{tokenizer_options[selected_tokenizer_name]}`;",
                f"- выбранный файл весов: `{selected_state_dict_path}`;",
            ]
        )
    )

    if st.button(
        "Загрузить transformers-артефакты",
        use_container_width=True,
    ):
        try:
            _reset_transformers_artifacts_state()
            st.session_state.loaded_transformers_artifacts_result = (
                load_transformers_artifacts(
                    model_options[selected_model_name],
                    tokenizer_options[selected_tokenizer_name],
                    state_dict_path=selected_state_dict_path,
                )
            )
            st.session_state.loaded_transformers_artifacts_selection_key = (
                current_selection_key
            )
        except TransformersArtifactsLoadingError as error:
            st.session_state.loaded_transformers_artifacts_result = None
            st.session_state.loaded_transformers_artifacts_selection_key = (
                current_selection_key
            )
            st.error(str(error))

    loaded_transformers_result = st.session_state.loaded_transformers_artifacts_result
    if loaded_transformers_result is None:
        st.info(
            "После загрузки здесь появятся сведения о типе модели, размере словаря, "
            "числе классов и пути к JSON-отчету."
        )
        return

    _render_loaded_transformers_artifacts_preview(loaded_transformers_result)


def _render_single_inference_section() -> None:
    """Отрисовывает блок классификации одной новости через доступную модель."""
    import streamlit as st

    if "single_inference_result" not in st.session_state:
        st.session_state.single_inference_result = None
    if "single_inference_request_key" not in st.session_state:
        st.session_state.single_inference_request_key = None

    available_sources = _get_available_inference_sources()

    st.subheader("Инференс одной новости")
    st.caption(
        "На этом этапе можно вставить текст отдельной новости и получить предсказанный "
        "класс вместе с вероятностями по всем классам через обученную или загруженную модель."
    )

    if not available_sources:
        _reset_single_inference_state()
        st.info(
            "Для инференса пока нет доступной модели. Сначала обучите модель в текущей "
            "сессии или загрузите сохраненные артефакты из файлов."
        )
        return

    selected_source_label = st.radio(
        "Источник модели",
        options=list(available_sources.keys()),
        horizontal=True,
    )
    news_text = st.text_area(
        "Текст новости для классификации",
        placeholder=(
            "Например: Правительство представило новый пакет мер поддержки "
            "региональной экономики и инвестиций."
        ),
        height=180,
        key="single_inference_text",
    )

    selected_source = available_sources[selected_source_label]
    current_request_key = f"{selected_source['source_name']}|{news_text}"
    if st.session_state.single_inference_request_key != current_request_key:
        st.session_state.single_inference_result = None
        st.session_state.single_inference_request_key = current_request_key

    st.markdown(
        f"- выбранный источник: `{selected_source['source_name']}`;"
    )

    if st.button(
        "Классифицировать новость",
        use_container_width=True,
    ):
        try:
            _reset_single_inference_state()
            st.session_state.single_inference_result = predict_single_news(
                news_text,
                model=selected_source["model"],
                vectorizer=selected_source["vectorizer"],
                source_name=str(selected_source["source_name"]),
            )
            st.session_state.single_inference_request_key = current_request_key
        except SingleTextInferenceError as error:
            st.session_state.single_inference_result = None
            st.session_state.single_inference_request_key = current_request_key
            st.error(str(error))

    inference_result = st.session_state.single_inference_result
    if inference_result is None:
        st.info(
            "После запуска инференса здесь появятся предсказанный класс, вероятность, "
            "очищенный текст и таблица вероятностей по всем классам."
        )
        return

    _render_single_inference_preview(inference_result)


def _render_batch_inference_section() -> None:
    """Отрисовывает блок пакетной классификации новостей из CSV-файла."""
    import streamlit as st

    if "batch_inference_result" not in st.session_state:
        st.session_state.batch_inference_result = None
    if "batch_inference_request_key" not in st.session_state:
        st.session_state.batch_inference_request_key = None

    available_sources = _get_available_inference_sources()

    st.subheader("Пакетный инференс CSV")
    st.caption(
        "На этом этапе можно загрузить CSV с новостями, автоматически найти колонку с "
        "текстом и получить таблицу предсказанных классов и вероятностей по всем строкам."
    )

    if not available_sources:
        _reset_batch_inference_state()
        st.info(
            "Для пакетного инференса пока нет доступной модели. Сначала обучите модель "
            "в текущей сессии или загрузите сохраненные артефакты."
        )
        return

    selected_source_label = st.radio(
        "Источник модели для CSV",
        options=list(available_sources.keys()),
        horizontal=True,
        key="batch_inference_source",
    )
    uploaded_file = st.file_uploader(
        "CSV-файл для пакетной классификации",
        type=["csv"],
        help=(
            "Поддерживаются колонки с текстом вроде `text`, `content`, `title`, "
            "`description`, `текст`."
        ),
        key="batch_inference_file",
    )

    selected_source = available_sources[selected_source_label]
    uploaded_file_key = None
    if uploaded_file is not None:
        uploaded_file_key = f"{uploaded_file.name}|{uploaded_file.size}"
    current_request_key = f"{selected_source['source_name']}|{uploaded_file_key}"
    if st.session_state.batch_inference_request_key != current_request_key:
        st.session_state.batch_inference_result = None
        st.session_state.batch_inference_request_key = current_request_key

    st.markdown(
        f"- выбранный источник: `{selected_source['source_name']}`;"
    )

    if st.button(
        "Классифицировать CSV-файл",
        use_container_width=True,
    ):
        if uploaded_file is None:
            st.warning("Загрузите CSV-файл для пакетной классификации.")
        else:
            try:
                uploaded_dataframe = pd.read_csv(uploaded_file)
                _reset_batch_inference_state()
                st.session_state.batch_inference_result = predict_batch_news(
                    uploaded_dataframe,
                    model=selected_source["model"],
                    vectorizer=selected_source["vectorizer"],
                    source_name=f"{selected_source['source_name']}::{uploaded_file.name}",
                )
                st.session_state.batch_inference_request_key = current_request_key
            except (pd.errors.ParserError, UnicodeDecodeError) as error:
                st.session_state.batch_inference_result = None
                st.session_state.batch_inference_request_key = current_request_key
                st.error(f"Не удалось прочитать CSV-файл: {error}")
            except BatchTextInferenceError as error:
                st.session_state.batch_inference_result = None
                st.session_state.batch_inference_request_key = current_request_key
                st.error(str(error))

    batch_inference_result = st.session_state.batch_inference_result
    if batch_inference_result is None:
        st.info(
            "После запуска пакетного инференса здесь появятся число обработанных строк, "
            "пути к CSV и JSON-отчету, а также предпросмотр таблицы предсказаний."
        )
        return

    _render_batch_inference_preview(batch_inference_result)
    _render_prediction_confidence_section(batch_inference_result)
    _render_batch_inference_evaluation_section(batch_inference_result)


def _render_prediction_confidence_section(
    batch_inference_result: BatchTextInferenceResult,
) -> None:
    """Отрисовывает блок анализа самых уверенных и неуверенных пакетных предсказаний."""
    import streamlit as st

    current_source_key = str(batch_inference_result.paths.predictions_path)
    if "batch_confidence_analysis_result" not in st.session_state:
        st.session_state.batch_confidence_analysis_result = None
    if "batch_confidence_analysis_source_key" not in st.session_state:
        st.session_state.batch_confidence_analysis_source_key = None

    if st.session_state.batch_confidence_analysis_source_key != current_source_key:
        st.session_state.batch_confidence_analysis_result = None
        st.session_state.batch_confidence_analysis_source_key = current_source_key

    st.subheader("Анализ уверенности предсказаний")
    st.caption(
        "На этом этапе можно сохранить top-N самых уверенных и самых неуверенных "
        "предсказаний модели для последующего анализа качества."
    )

    top_n = st.number_input(
        "Сколько строк сохранить в каждой группе",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        key="batch_confidence_top_n",
    )

    if st.button(
        "Сформировать анализ уверенности",
        use_container_width=True,
    ):
        try:
            _reset_prediction_confidence_state()
            st.session_state.batch_confidence_analysis_result = analyze_prediction_confidence(
                batch_inference_result,
                top_n=int(top_n),
            )
            st.session_state.batch_confidence_analysis_source_key = current_source_key
        except PredictionConfidenceAnalysisError as error:
            st.session_state.batch_confidence_analysis_result = None
            st.session_state.batch_confidence_analysis_source_key = current_source_key
            st.error(str(error))

    confidence_result = st.session_state.batch_confidence_analysis_result
    if confidence_result is None:
        st.info(
            "После запуска здесь появятся две таблицы: самые уверенные и самые "
            "неуверенные предсказания модели."
        )
        return

    _render_prediction_confidence_preview(confidence_result)


def _render_batch_inference_evaluation_section(
    batch_inference_result: BatchTextInferenceResult,
) -> None:
    """Отрисовывает блок оценки пакетного инференса на размеченном CSV."""
    import streamlit as st

    current_source_key = str(batch_inference_result.paths.predictions_path)
    if "batch_inference_evaluation_result" not in st.session_state:
        st.session_state.batch_inference_evaluation_result = None
    if "batch_inference_evaluation_source_key" not in st.session_state:
        st.session_state.batch_inference_evaluation_source_key = None

    if st.session_state.batch_inference_evaluation_source_key != current_source_key:
        st.session_state.batch_inference_evaluation_result = None
        st.session_state.batch_inference_evaluation_source_key = current_source_key

    st.subheader("Оценка пакетного инференса")
    st.caption(
        "Если во входном CSV есть колонка истинного класса, на этом этапе можно рассчитать "
        "Accuracy, Precision, Recall, F1-score и построить матрицу ошибок."
    )

    if st.button(
        "Рассчитать метрики для размеченного CSV",
        use_container_width=True,
    ):
        try:
            _reset_batch_inference_evaluation_state()
            st.session_state.batch_inference_evaluation_result = evaluate_batch_inference(
                batch_inference_result
            )
            st.session_state.batch_inference_evaluation_source_key = current_source_key
        except BatchInferenceEvaluationError as error:
            st.session_state.batch_inference_evaluation_result = None
            st.session_state.batch_inference_evaluation_source_key = current_source_key
            st.error(str(error))

    evaluation_result = st.session_state.batch_inference_evaluation_result
    if evaluation_result is None:
        st.info(
            "После расчета здесь появятся метрики качества и матрица ошибок для тех строк, "
            "где во входном CSV есть истинная метка класса."
        )
        return

    _render_batch_inference_evaluation_preview(evaluation_result)
    _render_batch_error_analysis_section(batch_inference_result, evaluation_result)


def _render_batch_error_analysis_section(
    batch_inference_result: BatchTextInferenceResult,
    evaluation_result: BatchInferenceEvaluationResult,
) -> None:
    """Отрисовывает блок сохранения только неверно классифицированных строк."""
    import streamlit as st

    current_source_key = str(evaluation_result.paths.report_path)
    if "batch_error_analysis_result" not in st.session_state:
        st.session_state.batch_error_analysis_result = None
    if "batch_error_analysis_source_key" not in st.session_state:
        st.session_state.batch_error_analysis_source_key = None

    if st.session_state.batch_error_analysis_source_key != current_source_key:
        st.session_state.batch_error_analysis_result = None
        st.session_state.batch_error_analysis_source_key = current_source_key

    st.subheader("Анализ ошибок")
    st.caption(
        "На этом этапе можно сохранить отдельную таблицу только с теми новостями, "
        "которые модель классифицировала неверно."
    )

    if st.button(
        "Сформировать таблицу ошибок",
        use_container_width=True,
    ):
        try:
            _reset_batch_error_analysis_state()
            st.session_state.batch_error_analysis_result = analyze_batch_errors(
                batch_inference_result,
                evaluation_result.report,
            )
            st.session_state.batch_error_analysis_source_key = current_source_key
        except BatchErrorAnalysisError as error:
            st.session_state.batch_error_analysis_result = None
            st.session_state.batch_error_analysis_source_key = current_source_key
            st.error(str(error))

    error_analysis_result = st.session_state.batch_error_analysis_result
    if error_analysis_result is None:
        st.info(
            "После запуска здесь появится таблица только с неверно "
            "классифицированными строками."
        )
        return

    _render_batch_error_analysis_preview(error_analysis_result)


def _render_experiment_registry_section() -> None:
    """Отрисовывает блок экспорта единого реестра экспериментов по проекту."""
    import streamlit as st

    if "experiment_registry_result" not in st.session_state:
        st.session_state.experiment_registry_result = None

    st.subheader("Сводка экспериментов")
    st.caption(
        "На этом этапе можно собрать единый CSV/JSON-реестр по найденным запускам обучения, "
        "оценки модели и пакетного тестирования на размеченных CSV."
    )

    if st.button(
        "Сформировать сводный реестр экспериментов",
        use_container_width=True,
    ):
        try:
            st.session_state.experiment_registry_result = export_experiment_registry()
        except ExperimentRegistryError as error:
            st.session_state.experiment_registry_result = None
            st.error(str(error))

    registry_result = st.session_state.experiment_registry_result
    if registry_result is None:
        st.info(
            "После запуска здесь появится единая таблица по экспериментам, а также пути к экспортированным CSV и JSON."
        )
        return

    _render_experiment_registry_preview(registry_result)


def _render_model_comparison_section() -> None:
    """Отрисовывает блок сравнения нескольких обученных моделей."""
    import streamlit as st

    if "model_comparison_result" not in st.session_state:
        st.session_state.model_comparison_result = None

    st.subheader("Сравнение моделей")
    st.caption(
        "На этом этапе можно построить единую таблицу сравнения по всем найденным обученным "
        "моделям и выбрать лучший запуск по `validation accuracy`."
    )

    if st.button(
        "Сформировать таблицу сравнения моделей",
        use_container_width=True,
    ):
        try:
            st.session_state.model_comparison_result = compare_trained_models()
        except ModelComparisonError as error:
            st.session_state.model_comparison_result = None
            st.error(str(error))

    comparison_result = st.session_state.model_comparison_result
    if comparison_result is None:
        st.info(
            "После запуска здесь появится таблица сравнения всех обученных моделей и пути к CSV/JSON-сводке."
        )
        return

    _render_model_comparison_preview(comparison_result)


def _render_html_report_section() -> None:
    """Отрисовывает блок формирования краткого HTML-отчета по текущей сессии."""
    import streamlit as st

    if "html_report_result" not in st.session_state:
        st.session_state.html_report_result = None

    st.subheader("HTML-отчет")
    st.caption(
        "На этом этапе можно собрать краткий HTML-отчет по текущей сессии: "
        "обучение модели, метрики, сравнение запусков, реестр экспериментов и анализ ошибок."
    )

    if st.button(
        "Сформировать HTML-отчет",
        use_container_width=True,
    ):
        try:
            st.session_state.html_report_result = export_session_html_report(
                training_result=st.session_state.get("training_result"),
                evaluation_result=st.session_state.get("evaluation_result"),
                comparison_result=st.session_state.get("model_comparison_result"),
                registry_result=st.session_state.get("experiment_registry_result"),
                error_analysis_result=st.session_state.get("batch_error_analysis_result"),
            )
        except HtmlReportExportError as error:
            st.session_state.html_report_result = None
            st.error(str(error))

    html_report_result = st.session_state.html_report_result
    if html_report_result is None:
        st.info(
            "После запуска здесь появится путь к HTML-файлу со сводкой по текущим результатам."
        )
        return

    _render_html_report_preview(html_report_result)


def _render_docx_report_section() -> None:
    """Отрисовывает блок формирования DOCX-отчета по текущей сессии."""
    import streamlit as st

    if "docx_report_result" not in st.session_state:
        st.session_state.docx_report_result = None

    st.subheader("DOCX-отчет")
    st.caption(
        "На этом этапе можно собрать DOCX-отчет по текущей сессии: "
        "обучение модели, метрики, сравнение запусков, реестр экспериментов и анализ ошибок."
    )

    if st.button(
        "Сформировать DOCX-отчет",
        use_container_width=True,
    ):
        try:
            st.session_state.docx_report_result = export_session_docx_report(
                training_result=st.session_state.get("training_result"),
                evaluation_result=st.session_state.get("evaluation_result"),
                comparison_result=st.session_state.get("model_comparison_result"),
                registry_result=st.session_state.get("experiment_registry_result"),
                error_analysis_result=st.session_state.get("batch_error_analysis_result"),
            )
        except DocxReportExportError as error:
            st.session_state.docx_report_result = None
            st.error(str(error))

    docx_report_result = st.session_state.docx_report_result
    if docx_report_result is None:
        st.info(
            "После запуска здесь появится путь к DOCX-файлу со сводкой по текущим результатам."
        )
        return

    _render_docx_report_preview(docx_report_result)


def _render_markdown_report_section() -> None:
    """Отрисовывает блок формирования Markdown-отчета по текущей сессии."""
    import streamlit as st

    if "markdown_report_result" not in st.session_state:
        st.session_state.markdown_report_result = None

    st.subheader("Markdown-отчет")
    st.caption(
        "На этом этапе можно собрать Markdown-отчет с таблицами и короткими "
        "текстовыми фрагментами для вставки в пояснительную записку ВКР."
    )

    if st.button(
        "Сформировать Markdown-отчет",
        use_container_width=True,
    ):
        try:
            st.session_state.markdown_report_result = export_session_markdown_report(
                training_result=st.session_state.get("training_result"),
                evaluation_result=st.session_state.get("evaluation_result"),
                comparison_result=st.session_state.get("model_comparison_result"),
                registry_result=st.session_state.get("experiment_registry_result"),
                error_analysis_result=st.session_state.get("batch_error_analysis_result"),
            )
        except MarkdownReportExportError as error:
            st.session_state.markdown_report_result = None
            st.error(str(error))

    markdown_report_result = st.session_state.markdown_report_result
    if markdown_report_result is None:
        st.info(
            "После запуска здесь появится путь к Markdown-файлу со сводкой по текущим результатам."
        )
        return

    _render_markdown_report_preview(markdown_report_result)


def _render_thesis_tables_section() -> None:
    """Отрисовывает блок выгрузки отдельных CSV-таблиц для ВКР."""
    import streamlit as st

    if "thesis_tables_result" not in st.session_state:
        st.session_state.thesis_tables_result = None

    st.subheader("Таблицы для ВКР")
    st.caption(
        "На этом этапе можно отдельно выгрузить CSV-таблицы по метрикам, "
        "сравнению моделей и анализу ошибок для вставки в текст ВКР."
    )

    if st.button(
        "Сформировать CSV-таблицы для ВКР",
        use_container_width=True,
    ):
        try:
            st.session_state.thesis_tables_result = export_thesis_tables(
                evaluation_result=st.session_state.get("evaluation_result"),
                comparison_result=st.session_state.get("model_comparison_result"),
                error_analysis_result=st.session_state.get("batch_error_analysis_result"),
            )
        except ThesisTablesExportError as error:
            st.session_state.thesis_tables_result = None
            st.error(str(error))

    thesis_tables_result = st.session_state.thesis_tables_result
    if thesis_tables_result is None:
        st.info(
            "После запуска здесь появятся пути к отдельным CSV-таблицам по текущим результатам."
        )
        return

    _render_thesis_tables_preview(thesis_tables_result)


def _render_plot_export_section() -> None:
    """Отрисовывает блок выгрузки PNG-графиков для ВКР и презентации."""
    import streamlit as st

    if "plot_export_result" not in st.session_state:
        st.session_state.plot_export_result = None

    st.subheader("PNG-графики")
    st.caption(
        "На этом этапе можно выгрузить PNG-графики по метрикам качества модели "
        "и сравнению нескольких обученных моделей."
    )

    if st.button(
        "Сформировать PNG-графики",
        use_container_width=True,
    ):
        try:
            st.session_state.plot_export_result = export_plots(
                evaluation_result=st.session_state.get("evaluation_result"),
                comparison_result=st.session_state.get("model_comparison_result"),
            )
        except PlotExportError as error:
            st.session_state.plot_export_result = None
            st.error(str(error))

    plot_export_result = st.session_state.plot_export_result
    if plot_export_result is None:
        st.info(
            "После запуска здесь появятся пути к PNG-графикам по текущим результатам."
        )
        return

    _render_plot_export_preview(plot_export_result)


def _render_confusion_heatmap_section() -> None:
    """Отрисовывает блок выгрузки PNG-тепловых карт матриц ошибок."""
    import streamlit as st

    if "confusion_heatmap_result" not in st.session_state:
        st.session_state.confusion_heatmap_result = None

    st.subheader("PNG-матрицы ошибок")
    st.caption(
        "На этом этапе можно выгрузить PNG-тепловые карты матриц ошибок для "
        "validation/test и для пакетной оценки на размеченном CSV."
    )

    if st.button(
        "Сформировать PNG-матрицы ошибок",
        use_container_width=True,
    ):
        try:
            st.session_state.confusion_heatmap_result = export_confusion_heatmaps(
                detailed_evaluation_result=st.session_state.get("detailed_evaluation_result"),
                batch_evaluation_result=st.session_state.get("batch_inference_evaluation_result"),
            )
        except ConfusionHeatmapExportError as error:
            st.session_state.confusion_heatmap_result = None
            st.error(str(error))

    confusion_heatmap_result = st.session_state.confusion_heatmap_result
    if confusion_heatmap_result is None:
        st.info(
            "После запуска здесь появятся пути к PNG-тепловым картам матриц ошибок."
        )
        return

    _render_confusion_heatmap_preview(confusion_heatmap_result)


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
    _render_saved_artifacts_loading_section()
    _render_transformers_artifacts_loading_section()
    _render_single_inference_section()
    _render_batch_inference_section()
    _render_experiment_registry_section()
    _render_model_comparison_section()
    _render_html_report_section()
    _render_docx_report_section()
    _render_markdown_report_section()
    _render_thesis_tables_section()
    _render_plot_export_section()
    _render_confusion_heatmap_section()

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
                f"Каталог отчетов по загрузке артефактов: {PROJECT_PATHS.loading_reports_dir}",
                f"Каталог отчетов по инференсу: {PROJECT_PATHS.inference_reports_dir}",
                f"Каталог сводных реестров экспериментов: {PROJECT_PATHS.experiment_reports_dir}",
                f"Каталог сравнений моделей: {PROJECT_PATHS.comparison_reports_dir}",
                f"Каталог анализа уверенности предсказаний: {PROJECT_PATHS.confidence_reports_dir}",
                f"Каталог анализа ошибок: {PROJECT_PATHS.error_analysis_reports_dir}",
                f"Каталог HTML-отчетов: {PROJECT_PATHS.html_reports_dir}",
                f"Каталог DOCX-отчетов: {PROJECT_PATHS.docx_reports_dir}",
                f"Каталог Markdown-отчетов: {PROJECT_PATHS.markdown_reports_dir}",
                f"Каталог CSV-таблиц для ВКР: {PROJECT_PATHS.thesis_tables_reports_dir}",
                f"Каталог PNG-графиков: {PROJECT_PATHS.plots_reports_dir}",
                f"Каталог PNG-матриц ошибок: {PROJECT_PATHS.heatmaps_reports_dir}",
            ]
        ),
        language="text",
    )

    st.info(
        "Текущая версия приложения является стартовой. "
        "Функциональные вкладки будут добавляться постепенно, отдельными коммитами."
    )
