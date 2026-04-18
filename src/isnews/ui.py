"""Минимальный пользовательский интерфейс демонстрационного приложения."""

from __future__ import annotations

from src.isnews.config import PROJECT_PATHS
from src.isnews.data_loading import DatasetLoadResult, DatasetValidationError
from src.isnews.data_loading import load_dataset_from_uploaded_bytes, load_dataset_from_url


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

    st.write("Названия колонок:")
    st.code(", ".join(str(column) for column in dataset_result.dataframe.columns))

    st.write("Первые 10 строк датасета:")
    st.dataframe(dataset_result.dataframe.head(10), use_container_width=True)


def _render_dataset_loading_section() -> None:
    """Отрисовывает блок загрузки датасета из локального файла или по URL."""
    import streamlit as st

    if "dataset_result" not in st.session_state:
        st.session_state.dataset_result = None

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
            try:
                st.session_state.dataset_result = load_dataset_from_uploaded_bytes(
                    file_bytes=uploaded_file.getvalue(),
                    source_name=uploaded_file.name,
                )
            except DatasetValidationError as error:
                st.session_state.dataset_result = None
                st.error(str(error))

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
                except DatasetValidationError as error:
                    st.session_state.dataset_result = None
                    st.error(str(error))

    dataset_result = st.session_state.dataset_result
    if dataset_result is None:
        st.info(
            "После загрузки файла здесь появятся размер датасета, названия колонок "
            "и первые строки таблицы."
        )
        return

    _render_dataset_preview(dataset_result)


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
                f"Каталог моделей: {PROJECT_PATHS.models_dir}",
                f"Каталог ноутбуков: {PROJECT_PATHS.notebooks_dir}",
                f"Каталог отчетов: {PROJECT_PATHS.reports_dir}",
            ]
        ),
        language="text",
    )

    st.info(
        "Текущая версия приложения является стартовой. "
        "Функциональные вкладки будут добавляться постепенно, отдельными коммитами."
    )
