# IsNews

Интеллектуальный сервис для классификации новостных публикаций по тематическим категориям.

Проект реализован на Python и содержит полный цикл работы с моделью: загрузка данных, предобработка, разбиение датасета, TF-IDF-векторизация, обучение нескольких моделей, сохранение и загрузка артефактов, одиночное и пакетное применение модели, оценка качества и экспорт отчетов.

## Быстрый запуск

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

После запуска откроется Streamlit-приложение. В интерфейсе можно загрузить CSV-датасет с колонками `text` и `label`, обучить модели, сохранить их, загрузить готовые артефакты и выполнить классификацию новостей.

## Воспроизведение обучения

Финальные демонстрационные модели и датасеты можно пересоздать командой:

```powershell
python scripts\train_demo_artifacts.py
```

Скрипт формирует учебный датасет новостей, запускает штатный ML-пайплайн проекта и сохраняет проверяемые артефакты:

- `data/raw/news_demo_dataset.csv` - исходный датасет для обучения.
- `data/processed/news_demo_dataset_processed.csv` - очищенный датасет.
- `data/splits/news_demo_dataset/train.csv` - обучающая выборка.
- `data/splits/news_demo_dataset/validation.csv` - валидационная выборка.
- `data/splits/news_demo_dataset/test.csv` - тестовая выборка.
- `data/features/news_demo_dataset_tfidf/*.npz` - TF-IDF признаки.
- `models/vectorizers/model_vectorizer.joblib` - сохраненный TF-IDF-векторизатор.
- `models/classifiers/model.joblib` - Logistic Regression.
- `models/classifiers/model1.joblib` - Multinomial Naive Bayes.
- `models/model_manifest.json` - manifest финальных артефактов.
- `reports/training/final_demo_training_summary.json` - итоговые метрики обучения.
- `reports/comparisons/final_model_comparison.csv` - сравнение моделей.

## Финальная проверка проекта

Для проверки сдаваемого комплекта выполните:

```powershell
python scripts\verify_project.py
```

Скрипт проверяет наличие обязательных файлов, корректность ноутбуков, закрепление версий в `requirements.txt`, загрузку моделей из файлов и Accuracy на сохраненной тестовой выборке.

## Текущие метрики демонстрационных моделей

Датасет содержит 180 новостных публикаций в 5 классах: политика, экономика, спорт, технологии и культура.

| Модель | Файл | Validation Accuracy | Test Accuracy |
| --- | --- | ---: | ---: |
| Logistic Regression | `models/classifiers/model.joblib` | 1.0 | 1.0 |
| MultinomialNB | `models/classifiers/model1.joblib` | 1.0 | 1.0 |

Метрики сохранены в `reports/training/final_demo_training_summary.json` и `reports/comparisons/final_model_comparison.csv`.

## Colab-ноутбуки

В каталоге `notebooks/` находятся файлы `.ipynb` для обучения и экспериментов:

- `colab_baseline_training.ipynb` - базовое обучение классической модели.
- `colab_transformers_training.ipynb` - нейросетевой эксперимент на Transformers.
- `colab_batch_inference_evaluation.ipynb` - пакетный инференс и оценка качества.

## Документация для ВКР

Дополнительные материалы для разделов ВКР находятся в `docs/`:

- `LIBRARY_JUSTIFICATION.md` - обоснование выбора библиотек.
- `MODEL_MATH.md` - математическое описание моделей.
- `TEST_PLAN.md` - план тестирования.
- `TEST_CASES.md` - тест-кейсы.
- `BUG_REPORTS.md` - журнал найденных и закрытых дефектов.
- `FINAL_READINESS_CHECKLIST.md` - чек-лист соответствия требованиям.

## Структура проекта

```text
isnews/
├── data/                 # исходные, обработанные и разделенные данные
├── docs/                 # материалы для ВКР и тестирования
├── models/               # сохраненные модели и векторизаторы
├── notebooks/            # Google Colab / Jupyter Notebook
├── reports/              # отчеты, метрики, сравнения и deployment-материалы
├── scripts/              # служебные скрипты обучения и проверки
├── src/isnews/           # основной код интеллектуального сервиса
├── main.py               # точка входа Streamlit-приложения
└── requirements.txt      # закрепленные зависимости проекта
```
