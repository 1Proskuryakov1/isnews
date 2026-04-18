# Каталог ноутбуков

В этом каталоге размещаются Jupyter Notebook-файлы для запуска в Google Colab.

Текущий ноутбук:

- `colab_baseline_training.ipynb` — базовый сценарий загрузки датасета, очистки текстов, обучения `TF-IDF + LogisticRegression`, расчета метрик и сохранения артефактов.
- `colab_batch_inference_evaluation.ipynb` — пакетный инференс по CSV через сохраненные `joblib`-артефакты, сохранение предсказаний и оценка качества на размеченном файле.
- `colab_transformers_training.ipynb` — обучение нейросетевой модели на базе `transformers` в Google Colab, расчет метрик и сохранение артефактов `save_pretrained`, токенизатора и файла весов `model1.pt`.
