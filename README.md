# CTR Click Probability Pipeline

Простой и полный пайплайн для обучения модели вероятности клика (CTR) на `data/dataset.csv`.

## Что делает пайплайн

- Загружает данные из CSV.
- Чистит аномалии:
  - использует только строки с `Показы > 0`;
  - ограничивает `Переходы` сверху значением `Показы`.
- Строит таргет вероятности клика: `click_probability = Переходы / Показы`.
- Обучает `CatBoostRegressor` на признаках:
  - `ID кампании`
  - `ID баннера`
  - `Тип баннера`
  - `Тип устройства`
  - `Показы`
- Считает метрики на валидации: `MAE`, `RMSE`, `R2`.
- Сохраняет артефакты:
  - модель: `models/model.cbm`
  - metadata и метрики: `models/model_meta.json`

## Запуск обучения

```bash
venv/bin/python src/train.py
```

Настройки обучения и MLflow задаются в `src/config.py` (класс `TrainConfig`), например:

```bash
mlflow_tracking_uri = "sqlite:///mlflow.db"
mlflow_experiment = "ctr-click-probability"
mlflow_run_name = "catboost_ctr"
mlflow_registered_model_name = "ctr_click_probability_model"
```

## MLflow versioning моделей

Что логируется в MLflow:
- параметры обучения и фичи;
- метрики валидации (`mae`, `rmse`, `r2`, `baseline_mae`);
- артефакты: `models/model.cbm` и `models/model_meta.json`;
- модель в Model Registry (новая версия в `ctr_click_probability_model`).

Запуск с локальной БД (рекомендуется):

```bash
venv/bin/python src/train.py
```

Запуск MLflow UI:

```bash
venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

После этого откройте `http://127.0.0.1:5000`.

## Запуск приложения для прогноза

```bash
venv/bin/streamlit run src/app.py
```

Приложение:
- делает прогноз вероятности клика для одного показа;
- оценивает экономику показа (ожидаемая ценность, маржа, решение покупать/не покупать);
- выводит метрики последнего обучения из `model_meta.json`.
