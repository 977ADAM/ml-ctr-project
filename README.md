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

Опционально:

```bash
venv/bin/python src/train.py \
  --data-path data/dataset.csv \
  --model-path models/model.cbm \
  --meta-path models/model_meta.json \
  --test-size 0.2 \
  --random-state 42
```

## Запуск приложения для прогноза

```bash
venv/bin/streamlit run src/app.py
```

Приложение:
- делает прогноз вероятности клика для одного показа;
- оценивает экономику показа (ожидаемая ценность, маржа, решение покупать/не покупать);
- выводит метрики последнего обучения из `model_meta.json`.
