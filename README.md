# CTR model pipeline (simple + practical)

Минимальный рабочий пайплайн для обучения модели, которая предсказывает вероятность клика (`CTR`) по `data/dataset.csv`.

## Что делает
- читает `data/dataset.csv`
- берёт `CTR` как целевую переменную
- исключает утечку таргета (`CTR`, `Переходы`) из признаков
- обучает `CatBoostRegressor` на категориальных и числовых признаках
- учитывает `Показы` как веса наблюдений
- сохраняет модель и метрики
- выполняет инференс в отдельном скрипте

## Запуск обучения
```bash
venv/bin/python src/train.py
```

Артефакты после обучения:
- `models/ctr_model.cbm` — обученная модель
- `models/metrics.txt` — метрики и список признаков

## Запуск предсказаний
```bash
venv/bin/python src/predict.py --input data/dataset.csv --output models/predictions.csv
```

Выход:
- `models/predictions.csv` с колонками:
  - `predicted_ctr`
  - `predicted_clicks` (если в данных есть `Показы`)

## Параметры (опционально)
```bash
venv/bin/python src/train.py --data data/dataset.csv --model-dir models --model-name ctr_model.cbm --target CTR
```

## Streamlit интерфейс
Запуск:
```bash
venv/bin/streamlit run src/app.py
```

В интерфейсе есть:
- ручной ввод признаков и прогноз `CTR`
- переменная `Стоимость 1 показа`
- переменная `Ценность 1 клика`
- бизнес-решение: покупать показ или нет

Правило решения:
```text
покупать, если predicted_ctr * ценность_клика >= стоимость_показа
```
