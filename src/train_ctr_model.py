# train_ctr_model.py
# pip install pandas scikit-learn joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib


DATA_PATH = "data/dataset.csv"          # <-- поменяй путь при необходимости
MODEL_PATH = "models/ctr_model.joblib"    # куда сохранить модель


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Базовая проверка наличия нужных колонок
    required = {"CTR", "Показы", "Переходы", "Тип баннера", "Тип устройства", "ID кампании", "ID баннера"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В датасете нет колонок: {missing}")

    return df


def build_pipeline(cat_cols, num_cols) -> Pipeline:
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, cat_cols),
            ("num", numeric_transformer, num_cols),
        ],
        remainder="drop"
    )

    # Модель хорошо работает на табличных данных без сильной настройки
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_depth=6,
        max_iter=500,
        random_state=42
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    return pipe


def evaluate(y_true, y_pred) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("\n=== Метрики ===")
    print(f"MAE : {mae:.6f}")
    print(f"R^2 : {r2:.6f}")


def main():
    df = load_data(DATA_PATH)

    target = "CTR"

    cat_cols = ["Тип баннера", "Тип устройства"]
    num_cols = ["ID кампании", "ID баннера", "Показы"]

    X = df[cat_cols + num_cols].copy()
    y = df[target].copy()

    # Сплит
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline(cat_cols=cat_cols, num_cols=num_cols)

    print("Обучение модели...")
    pipe.fit(X_train, y_train)

    print("Оценка на тесте...")
    pred = pipe.predict(X_test)
    pred = np.clip(pred, 0, 1)
    evaluate(y_test, pred)

    # Сохраним модель
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nМодель сохранена: {MODEL_PATH}")

    # Пример инференса на новых данных
    example = pd.DataFrame([{
        "ID кампании": 3405596,
        "ID баннера": 15262577,
        "Тип баннера": "interactive",
        "Тип устройства": "Компьютер",
        "Показы": 12596,
    }])

    example_pred = float(np.clip(pipe.predict(example)[0], 0, 1))
    print(f"\nПример: прогноз CTR = {example_pred:.6f}")


if __name__ == "__main__":
    main()