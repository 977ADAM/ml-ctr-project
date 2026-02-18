# =========================
# 1. Импорт библиотек
# =========================
import pandas as pd
import numpy as np
import mlflow
import mlflow.catboost
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .config import config
    from .schema import FEATURE_SCHEMA
except ImportError:
    from config import config
    from schema import FEATURE_SCHEMA

def main():
    mlflow.set_experiment("CTR_Pipeline_Experiment")
    # =========================
    # 2. Загрузка данных
    # =========================
    df = pd.read_csv(config.dataset)

    # =========================
    # 3. Подготовка данных
    # =========================
    # Проверяем наличие целевой переменной
    if config.target not in df.columns:
        raise KeyError(f"Target column '{config.target}' not found in dataset. Available columns: {list(df.columns)}")

    # Убираем ID (они не несут полезной информации)
    df = df.drop(columns=FEATURE_SCHEMA.drop_columns, errors='ignore')

    # Целевая переменная
    y = df[config.target]

    # Признаки
    X = df.drop(columns=[config.target])

    # Категориальные признаки
    # Оставляем только реально существующие в X колонки
    cat_features = [c for c in FEATURE_SCHEMA.categorical if c in X.columns]

    # =========================
    # 4. Разделение на train/test
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    with mlflow.start_run():
        mlflow.set_tag("pipeline_version", config.version)
        mlflow.set_tag("target", config.target)

        # Логируем параметры
        mlflow.log_params({
            "iterations": config.iterations,
            "depth": config.depth,
            "learning_rate": config.learning_rate,
            "loss_function": config.loss_function,
            "eval_metric": config.eval_metric,
            "test_size": config.test_size,
            "random_seed": config.random_seed,
            "random_state_split": config.random_state,
            "early_stopping_rounds": config.early_stopping_rounds,
        })

        model = CatBoostRegressor(
            iterations=config.iterations,
            depth=config.depth,
            learning_rate=config.learning_rate,
            loss_function=config.loss_function,
            eval_metric=config.eval_metric,
            random_seed=config.random_seed,
            verbose=config.verbose
        )

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            eval_set=(X_test, y_test),
            early_stopping_rounds=config.early_stopping_rounds
        )

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Логируем метрики
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})

        # Логируем модель
        mlflow.catboost.log_model(model, "catboost_model")

        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R2: {r2:.4f}")

if __name__ == "__main__":
    main()