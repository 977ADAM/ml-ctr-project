# =========================
# 1. Импорт библиотек
# =========================
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .config import config
    from .schema import FEATURE_SCHEMA
except ImportError:
    from config import config
    from schema import FEATURE_SCHEMA

def main():
    # =========================
    # 2. Загрузка данных
    # =========================
    df = pd.read_csv(config.dataset)

    # =========================
    # 3. Подготовка данных
    # =========================

    # Убираем ID (они не несут полезной информации)
    df = df.drop(columns=FEATURE_SCHEMA.drop_columns)

    # Целевая переменная
    y = df[config.target]

    # Признаки
    X = df.drop(columns=[config.target])

    # Категориальные признаки
    cat_features = FEATURE_SCHEMA.categorical

    # =========================
    # 4. Разделение на train/test
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    # =========================
    # 5. Обучение модели
    # =========================
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

    # =========================
    # 6. Оценка качества
    # =========================
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2: {r2:.4f}")


    model.save_model("model.pkl")


if __name__ == "__main__":
    main()