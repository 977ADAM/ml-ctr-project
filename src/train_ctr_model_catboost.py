import mlflow
import mlflow.catboost
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor, Pool


try:
    from .schema import FEATURE_SCHEMA
    from .config import get_train_config
except ImportError:
    from schema import FEATURE_SCHEMA
    from config import get_train_config


config = get_train_config()

mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment(config.mlflow_experiment)

DATA_PATH = config.data_path

def load_and_prepare_data(data_path: str):
    df = pd.read_csv(data_path)

    # TARGET это доля (clicks/impressions)
    # Важно: weight-колонка не должна попадать в признаки (иначе leakage)
    TARGET = FEATURE_SCHEMA.target
    WEIGHT_COL = FEATURE_SCHEMA.weight

    missing = [c for c in [TARGET, WEIGHT_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")

    X = df.drop(columns=[TARGET, WEIGHT_COL]).copy()
    y = df[TARGET].astype(float)
    w = df[WEIGHT_COL].astype(float)

    # Базовая валидация
    if y.isna().any():
        raise ValueError(f"Целевой столбец '{TARGET}' contains NaNs.")
    if w.isna().any():
        raise ValueError(f"Weight column '{WEIGHT_COL}' contains NaNs.")
    if (w < 0).any():
        raise ValueError("Weights must be non-negative.")
    if (w == 0).all():
        raise ValueError("All weights are zero. Weighted metrics are undefined.")

    cat_features = FEATURE_SCHEMA.categorical
    num_features = FEATURE_SCHEMA.numerical

    # Важно: в датасете могут отсутствовать некоторые фичи из схемы.
    # CatBoost Pool упадёт, если передать несуществующие cat_features.
    cat_features = [c for c in cat_features if c in X.columns]
    num_features = [c for c in num_features if c in X.columns]

    for c in cat_features:
        X[c] = X[c].astype("string").fillna("__MISSING__")

    for c in num_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if num_features:
        X[num_features] = X[num_features].fillna(0.0)

    return X, y, w, TARGET, WEIGHT_COL, cat_features, num_features



test_size = 0.15
valid_size = 0.15
RANDOM_SEED = 42

valid_rel = valid_size / (1.0 - test_size)


# Запуск контекста MLflow и сохранение гиперпараметров
def experiment(
    run_name,
    train_pool,
    valid_pool,
    X_test,
    y_test,
    w_test,
    cat_features,
    target,
    weight_col,
    loss_function=None,
    eval_metric=None
):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "data_path": DATA_PATH,
            "model_class": "CatBoostRegressor",
            "n_rows": int(len(y_test) + train_pool.num_row() + valid_pool.num_row()),
            "n_features": int(train_pool.num_col()),
        })

        if loss_function is not None:
            mlflow.log_param("loss_function", loss_function)
        if eval_metric is not None:
            mlflow.log_param("eval_metric", eval_metric)

        mlflow.log_params({
            "depth": 8,
            "learning_rate": 0.03,
            "iterations": 2000,
            "early_stopping_rounds": 200,
        })

        # В MLflow параметры лучше логировать строкой (детерминированно и без сюрпризов сериализации)
        mlflow.log_param("cat_features", json.dumps(cat_features, ensure_ascii=False))
        mlflow.log_param("target", target)
        mlflow.log_param("weight_col", weight_col)

        model_kwargs = dict(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            random_seed=RANDOM_SEED,
            early_stopping_rounds=200,
            use_best_model=True,
            verbose=200
        )
        if loss_function is not None:
            model_kwargs["loss_function"] = loss_function
        if eval_metric is not None:
            model_kwargs["eval_metric"] = eval_metric

        model = CatBoostRegressor(**model_kwargs)
        mlflow.log_param("model_kwargs", json.dumps(model_kwargs, ensure_ascii=False, sort_keys=True))

        model.fit(train_pool, eval_set=valid_pool)

        test_pool = Pool(X_test, y_test, cat_features=cat_features, weight=w_test)
        pred = model.predict(test_pool)

        if np.sum(w_test) > 0:
            wrmse = np.sqrt(np.average((y_test - pred) ** 2, weights=w_test))
        else:
            wrmse = float("nan")

        rmse  = np.sqrt(mean_squared_error(y_test, pred))

        print(f"RMSE (unweighted): {rmse:.6f}")
        print(f"RMSE (weighted):   {wrmse:.6f}")
        print(f"Best iteration:    {model.get_best_iteration()}")


        mlflow.log_metric("rmse", rmse)
        if np.isfinite(wrmse):
            mlflow.log_metric("wrmse", wrmse)
        mlflow.log_metric("best_iteration", model.get_best_iteration())


        # Сохранение модели в MLflow
        mlflow.catboost.log_model(model, artifact_path="model")
        print("Model and metadata saved.")



if __name__ == "__main__":
    X, y, w, TARGET, WEIGHT_COL, cat_features, num_features = load_and_prepare_data(DATA_PATH)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=test_size, random_state=RANDOM_SEED
    )
    valid_rel = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(
        X_train, y_train, w_train, test_size=valid_rel, random_state=RANDOM_SEED
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features, weight=w_train)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features, weight=w_valid)

    experiment(
        run_name="catboost_ctr_model_MAE",
        train_pool=train_pool, valid_pool=valid_pool, X_test=X_test, y_test=y_test, w_test=w_test,
        cat_features=cat_features, target=TARGET, weight_col=WEIGHT_COL,
        loss_function="MAE",
        eval_metric="MAE"
    )

    experiment(
        run_name="catboost_ctr_model_RMSE",
        train_pool=train_pool, valid_pool=valid_pool, X_test=X_test, y_test=y_test, w_test=w_test,
        cat_features=cat_features, target=TARGET, weight_col=WEIGHT_COL,
        loss_function="RMSE",
        eval_metric="RMSE"
    )