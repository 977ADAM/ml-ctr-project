import mlflow
import mlflow.catboost
import pandas as pd
import json
import numpy as np
import logging
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor, Pool
from typing import List


try:
    from .schema import FEATURE_SCHEMA
    from .config import get_train_config
except ImportError:
    from schema import FEATURE_SCHEMA
    from config import get_train_config


config = get_train_config()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment(config.mlflow_experiment)

DATA_PATH = config.data_path

def load_and_prepare_data(data_path: str):
    df = pd.read_csv(data_path)

    # TARGET это доля (clicks/impressions)
    # Важно: weight-колонка не должна попадать в признаки (иначе leakage)
    TARGET = FEATURE_SCHEMA.target
    WEIGHT_COL = FEATURE_SCHEMA.weight
    CLICKS_COL = FEATURE_SCHEMA.clicks

    missing = [c for c in [TARGET, WEIGHT_COL, CLICKS_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")

    X = df.drop(columns=[TARGET, WEIGHT_COL, CLICKS_COL]).copy()
    y = df[TARGET].astype(float)
    w = df[WEIGHT_COL].astype(float)

    # Базовая валидация
    if y.isna().any():
        raise ValueError(f"Целевой столбец '{TARGET}' contains NaNs.")
    # if ((y < 0) | (y > 1)).any():
    #     raise ValueError(f"Цель '{TARGET}' ожидается в [0, 1] диапазон значений указан,\n но обнаружены значения, выходящие за его пределы.")
    if w.isna().any():
        raise ValueError(f"Weight column '{WEIGHT_COL}' contains NaNs.")
    if (w < 0).any():
        raise ValueError("Weights must be non-negative.")
    if (w == 0).all():
        raise ValueError("All weights are zero. Weighted metrics are undefined.")

    cat_features = [c for c in FEATURE_SCHEMA.categorical if c in X.columns]
    num_features = FEATURE_SCHEMA.numerical

    extra_non_numeric = [
        c for c in X.columns
        if c not in set(cat_features) and c not in set(num_features) and (not is_numeric_dtype(X[c]))
    ]
    if extra_non_numeric:
        # Добавляем в категориальные детерминированно (по порядку колонок)
        cat_features = cat_features + [c for c in extra_non_numeric if c not in cat_features]

    for c in cat_features:
        X[c] = X[c].astype("string").fillna("__MISSING__")

    leftover_non_numeric = [
        c for c in X.columns
        if c not in set(cat_features) and c not in set(num_features) and (not is_numeric_dtype(X[c]))
    ]
    for c in leftover_non_numeric:
        X[c] = X[c].astype("string").fillna("__MISSING__")
        if c not in cat_features:
            cat_features.append(c)

    return X, y, w, TARGET, WEIGHT_COL, cat_features, num_features



test_size = 0.15
valid_size = 0.15
RANDOM_SEED = 42

def _cat_feature_indices(X: pd.DataFrame, cat_feature_names: List[str]) -> List[int]:
    """
    CatBoost стабильнее работает, когда cat_features переданы индексами колонок.
    Это также защищает от различий между версиями CatBoost/типами входных данных.
    """
    cols = list(X.columns)
    name_set = set(cat_feature_names)
    return [i for i, c in enumerate(cols) if c in name_set]


# Запуск контекста MLflow и сохранение гиперпараметров
def experiment(
    run_name,
    train_pool,
    valid_pool,
    X_test,
    y_test,
    w_test,
    cat_features,
    cat_features_idx,
    num_features,
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
        mlflow.log_param("cat_features_idx", json.dumps(cat_features_idx, ensure_ascii=False))
        mlflow.log_param("target", target)
        mlflow.log_param("weight_col", weight_col)
        mlflow.log_param("num_features", json.dumps(num_features, ensure_ascii=False))

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

        test_pool = Pool(X_test, y_test, cat_features=cat_features_idx, weight=w_test)
        pred = model.predict(test_pool)

        y_test_np = np.asarray(y_test, dtype=float)
        w_test_np = np.asarray(w_test, dtype=float)
        pred_np = np.asarray(pred, dtype=float)

        if np.sum(w_test_np) > 0:
            wrmse = np.sqrt(np.average((y_test_np - pred_np) ** 2, weights=w_test_np))
            wmae = np.average(np.abs(y_test_np - pred_np), weights=w_test_np)
        else:
            wrmse = float("nan")
            wmae = float("nan")

        rmse  = np.sqrt(mean_squared_error(y_test_np, pred_np))
        mae = mean_absolute_error(y_test_np, pred_np)

        logger.info("RMSE (unweighted): %.6f", rmse)
        logger.info("RMSE (weighted):   %.6f", wrmse)
        logger.info("MAE  (unweighted): %.6f", mae)
        logger.info("MAE  (weighted):   %.6f", wmae)
        logger.info("Best iteration:    %s", model.get_best_iteration())


        mlflow.log_metric("rmse", rmse)
        if np.isfinite(wrmse):
            mlflow.log_metric("wrmse", wrmse)
        mlflow.log_metric("mae", mae)
        if np.isfinite(wmae):
            mlflow.log_metric("wmae", wmae)
        mlflow.log_metric("best_iteration", model.get_best_iteration())

        # Для удобства трекинга: логируем "главную" метрику, соответствующую eval_metric
        if (eval_metric or "").upper() == "MAE":
            mlflow.log_metric("primary_metric", mae)
            if np.isfinite(wmae):
                mlflow.log_metric("primary_metric_weighted", wmae)
        elif (eval_metric or "").upper() == "RMSE":
            mlflow.log_metric("primary_metric", rmse)
            if np.isfinite(wrmse):
                mlflow.log_metric("primary_metric_weighted", wrmse)


        # Сохранение модели в MLflow
        mlflow.catboost.log_model(model, artifact_path="model")
        logger.info("Model and metadata saved.")



if __name__ == "__main__":
    X, y, w, TARGET, WEIGHT_COL, cat_features, num_features = load_and_prepare_data(DATA_PATH)
    cat_features_idx = _cat_feature_indices(X, cat_features)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=test_size, random_state=RANDOM_SEED
    )
    valid_rel = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(
        X_train, y_train, w_train, test_size=valid_rel, random_state=RANDOM_SEED
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features_idx, weight=w_train)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features_idx, weight=w_valid)

    experiment(
        run_name="catboost_ctr_model_MAE",
        train_pool=train_pool, valid_pool=valid_pool, X_test=X_test, y_test=y_test, w_test=w_test,
        cat_features=cat_features, cat_features_idx=cat_features_idx, target=TARGET, weight_col=WEIGHT_COL,
        num_features=num_features,
        loss_function="MAE",
        eval_metric="MAE"
    )