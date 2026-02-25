import mlflow
import mlflow.catboost
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool


try:
    from .schema import FEATURE_SCHEMA
    from .config import get_train_config
except ImportError:
    from schema import FEATURE_SCHEMA
    from config import get_train_config


config = get_train_config()

mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment("CTR Prediction with CatBoost")

DATA_PATH = "data/dataset.csv"

df = pd.read_csv(DATA_PATH)

# TARGET это доля (clicks/impressions)
# Важно: weight-колонка не должна попадать в признаки (иначе leakage)
TARGET = FEATURE_SCHEMA.target
WEIGHT_COL = FEATURE_SCHEMA.weight

missing = [c for c in [TARGET, WEIGHT_COL] if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns in dataset: {missing}")

X = df.drop(columns=[TARGET, WEIGHT_COL])
y = df[TARGET].astype(float)
w = df[WEIGHT_COL].astype(float)

cat_features = FEATURE_SCHEMA.categorical
num_features = FEATURE_SCHEMA.numerical

# Важно: в датасете могут отсутствовать некоторые фичи из схемы.
# CatBoost Pool упадёт, если передать несуществующие cat_features.
cat_features = [c for c in cat_features if c in X.columns]
num_features = [c for c in num_features if c in X.columns]



# Веса должны быть неотрицательными
if (w < 0).any():
    raise ValueError("Weights must be non-negative.")

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.15, random_state=42
)
X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(
    X_train, y_train, w_train, test_size=0.1765, random_state=42  # ~0.15 от исходного
)

train_pool = Pool(X_train, y_train, cat_features=cat_features, weight=w_train)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features, weight=w_valid)


# Запуск контекста MLflow и сохранение гиперпараметров
def experiment(run_name, train_pool, valid_pool, X_test, y_test, w_test, loss_function=None, eval_metric=None):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("data_path", DATA_PATH)
        mlflow.log_param("model_class", "CatBoostRegressor")

        if loss_function is not None:
            mlflow.log_param("loss_function", loss_function)
        if eval_metric is not None:
            mlflow.log_param("eval_metric", eval_metric)

        mlflow.log_param("depth", 8)
        mlflow.log_param("learning_rate", 0.03)
        mlflow.log_param("iterations", 2000)
        mlflow.log_param("early_stopping_rounds", 200)

        # В MLflow параметры лучше логировать строкой (детерминированно и без сюрпризов сериализации)
        mlflow.log_param("cat_features", json.dumps(cat_features, ensure_ascii=False))
        mlflow.log_param("num_features", json.dumps(num_features, ensure_ascii=False))
        mlflow.log_param("target", TARGET)
        mlflow.log_param("weight_col", WEIGHT_COL)

        model_kwargs = dict(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            random_seed=42,
            early_stopping_rounds=200,
            use_best_model=True,
            verbose=200
        )
        if loss_function is not None:
            model_kwargs["loss_function"] = loss_function
        if eval_metric is not None:
            model_kwargs["eval_metric"] = eval_metric

        model = CatBoostRegressor(**model_kwargs)

        model.fit(train_pool, eval_set=valid_pool)

        test_pool = Pool(X_test, cat_features=cat_features)
        pred = model.predict(test_pool)

        wrmse = np.sqrt(np.average((y_test - pred) ** 2, weights=w_test))
        rmse  = np.sqrt(mean_squared_error(y_test, pred))

        print(f"RMSE (unweighted): {rmse:.6f}")
        print(f"RMSE (weighted):   {wrmse:.6f}")
        print(f"Best iteration:    {model.get_best_iteration()}")


        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("wrmse", wrmse)
        mlflow.log_metric("best_iteration", model.get_best_iteration())


        # Сохранение модели в MLflow
        mlflow.catboost.log_model(model, artifact_path="model")
        print("Model and metadata saved.")



if __name__ == "__main__":
    experiment(
        run_name="catboost_ctr_model_MAE",
        train_pool=train_pool, valid_pool=valid_pool, X_test=X_test, y_test=y_test, w_test=w_test,
        loss_function="MAE",
        eval_metric="MAE"
    )

    experiment(
        run_name="catboost_ctr_model_RMSE",
        train_pool=train_pool, valid_pool=valid_pool, X_test=X_test, y_test=y_test, w_test=w_test,
        loss_function="RMSE",
        eval_metric="RMSE"
    )