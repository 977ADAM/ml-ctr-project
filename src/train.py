from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from config import TrainConfig, get_train_config

FEATURE_COLUMNS = [
    "ID кампании",
    "ID баннера",
    "Тип баннера",
    "Тип устройства",
    "Показы",
]
CAT_COLUMNS = ["ID кампании", "ID баннера", "Тип баннера", "Тип устройства"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + ["Переходы"]
N_SPLITS = 5
N_TRIALS = 40


def validate_input_frame(df: pd.DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")


def load_raw_data(data_path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    validate_input_frame(df)
    return df


def build_training_dataset(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = raw_df.copy()
    df = df[df["Показы"] > 0].copy()
    if df.empty:
        raise ValueError("Dataset has no rows with 'Показы' > 0.")

    shows = df["Показы"].astype(float).to_numpy()
    clicks = df["Переходы"].clip(lower=0).astype(float).to_numpy()
    clipped_clicks = np.minimum(clicks, shows)
    target = np.clip(clipped_clicks / shows, 0.0, 1.0)

    X = df[FEATURE_COLUMNS].copy()
    y = pd.Series(target, index=df.index, name="click_probability")
    groups = df["ID баннера"].copy()
    return X, y, groups


def suggest_catboost_params(trial: optuna.Trial, random_state: int) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
        "cat_features": CAT_COLUMNS,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": random_state,
        "verbose": False,
    }


def cv_mae_for_params(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    params: dict[str, Any],
) -> tuple[float, list[int]]:
    cv = GroupKFold(n_splits=N_SPLITS)
    fold_mae_scores: list[float] = []
    fold_best_iterations: list[int] = []

    for train_idx, valid_idx in cv.split(X, y, groups=groups):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=200,
            use_best_model=True,
            verbose=False,
        )

        preds = np.clip(model.predict(X_valid), 0.0, 1.0)
        fold_mae_scores.append(float(mean_absolute_error(y_valid, preds)))

        best_iter = model.get_best_iteration()
        fold_best_iterations.append(int(best_iter if best_iter >= 0 else model.tree_count_))

    return float(np.mean(fold_mae_scores)), fold_best_iterations


def tune_hyperparameters(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    random_state: int,
) -> tuple[dict[str, Any], float]:
    best_fold_iterations: list[int] = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_fold_iterations
        params = suggest_catboost_params(trial, random_state=random_state)
        mean_cv_mae, fold_iterations = cv_mae_for_params(
            X=X,
            y=y,
            groups=groups,
            params=params,
        )
        trial.set_user_attr("fold_best_iterations", fold_iterations)
        best_fold_iterations = fold_iterations
        return mean_cv_mae

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = dict(study.best_trial.params)
    best_params["iterations"] = int(best_params["iterations"])
    best_params["depth"] = int(best_params["depth"])

    fold_iterations = study.best_trial.user_attrs.get(
        "fold_best_iterations", best_fold_iterations
    )
    if fold_iterations:
        best_params["iterations"] = int(np.mean(fold_iterations))

    return best_params, float(study.best_value)


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, Any],
    random_state: int,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        **best_params,
        cat_features=CAT_COLUMNS,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=random_state,
        verbose=False,
    )
    model.fit(X, y, verbose=200)
    return model


def get_model_best_iteration(model: CatBoostRegressor) -> int:
    best_iter = model.get_best_iteration()
    return int(best_iter if best_iter >= 0 else model.tree_count_)


def save_artifacts(
    model: CatBoostRegressor,
    config: TrainConfig,
    metrics: dict[str, float],
    n_train_rows: int,
    n_valid_rows: int,
    best_params: dict[str, Any],
    mlflow_run_id: str,
    save_model_file: bool = True,
) -> None:
    if save_model_file:
        config.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(config.model_path))

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(config.data_path),
        "model_path": str(config.model_path),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CAT_COLUMNS,
        "target_name": "click_probability",
        "rows_train": n_train_rows,
        "rows_valid": n_valid_rows,
        "best_iteration": get_model_best_iteration(model),
        "best_params": best_params,
        "metrics_valid": metrics,
        "random_state": config.random_state,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_tracking_uri": config.mlflow_tracking_uri,
        "mlflow_experiment": config.mlflow_experiment,
        "mlflow_registered_model_name": config.mlflow_registered_model_name,
    }
    config.meta_path.parent.mkdir(parents=True, exist_ok=True)
    with config.meta_path.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=2)


def log_to_mlflow(
    *,
    config: TrainConfig,
    model: CatBoostRegressor,
    metrics: dict[str, float],
    best_params: dict[str, Any],
    n_train_rows: int,
    n_valid_rows: int,
) -> str:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    with mlflow.start_run(run_name=config.mlflow_run_name) as run:
        run_id = run.info.run_id

        mlflow.log_param("model_iterations", best_params["iterations"])
        mlflow.log_param("model_learning_rate", best_params["learning_rate"])
        mlflow.log_param("model_depth", best_params["depth"])
        mlflow.log_param("model_l2_leaf_reg", best_params["l2_leaf_reg"])

        mlflow.log_param("data_path", str(config.data_path))
        mlflow.log_param("cv_folds", N_SPLITS)
        mlflow.log_param("optuna_trials", N_TRIALS)
        mlflow.log_param("random_state", config.random_state)
        mlflow.log_param("n_train_rows", n_train_rows)
        mlflow.log_param("n_valid_rows", n_valid_rows)
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("categorical_columns", ",".join(CAT_COLUMNS))

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))
        mlflow.log_metric("best_iteration", float(get_model_best_iteration(model)))

        mlflow.log_artifact(str(config.model_path), artifact_path="artifacts")
        mlflow.log_artifact(str(config.meta_path), artifact_path="artifacts")
        model_uri = mlflow.catboost.log_model(
            cb_model=model,
            name="model",
            registered_model_name=config.mlflow_registered_model_name,
        ).model_uri
        mlflow.set_tag("model_uri", model_uri)
        return run_id


def main() -> None:
    config = get_train_config()

    raw_df = load_raw_data(config.data_path)
    X, y, groups = build_training_dataset(raw_df)

    best_params, cv_mae = tune_hyperparameters(
        X=X,
        y=y,
        groups=groups,
        random_state=config.random_state,
    )

    model = train_final_model(
        X=X,
        y=y,
        best_params=best_params,
        random_state=config.random_state,
    )

    metrics = {
        "cv_mae": float(cv_mae),
        "mae": float(cv_mae),
    }

    save_artifacts(
        model=model,
        config=config,
        metrics=metrics,
        n_train_rows=len(X),
        n_valid_rows=0,
        best_params=best_params,
        mlflow_run_id="pending",
    )

    run_id = log_to_mlflow(
        config=config,
        model=model,
        metrics=metrics,
        best_params=best_params,
        n_train_rows=len(X),
        n_valid_rows=0,
    )

    save_artifacts(
        model=model,
        config=config,
        metrics=metrics,
        n_train_rows=len(X),
        n_valid_rows=0,
        best_params=best_params,
        mlflow_run_id=run_id,
        save_model_file=False,
    )

    print("Training complete.")
    print(f"Model saved to: {config.model_path}")
    print(f"Metadata saved to: {config.meta_path}")
    print(
        f"MLflow: tracking_uri={config.mlflow_tracking_uri}, "
        f"experiment={config.mlflow_experiment}, run_id={run_id}"
    )
    print(f"Best params: {best_params}")
    print(f"CV MAE (5-fold GroupKFold): {cv_mae:.6f}")
    print(f"Best iteration: {get_model_best_iteration(model)}")


if __name__ == "__main__":
    main()
