from __future__ import annotations
import json
from typing import Any
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

try:
    from .cli_args import CTRTrainCLI
except ImportError:
    from cli_args import CTRTrainCLI


def prepare_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    
    if target_col in df.columns:

        y = df[target_col].astype(float)

        X = df.drop(columns=[target_col])

        return X, y

    if {"Переходы", "Показы"}.issubset(df.columns):

        safe_shows = df["Показы"].replace(0, np.nan)

        y = (df["Переходы"] / safe_shows).fillna(0.0).astype(float)

        X = df.copy()

        return X, y

    raise ValueError(
        f"Цель '{target_col}' не найдено и невозможно сделать вывод CTR от 'Переходы'/'Показы'."
    )


def weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None) -> dict[str, float]:
    kwargs: dict[str, Any] = {}
    if w is not None:
        kwargs["sample_weight"] = w

    rmse = mean_squared_error(y_true, y_pred, **kwargs) ** 0.5
    mae = mean_absolute_error(y_true, y_pred, **kwargs)
    r2 = r2_score(y_true, y_pred, **kwargs)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def main() -> None:
    args = CTRTrainCLI().parse()

    df = pd.read_csv(args.data)

    # Удалите столбцы с утечками в целевом объекте.
    X, y = prepare_target(df, args.target)

    leakage_cols = [c for c in ["CTR", "Переходы"] if c in X.columns]
    
    if leakage_cols:
        X = X.drop(columns=leakage_cols)

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    sample_weight = None
    if "Показы" in df.columns:
        sample_weight = df["Показы"].clip(lower=1).to_numpy(dtype=float)

    split_kwargs = {"test_size": 0.2, "random_state": args.random_state}
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        sample_weight if sample_weight is not None else np.ones(len(df)),
        **split_kwargs,
    )

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=args.random_state,
        verbose=False,
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_cols, weight=w_train)
    test_pool = Pool(X_test, y_test, cat_features=cat_cols, weight=w_test)

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    preds = np.clip(model.predict(X_test), 0.0, 1.0)
    metrics = weighted_metrics(y_test.to_numpy(), preds, w_test)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / args.model_name
    meta_path = args.model_dir / "metrics.txt"

    model.save_model(model_path)

    with meta_path.open("w", encoding="utf-8") as f:
        f.write(f"rows={len(df)}\n")
        f.write(f"features={','.join(X.columns)}\n")
        f.write(f"categorical={','.join(cat_cols)}\n")
        f.write(json.dumps(metrics, ensure_ascii=False, indent=2))
        f.write("\n")

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {meta_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
