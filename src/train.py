from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "ID кампании",
    "ID баннера",
    "Тип баннера",
    "Тип устройства",
    "Показы",
]
CAT_COLUMNS = ["ID кампании", "ID баннера", "Тип баннера", "Тип устройства"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + ["Переходы"]


@dataclass
class TrainConfig:
    data_path: Path
    model_path: Path
    meta_path: Path
    test_size: float
    random_state: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Training pipeline for CTR click probability model."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/model.cbm"))
    parser.add_argument("--meta-path", type=Path, default=Path("models/model_meta.json"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    return TrainConfig(
        data_path=args.data_path,
        model_path=args.model_path,
        meta_path=args.meta_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )


def validate_input_frame(df: pd.DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")


def load_raw_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    validate_input_frame(df)
    return df


def build_training_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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
    return X, y


def make_stratified_target(y: pd.Series) -> pd.Series | None:
    if y.nunique() < 2:
        return None
    try:
        bins = pd.qcut(y, q=min(10, y.nunique()), duplicates="drop")
    except ValueError:
        return None
    if bins.nunique() < 2:
        return None
    return bins


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify_target = make_stratified_target(y)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        cat_features=CAT_COLUMNS,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=random_state,
        verbose=False,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=200,
        use_best_model=True,
        verbose=200,
    )
    return model


def evaluate_model(
    model: CatBoostRegressor,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    train_target_mean: float,
) -> dict[str, float]:
    preds = np.clip(model.predict(X_valid), 0.0, 1.0)
    mae = mean_absolute_error(y_valid, preds)
    rmse = float(np.sqrt(mean_squared_error(y_valid, preds)))
    r2 = r2_score(y_valid, preds)
    baseline_preds = np.full(len(y_valid), train_target_mean)
    baseline_mae = mean_absolute_error(y_valid, baseline_preds)
    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "baseline_mae": float(baseline_mae),
    }


def save_artifacts(
    model: CatBoostRegressor,
    config: TrainConfig,
    metrics: dict[str, float],
    n_train_rows: int,
    n_valid_rows: int,
) -> None:
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
        "best_iteration": int(model.get_best_iteration()),
        "metrics_valid": metrics,
        "random_state": config.random_state,
    }
    config.meta_path.parent.mkdir(parents=True, exist_ok=True)
    with config.meta_path.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False, indent=2)


def main() -> None:
    config = parse_args()

    raw_df = load_raw_data(config.data_path)
    X, y = build_training_dataset(raw_df)
    X_train, X_valid, y_train, y_valid = split_data(
        X=X,
        y=y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        random_state=config.random_state,
    )
    metrics = evaluate_model(
        model=model,
        X_valid=X_valid,
        y_valid=y_valid,
        train_target_mean=float(y_train.mean()),
    )
    save_artifacts(
        model=model,
        config=config,
        metrics=metrics,
        n_train_rows=len(X_train),
        n_valid_rows=len(X_valid),
    )

    print("Training complete.")
    print(f"Model saved to: {config.model_path}")
    print(f"Metadata saved to: {config.meta_path}")
    print(
        "Validation metrics: "
        f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
        f"R2={metrics['r2']:.6f}, baseline_MAE={metrics['baseline_mae']:.6f}"
    )
    print(f"Best iteration: {model.get_best_iteration()}")


if __name__ == "__main__":
    main()
