from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict click probability (CTR)")
    parser.add_argument("--input", type=Path, required=True, help="CSV for inference")
    parser.add_argument("--model", type=Path, default=Path("models/ctr_model.cbm"))
    parser.add_argument("--output", type=Path, default=Path("models/predictions.csv"))
    parser.add_argument("--cost-per-impression", type=float, default=None)
    parser.add_argument("--click-value", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    # Keep the same inference features used in training.
    drop_cols = [c for c in ["CTR", "Переходы"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    model = CatBoostRegressor()
    model.load_model(args.model)

    pred_ctr = np.clip(model.predict(X), 0.0, 1.0)
    result = df.copy()
    result["predicted_ctr"] = pred_ctr

    if "Показы" in result.columns:
        result["predicted_clicks"] = (result["Показы"] * result["predicted_ctr"]).round(0).astype(int)

    if args.cost_per_impression is not None:
        result["expected_value_per_impression"] = result["predicted_ctr"] * args.click_value
        result["impression_cost"] = args.cost_per_impression
        result["buy_impression"] = (
            result["expected_value_per_impression"] >= result["impression_cost"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
