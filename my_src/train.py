import os
import json

from datetime import datetime, timezone
from xml.parsers.expat import model
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.base import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

try:
    from .config import config
    from .schema import FEATURE_SCHEMA
except ImportError:
    from config import config
    from schema import FEATURE_SCHEMA


def load_data(path, target_col=config.target) -> str:
    df = pd.read_csv(path)

    y = df[target_col].astype(int).values

    df = df.drop(columns=[target_col])

    X = df

    return X, y

def build_model():
    model = CatBoostRegressor(
        iterations=config.iterations,
        depth=config.depth,
        learning_rate=config.learning_rate,
        loss_function=config.loss_function,
        eval_metric=config.eval_metric,
        random_seed=config.random_seed,
        verbose=config.verbose
    )

    return model


def save_model(model, model_path=config.model_path):

    joblib.dump(model, model_path)

    metadata = {
        "version": config.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(config.metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def main():

    X, y = load_data(config.dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    model = build_model()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    

    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))



    save_model(
        model=model,
        metrics={
            "accuracy": acc,
        },
        params={
            "random_state": config.random_state,
            "test_size": config.test_size,
        },
        feature_schema = {

        },
        version=config.version,
    )

if __name__ == "__main__":
    main()