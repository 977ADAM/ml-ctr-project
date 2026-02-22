from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path = Path("data/dataset.csv")
    model_path: Path = Path("models/model.cbm")
    meta_path: Path = Path("models/model_meta.json")
    test_size: float = 0.2
    random_state: int = 42
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "ctr-click-probability"
    mlflow_run_name: str = "catboost_ctr"
    mlflow_registered_model_name: str = "ctr_click_probability_model"


def get_train_config() -> TrainConfig:
    return TrainConfig()
