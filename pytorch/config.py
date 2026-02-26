from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    test_size: float = 0.2
    seed: int = 42
    emb_dim: int = 16
    hidden: tuple = (64, 32)
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 30
    device: str = "cpu"

    early_stopping_patience = 20
    early_stopping_min_delta = 1e-4

    MODEL_DIR = "pytorch/models"
    MODEL_NAME = "modelgkf.pt"
    META_NAME = "metagkf.json"
    DATA_PATH = "data/dataset.csv"
