from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    test_size: float = 0.2
    seed: int = 42
    emb_dim: int = 8
    hidden: tuple = (256, 512)
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 150
    device: str = "cpu"
    weight_decay: float = 1e-6

    early_stopping_patience = 20
    early_stopping_min_delta = 1e-4

    MODEL_DIR = "pytorch/models"
    MODEL_NAME = "model.pt"
    META_NAME = "meta.json"
    DATA_PATH = "data/dataset.csv"
