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
    batch_size: int = 64
    epochs: int = 100
    device: str = "cpu"

    early_stopping_patience = 15
    early_stopping_min_delta = 1e-4
