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
    batch_size: int = 128
    epochs: int = 25
    device: str = "cpu"
