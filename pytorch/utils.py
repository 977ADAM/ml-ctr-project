import pandas as pd
import numpy as np
import torch
from typing import Tuple


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def prepare_targets(df: pd.DataFrame, impr_col: str, click_col: str) -> Tuple[np.ndarray, np.ndarray]:
    impr = df[impr_col].astype(np.float32).values
    clicks = df[click_col].astype(np.float32).values
    return clicks, impr