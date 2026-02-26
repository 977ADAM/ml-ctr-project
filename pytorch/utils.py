import pandas as pd
import numpy as np
import torch
from typing import Tuple, Dict, Sequence


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

def binomial_logloss(clicks: np.ndarray, impr: np.ndarray, p: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1 - eps)
    ll = clicks * np.log(p) + (impr - clicks) * np.log(1 - p)
    return float(-ll.sum() / impr.sum())

def binomial_nll_from_logits(logits: torch.Tensor, clicks: torch.Tensor, impr: torch.Tensor) -> torch.Tensor:
    # stable:
    # log(sigmoid(x)) = -softplus(-x)
    # log(1-sigmoid(x)) = -softplus(x)
    nll = clicks * torch.nn.functional.softplus(-logits) + (impr - clicks) * torch.nn.functional.softplus(logits)
    return nll.sum() / impr.sum()

def make_loader(X_cat: np.ndarray, clicks: np.ndarray, impr: np.ndarray, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(clicks, dtype=torch.float32),
        torch.tensor(impr, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def fit_mappings(df: pd.DataFrame, cat_cols: Sequence[str]) -> Dict[str, Dict[str, object]]:
    mappings: Dict[str, Dict[str, object]] = {}
    for col in cat_cols:
        uniq = pd.Index(df[col].astype(str).unique())
        classes = ["__UNK__"] + uniq.tolist()  # UNK=0
        value_to_idx = {v: i for i, v in enumerate(classes)}
        mappings[col] = {"classes": classes, "value_to_idx": value_to_idx}
    return mappings

def transform_cats(df: pd.DataFrame, cat_cols: Sequence[str], mappings: Dict[str, Dict[str, object]]) -> np.ndarray:
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for j, col in enumerate(cat_cols):
        m = mappings[col]["value_to_idx"]
        vals = df[col].astype(str).values
        X_cat[:, j] = np.fromiter((m.get(v, 0) for v in vals), dtype=np.int64, count=len(vals))
    return X_cat