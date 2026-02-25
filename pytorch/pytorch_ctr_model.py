# train_ctr_agg.py
# pip install pandas numpy scikit-learn torch

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


# ---------- utils ----------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def binomial_logloss(clicks: np.ndarray, impr: np.ndarray, p: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1 - eps)
    ll = clicks * np.log(p) + (impr - clicks) * np.log(1 - p)
    return float(-ll.sum() / impr.sum())  # среднее на 1 показ


# ---------- model ----------
class CTRNet(nn.Module):
    """
    Эмбеддинги для категориальных фич + MLP -> logit(p).
    """
    def __init__(self, cardinalities, emb_dim=16, hidden=(64, 32), dropout=0.1):
        super().__init__()

        self.embs = nn.ModuleList()
        emb_out = 0
        for c in cardinalities:
            d = min(emb_dim, int(math.ceil(c ** 0.25) * 4))  # норм эвристика
            self.embs.append(nn.Embedding(c, d))
            emb_out += d

        layers = []
        in_dim = emb_out
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        z = torch.cat(embs, dim=1)
        logit = self.mlp(z).squeeze(1)
        return logit


def binomial_nll_from_logits(logits, clicks, impr):
    """
    NLL для биномиального распределения:
    -[k*log(sigmoid)+ (n-k)*log(1-sigmoid)]
    Реализовано стабильно через log-sigmoid.
    """
    # log(sigmoid(x)) = -softplus(-x)
    # log(1-sigmoid(x)) = -softplus(x)
    nll = clicks * torch.nn.functional.softplus(-logits) + (impr - clicks) * torch.nn.functional.softplus(logits)
    return nll.sum() / impr.sum()  # среднее на 1 показ


# ---------- training ----------
@dataclass
class TrainConfig:
    test_size: float = 0.2
    seed: int = 42
    emb_dim: int = 16
    hidden: tuple = (64, 32)
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 168
    device: str = "cpu"

def make_loader(X_cat, clicks, impr, batch_size, shuffle=True):
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(clicks, dtype=torch.float32),
        torch.tensor(impr, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def fit_mappings(df: pd.DataFrame, cat_cols):
    """
    Фитим словари по train. Индекс 0 зарезервирован под UNK.
    """
    mappings = {}
    for col in cat_cols:
        uniq = pd.Index(df[col].astype(str).unique())
        # UNK=0, остальные с 1
        classes = ["__UNK__"] + uniq.tolist()
        value_to_idx = {v: i for i, v in enumerate(classes)}
        mappings[col] = {"classes": classes, "value_to_idx": value_to_idx}
    return mappings

def transform_cats(df: pd.DataFrame, cat_cols, mappings) -> np.ndarray:

    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)

    for j, col in enumerate(cat_cols):
        m = mappings[col]["value_to_idx"]
        vals = df[col].astype(str).values
        # unknown -> 0
        X_cat[:, j] = np.fromiter((m.get(v, 0) for v in vals), dtype=np.int64, count=len(vals))
    return X_cat

def prepare_targets(df: pd.DataFrame, impr_col, click_col):
    impr = df[impr_col].astype(np.float32).values
    clicks = df[click_col].astype(np.float32).values

    # базовые проверки (лучше держать включёнными)
    # if np.any(impr <= 0):
    #     raise ValueError("Найдены строки с Показы <= 0 — их нужно удалить/исправить.")
    # if np.any(clicks < 0) or np.any(clicks > impr):
    #     raise ValueError("Найдены некорректные Переходы (clicks) относительно Показы (impr).")
    return clicks, impr

def train_one_run(csv_path="dataset.csv", out_dir="pytorch/models"):
    cfg = TrainConfig()
    set_seed(cfg.seed)

    df = pd.read_csv(csv_path)

    # Под ваш датасет (как в примере):
    cat_cols = ["ID кампании", "ID баннера", "Тип баннера", "Тип устройства"]
    impr_col = "Показы"
    click_col = "Переходы"

    clicks, impr = prepare_targets(df, impr_col, click_col)
    df_tr, df_te, c_tr, c_te, n_tr, n_te = train_test_split(
        df, clicks, impr, test_size=cfg.test_size, random_state=cfg.seed
    )

    mappings = fit_mappings(df_tr, cat_cols)

    X_tr = transform_cats(df_tr, cat_cols, mappings)
    X_te = transform_cats(df_te, cat_cols, mappings)

    cardinalities = [len(mappings[col]["classes"]) for col in cat_cols]
    model = CTRNet(cardinalities, emb_dim=cfg.emb_dim, hidden=cfg.hidden, dropout=cfg.dropout).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    tr_loader = make_loader(X_tr, c_tr, n_tr, cfg.batch_size, shuffle=True)
    te_loader = make_loader(X_te, c_te, n_te, cfg.batch_size, shuffle=False)

    best_val = 1e9
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, kb, nb in tr_loader:
            xb, kb, nb = xb.to(cfg.device), kb.to(cfg.device), nb.to(cfg.device)
            opt.zero_grad()
            logits = model(xb)
            loss = binomial_nll_from_logits(logits, kb, nb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        model.eval()
        with torch.no_grad():
            # оценим logloss на 1 показ
            all_logits = []
            all_k = []
            all_n = []
            for xb, kb, nb in te_loader:
                xb = xb.to(cfg.device)
                logits = model(xb).detach().cpu().numpy()
                all_logits.append(logits)
                all_k.append(kb.numpy())
                all_n.append(nb.numpy())

            logits = np.concatenate(all_logits)
            k = np.concatenate(all_k)
            n = np.concatenate(all_n)
            p = sigmoid_np(logits)
            val = binomial_logloss(k, n, p)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss/len(tr_loader):.6f} | val_logloss={val:.6f}")

        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), out_dir / "model.pt")

    # сохраняем метаданные (маппинги категорий + список колонок)
    # value_to_idx в json не пишем (он восстановится из classes)
    mappings_to_save = {k: {"classes": v["classes"]} for k, v in mappings.items()}
    meta = {
        "cat_cols": cat_cols,
        "impr_col": impr_col,
        "click_col": click_col,
        "mappings": mappings_to_save,
        "cardinalities": cardinalities,
        "arch": {"emb_dim": cfg.emb_dim, "hidden": list(cfg.hidden), "dropout": cfg.dropout},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved to: {out_dir.resolve()}")
    print(f"Best val logloss: {best_val:.6f}")


# ---------- inference ----------
def load_model(model_dir="ctr_model", device=None):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = CTRNet(
        meta["cardinalities"],
        emb_dim=meta["arch"]["emb_dim"],
        hidden=tuple(meta["arch"]["hidden"]),
        dropout=meta["arch"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.eval()
    return model, meta, device


def encode_row(row: dict, meta: dict) -> np.ndarray:
    x = []
    for col in meta["cat_cols"]:
        classes = meta["mappings"][col]["classes"]
        v = str(row.get(col, ""))
        # unknown -> 0 (можно сделать отдельный UNK, но для простоты так)
        idx = classes.index(v) if v in classes else 0
        x.append(idx)
    return np.array(x, dtype=np.int64)


@torch.no_grad()
def predict_ctr(rows, model_dir="ctr_model"):
    model, meta, device = load_model(model_dir=model_dir)

    X = np.stack([encode_row(r, meta) for r in rows], axis=0)

    xb = torch.tensor(X, dtype=torch.long).to(device)

    logits = model(xb).detach().cpu().numpy()

    p = sigmoid_np(logits)

    return p


if __name__ == "__main__":
    # Пример запуска обучения:
    # python train_ctr_agg.py
    train_one_run(csv_path="data/dataset.csv", out_dir="pytorch/models")

    # Пример инференса:
    rows = [
        {"ID кампании": 3405596, "ID баннера": 15262577, "Тип баннера": "interactive", "Тип устройства": "Компьютер", "Показы": 12596},
        {"ID кампании": 9, "ID баннера": 9, "Тип баннера": "interactive", "Тип устройства": "Смартфон", "Показы": 500},
    ]
    preds = predict_ctr(rows, model_dir="ctr_model")
    print("Pred CTR 2144:", preds)