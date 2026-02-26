"""
Optuna tuning module for your CTRNet architecture.

Features
- Supports holdout split (train_test_split) and GroupKFold CV (like your current training code).
- Tunes architecture + optimizer hyperparams: emb_dim, hidden, dropout, lr, weight_decay, batch_size.
- Early stopping + Optuna pruning on validation logloss.
- Saves:
    - optuna_study_best.json (best params + score)
    - optuna_study.sqlite (optional, if --storage specified)
    - Retrained best model + meta in out_dir (same format as your current pipeline).

Usage examples
--------------
Holdout:
python optuna_tuner.py \
  --data data/dataset.csv \
  --cat-cols "ID кампании" "ID баннера" "Тип баннера" "Тип устройства" \
  --impr-col "Показы" --click-col "Переходы" \
  --mode holdout \
  --trials 50 \
  --out-dir pytorch/models_optuna

GroupKFold:
python optuna_tuner.py \
  --data data/dataset.csv \
  --cat-cols "ID кампании" "ID баннера" "Тип баннера" "Тип устройства" \
  --impr-col "Показы" --click-col "Переходы" \
  --mode gkf \
  --group-cols "ID кампании" "ID баннера" \
  --n-splits 5 \
  --trials 80 \
  --out-dir pytorch/models_optuna_gkf

Notes
-----
- This module assumes your model consumes ONLY categorical features already present as columns in --cat-cols.
- Targets are aggregates (clicks, impressions), optimized with binomial NLL / logloss per impression.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split, GroupKFold

try:
    from .config import Config
    from .model import CTRNet
except ImportError:
    from config import Config
    from model import CTRNet

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("optuna_tuner")

# ---------------- utils ----------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

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

def prepare_targets(df: pd.DataFrame, impr_col: str, click_col: str) -> Tuple[np.ndarray, np.ndarray]:
    impr = df[impr_col].astype(np.float32).values
    clicks = df[click_col].astype(np.float32).values
    return clicks, impr

def _hidden_from_trial(trial: optuna.Trial) -> Tuple[int, ...]:
    n_layers = trial.suggest_int("n_layers", 1, 4)
    h: List[int] = []
    for i in range(n_layers):
        # powers of 2 are usually stable for MLP
        h_i = trial.suggest_categorical(f"h{i}", [16, 32, 64, 128, 256, 512])
        h.append(int(h_i))
    return tuple(h)

def suggest_hparams(trial: optuna.Trial, base_cfg: Config) -> Config:
    """
    Return a new Config with tuned params (Config is frozen, so we use dataclasses.replace).
    """
    emb_dim = int(trial.suggest_categorical("emb_dim", [8, 16, 32, 64]))
    hidden = _hidden_from_trial(trial)
    dropout = float(trial.suggest_float("dropout", 0.0, 0.5))
    lr = float(trial.suggest_float("lr", 1e-4, 3e-3, log=True))
    batch_size = int(trial.suggest_categorical("batch_size", [64, 128, 256, 512]))
    # store separately (not part of Config) via trial.user_attrs
    weight_decay = float(trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
    trial.set_user_attr("weight_decay", weight_decay)

    return replace(
        base_cfg,
        emb_dim=emb_dim,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
    )

# ---------------- training core ----------------
def _train_eval_one_split(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    cat_cols: Sequence[str],
    impr_col: str,
    click_col: str,
    cfg: Config,
    weight_decay: float,
    trial: Optional[optuna.Trial] = None,
) -> Tuple[float, Dict[str, torch.Tensor], Dict[str, Dict[str, object]], List[int]]:
    """
    Train on df_train, evaluate on df_val.
    Returns: best_val_logloss, best_state_dict, mappings, cardinalities
    """
    clicks_tr, impr_tr = prepare_targets(df_train, impr_col, click_col)
    clicks_va, impr_va = prepare_targets(df_val, impr_col, click_col)

    mappings = fit_mappings(df_train, cat_cols)
    X_tr = transform_cats(df_train, cat_cols, mappings)
    X_va = transform_cats(df_val, cat_cols, mappings)

    cardinalities = [len(mappings[col]["classes"]) for col in cat_cols]

    device = torch.device(cfg.device)
    model = CTRNet(cardinalities, emb_dim=cfg.emb_dim, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)

    tr_loader = make_loader(X_tr, clicks_tr, impr_tr, cfg.batch_size, shuffle=True)
    va_loader = make_loader(X_va, clicks_va, impr_va, cfg.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, kb, nb in tr_loader:
            xb, kb, nb = xb.to(device), kb.to(device), nb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = binomial_nll_from_logits(logits, kb, nb)
            loss.backward()
            opt.step()

        # eval
        model.eval()
        with torch.no_grad():
            all_logits, all_k, all_n = [], [], []
            for xb, kb, nb in va_loader:
                xb = xb.to(device)
                all_logits.append(model(xb).detach().cpu().numpy())
                all_k.append(kb.numpy())
                all_n.append(nb.numpy())

            logits = np.concatenate(all_logits)
            k = np.concatenate(all_k)
            n = np.concatenate(all_n)
            p = sigmoid_np(logits)
            val = binomial_logloss(k, n, p)

        # report to optuna
        if trial is not None:
            trial.report(val, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch={epoch} val_logloss={val:.6f}")

        # early stopping
        improved = val < (best_val - cfg.early_stopping_min_delta)
        if improved:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                break

    assert best_state is not None
    return best_val, best_state, mappings, cardinalities

def _objective_holdout(
    trial: optuna.Trial,
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    impr_col: str,
    click_col: str,
    base_cfg: Config,
) -> float:
    cfg = suggest_hparams(trial, base_cfg)
    weight_decay = float(trial.user_attrs["weight_decay"])

    set_seed(cfg.seed)

    clicks, impr = prepare_targets(df, impr_col, click_col)
    df_tr, df_va, c_tr, c_va, n_tr, n_va = train_test_split(
        df, clicks, impr, test_size=cfg.test_size, random_state=cfg.seed
    )
    # keep targets inside dfs for convenience
    df_tr = df_tr.copy()
    df_va = df_va.copy()
    df_tr[click_col] = c_tr
    df_tr[impr_col] = n_tr
    df_va[click_col] = c_va
    df_va[impr_col] = n_va

    best_val, *_ = _train_eval_one_split(
        df_tr, df_va, cat_cols, impr_col, click_col, cfg, weight_decay, trial=trial
    )
    return best_val

def _objective_gkf(
    trial: optuna.Trial,
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    impr_col: str,
    click_col: str,
    base_cfg: Config,
    group_cols: Sequence[str],
    n_splits: int,
) -> float:
    cfg = suggest_hparams(trial, base_cfg)
    weight_decay = float(trial.user_attrs["weight_decay"])

    set_seed(cfg.seed)

    groups = df[list(group_cols)].astype(str).agg("_".join, axis=1).values
    gkf = GroupKFold(n_splits=n_splits)

    fold_scores: List[float] = []
    # One shared trial, report mean-to-date (so pruning can act early)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, y=None, groups=groups), start=1):
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        best_val, *_ = _train_eval_one_split(
            df_tr, df_va, cat_cols, impr_col, click_col, cfg, weight_decay, trial=None  # per-epoch pruning handled below
        )
        fold_scores.append(best_val)

        mean_so_far = float(np.mean(fold_scores))
        trial.report(mean_so_far, step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at fold={fold} mean_logloss={mean_so_far:.6f}")

    return float(np.mean(fold_scores))

# ---------------- save / retrain ----------------
def _mappings_to_save(mappings: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {k: {"classes": v["classes"]} for k, v in mappings.items()}

def retrain_and_save_best_holdout(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    impr_col: str,
    click_col: str,
    out_dir: str,
    cfg: Config,
    weight_decay: float,
) -> Dict[str, object]:
    """
    Retrain on full data (no validation) for cfg.epochs, then save model.pt + meta.json in out_dir.
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)

    clicks, impr = prepare_targets(df, impr_col, click_col)
    mappings = fit_mappings(df, cat_cols)
    X = transform_cats(df, cat_cols, mappings)
    cardinalities = [len(mappings[col]["classes"]) for col in cat_cols]

    device = torch.device(cfg.device)
    model = CTRNet(cardinalities, emb_dim=cfg.emb_dim, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)

    loader = make_loader(X, clicks, impr, cfg.batch_size, shuffle=True)

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        for xb, kb, nb in loader:
            xb, kb, nb = xb.to(device), kb.to(device), nb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = binomial_nll_from_logits(logits, kb, nb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        if epoch % 10 == 0:
            logger.info(f"[retrain] epoch={epoch} loss={epoch_loss/len(loader):.6f}")

    torch.save(model.state_dict(), out_dir_p / "model.pt")
    meta = {
        "cat_cols": list(cat_cols),
        "impr_col": impr_col,
        "click_col": click_col,
        "mappings": _mappings_to_save(mappings),
        "cardinalities": cardinalities,
        "arch": {"emb_dim": cfg.emb_dim, "hidden": list(cfg.hidden), "dropout": cfg.dropout},
        "tuning": {"lr": cfg.lr, "batch_size": cfg.batch_size, "weight_decay": weight_decay},
    }
    (out_dir_p / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

def retrain_and_save_best_gkf(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    impr_col: str,
    click_col: str,
    out_dir: str,
    cfg: Config,
    weight_decay: float,
) -> Dict[str, object]:
    """
    For your inference pipeline, you usually save ONE "best overall" model with mappings fitted on training data.
    Here, we fit mappings on FULL data and train ONE final model (not per-fold) and save as modelgkf.pt/metagkf.json
    to keep your current inference naming consistent.
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)

    clicks, impr = prepare_targets(df, impr_col, click_col)
    mappings = fit_mappings(df, cat_cols)
    X = transform_cats(df, cat_cols, mappings)
    cardinalities = [len(mappings[col]["classes"]) for col in cat_cols]

    device = torch.device(cfg.device)
    model = CTRNet(cardinalities, emb_dim=cfg.emb_dim, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)

    loader = make_loader(X, clicks, impr, cfg.batch_size, shuffle=True)

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        for xb, kb, nb in loader:
            xb, kb, nb = xb.to(device), kb.to(device), nb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = binomial_nll_from_logits(logits, kb, nb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        if epoch % 10 == 0:
            logger.info(f"[retrain] epoch={epoch} loss={epoch_loss/len(loader):.6f}")

    torch.save(model.state_dict(), out_dir_p / "modelgkf.pt")
    meta = {
        "cat_cols": list(cat_cols),
        "impr_col": impr_col,
        "click_col": click_col,
        "mappings": _mappings_to_save(mappings),
        "cardinalities": cardinalities,
        "arch": {"emb_dim": cfg.emb_dim, "hidden": list(cfg.hidden), "dropout": cfg.dropout},
        "tuning": {"lr": cfg.lr, "batch_size": cfg.batch_size, "weight_decay": weight_decay},
        "cv": {"final_train": "full_data_single_model"},
    }
    (out_dir_p / "metagkf.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

# ---------------- run tuning ----------------
def run_optuna(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    impr_col: str,
    click_col: str,
    mode: str,
    out_dir: str,
    group_cols: Optional[Sequence[str]] = None,
    n_splits: int = 5,
    trials: int = 50,
    timeout: Optional[int] = None,
    storage: Optional[str] = None,
    study_name: str = "ctrnet_optuna",
    device: Optional[str] = None,
) -> Dict[str, object]:
    base_cfg = Config()
    if device is not None:
        base_cfg = replace(base_cfg, device=device)

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    if mode == "holdout":
        objective = lambda t: _objective_holdout(t, df, cat_cols, impr_col, click_col, base_cfg)
    elif mode == "gkf":
        if not group_cols:
            raise ValueError("mode=gkf requires --group-cols")
        objective = lambda t: _objective_gkf(t, df, cat_cols, impr_col, click_col, base_cfg, group_cols, n_splits)
    else:
        raise ValueError("mode must be one of: holdout, gkf")

    logger.info(f"Starting Optuna: mode={mode} trials={trials} timeout={timeout} out_dir={out_dir}")
    study.optimize(objective, n_trials=trials, timeout=timeout, gc_after_trial=True, show_progress_bar=False)

    best = study.best_trial
    best_params = dict(best.params)
    # weight_decay stored in user attrs
    best_params["weight_decay"] = float(best.user_attrs.get("weight_decay", 0.0))

    result = {
        "best_value": float(best.value),
        "best_params": best_params,
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "mode": mode,
        "group_cols": list(group_cols) if group_cols else None,
        "n_splits": n_splits if mode == "gkf" else None,
    }

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    (out_dir_p / "optuna_study_best.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # retrain final model with best params and save in out_dir
    tuned_cfg = replace(
        base_cfg,
        emb_dim=int(best_params["emb_dim"]),
        hidden=tuple(int(best_params[k]) for k in sorted([k for k in best_params if k.startswith("h")], key=lambda s: int(s[1:]))),
        dropout=float(best_params["dropout"]),
        lr=float(best_params["lr"]),
        batch_size=int(best_params["batch_size"]),
    )

    # if hidden wasn't reconstructed (edge cases), fall back
    if len(tuned_cfg.hidden) == 0:
        tuned_cfg = replace(tuned_cfg, hidden=(64, 32))

    if mode == "holdout":
        retrain_and_save_best_holdout(df, cat_cols, impr_col, click_col, out_dir, tuned_cfg, best_params["weight_decay"])
    else:
        retrain_and_save_best_gkf(df, cat_cols, impr_col, click_col, out_dir, tuned_cfg, best_params["weight_decay"])

    logger.info(f"Best logloss: {result['best_value']:.6f}")
    logger.info(f"Best params: {json.dumps(best_params, ensure_ascii=False)}")
    logger.info(f"Saved artifacts to: {out_dir_p.resolve()}")

    return result

# ---------------- cli ----------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuner for CTRNet (categorical embeddings + MLP).")
    p.add_argument("--data", required=True, help="Path to CSV dataset.")
    p.add_argument("--cat-cols", nargs="+", required=True, help="Categorical feature columns.")
    p.add_argument("--impr-col", required=True, help="Impressions column.")
    p.add_argument("--click-col", required=True, help="Clicks column.")
    p.add_argument("--mode", choices=["holdout", "gkf"], default="holdout")
    p.add_argument("--group-cols", nargs="+", default=None, help="Group columns for GroupKFold.")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--timeout", type=int, default=None, help="Seconds.")
    p.add_argument("--out-dir", default="pytorch/models_optuna")
    p.add_argument("--study-name", default="ctrnet_optuna")
    p.add_argument("--storage", default=None, help='e.g. "sqlite:///optuna_study.sqlite"')
    p.add_argument("--device", default=None, help='e.g. "cuda" or "cpu"')
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)

    # basic sanity: keep only required columns
    needed = set(args.cat_cols + [args.impr_col, args.click_col])
    if args.mode == "gkf" and args.group_cols:
        needed |= set(args.group_cols)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    run_optuna(
        df=df,
        cat_cols=args.cat_cols,
        impr_col=args.impr_col,
        click_col=args.click_col,
        mode=args.mode,
        out_dir=args.out_dir,
        group_cols=args.group_cols,
        n_splits=args.n_splits,
        trials=args.trials,
        timeout=args.timeout,
        storage=args.storage,
        study_name=args.study_name,
        device=args.device,
    )

if __name__ == "__main__":
    main()
