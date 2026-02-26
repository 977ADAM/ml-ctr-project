from __future__ import annotations

import copy
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import GroupKFold

try:
    from .config import Config
    from .model import CTRNet
    from .utils import (set_seed, sigmoid_np, prepare_targets,
                    binomial_logloss, binomial_nll_from_logits,
                    make_loader, fit_mappings, transform_cats)
    
except ImportError:
    from config import Config
    from model import CTRNet
    from utils import (set_seed, sigmoid_np, prepare_targets,
                    binomial_logloss, binomial_nll_from_logits,
                    make_loader, fit_mappings, transform_cats)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("optuna_tuner")

def _hidden_from_trial(trial: optuna.Trial) -> Tuple[int, ...]:
    n_layers = trial.suggest_int("n_layers", 1, 4)
    h: List[int] = []
    for i in range(n_layers):
        h_i = trial.suggest_categorical(f"h{i}", [16, 32, 64, 128, 256, 512])
        h.append(int(h_i))
    return tuple(h)


def suggest_hparams(trial: optuna.Trial, base_cfg: Config) -> Config:
    emb_dim = int(trial.suggest_categorical("emb_dim", [8, 16, 32, 64]))
    hidden = _hidden_from_trial(trial)
    dropout = float(trial.suggest_float("dropout", 0.0, 0.5))
    lr = float(trial.suggest_float("lr", 1e-4, 3e-3, log=True))
    batch_size = int(trial.suggest_categorical("batch_size", [64, 128, 256, 512]))
    weight_decay = float(trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
    trial.set_user_attr("weight_decay", weight_decay)

    return replace(
        base_cfg,
        emb_dim=emb_dim,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )




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


class Objective:

    def __init__(self, df, cat_cols, impr_col, click_col,
                 base_cfg, group_cols, n_splits):
        self.df = df
        self.cat_cols = cat_cols
        self.impr_col = impr_col
        self.click_col = click_col
        self.base_cfg = base_cfg
        self.group_cols = group_cols
        self.n_splits = n_splits

    def __call__(self, trial: optuna.Trial) -> float:
        # 1. Сэмплируем гиперпараметры
        cfg = self.suggest_hparams(trial)
        weight_decay = cfg.weight_decay
        set_seed(cfg.seed)
        # 2. Формируем группы для GroupKFold
        groups = (
            self.df[list(self.group_cols)]
            .astype(str)
            .agg("_".join, axis=1)
            .values
        )
        gkf = GroupKFold(n_splits=self.n_splits)
        fold_scores: List[float] = []
        # 3. Кросс-валидация
        for fold, (train_idx, valid_idx) in enumerate(
            gkf.split(self.df, y=None, groups=groups),
            start=1,
        ):
            df_train = self.df.iloc[train_idx].reset_index(drop=True)
            df_valid = self.df.iloc[valid_idx].reset_index(drop=True)

            best_valid, *_ = _train_eval_one_split(
                df_train=df_train,
                df_valid=df_valid,
                cat_cols=self.cat_cols,
                impr_col=self.impr_col,
                click_col=self.click_col,
                cfg=cfg,
                weight_decay=weight_decay,
                trial=None,
            )

            fold_scores.append(best_valid)

            # 4. Репортим mean-to-date для pruning
            mean_so_far = float(np.mean(fold_scores))
            trial.report(mean_so_far, step=fold)

            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Обрезано в месте сгиба={fold} mean_logloss={mean_so_far:.6f}"
                )
        
        return float(np.mean(fold_scores))
    
    def suggest_hparams(self, trial: optuna.Trial) -> Config:
        emb_dim = int(trial.suggest_categorical("emb_dim", [8, 16, 32, 64]))
        hidden = _hidden_from_trial(trial)
        dropout = float(trial.suggest_float("dropout", 0.0, 0.5))
        lr = float(trial.suggest_float("lr", 1e-4, 3e-3, log=True))
        batch_size = int(trial.suggest_categorical("batch_size", [64, 128, 256, 512]))
        weight_decay = float(trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))

        return replace(
            self.base_cfg,
            emb_dim=emb_dim,
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
        )















def _mappings_to_save(mappings: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {k: {"classes": v["classes"]} for k, v in mappings.items()}

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
        mode: str,
        out_dir: str,
        study_name: str = "ctrnet_optuna",
        n_splits: int = 5,
        trials: int = 50,
        timeout: Optional[int] = None,
    ):
    df = pd.read_csv("data/dataset.csv")
    base_cfg = Config()

    cat_cols = ["ID кампании", "ID баннера", "Тип баннера", "Тип устройства"]
    impr_col = "Показы"
    click_col = "Переходы"
    group_cols = ["ID кампании", "ID баннера"]

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,
    )

    objective = Objective(df, cat_cols, impr_col, click_col,
                      base_cfg, group_cols, n_splits)
    
    study.optimize(objective, n_trials=trials, timeout=timeout,
                gc_after_trial=True, show_progress_bar=False)
    
    best = study.best_trial
    best_params = dict(best.params)
    best_params["weight_decay"] = float(best.user_attrs.get("weight_decay", 0.0))

    result = {
        "best_value": float(best.value),
        "best_params": best_params,
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "mode": mode,
        "group_cols": list(group_cols) if group_cols else None,
        "n_splits": n_splits
    }

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    (out_dir_p / "optuna_study_best.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    tuned_cfg = replace(
        base_cfg,
        emb_dim=int(best_params["emb_dim"]),
        hidden=tuple(int(best_params[k]) for k in sorted([k for k in best_params if k.startswith("h")], key=lambda s: int(s[1:]))),
        dropout=float(best_params["dropout"]),
        lr=float(best_params["lr"]),
        batch_size=int(best_params["batch_size"]),
    )

    if len(tuned_cfg.hidden) == 0:
        tuned_cfg = replace(tuned_cfg, hidden=(64, 32))

    retrain_and_save_best_gkf(df, cat_cols, impr_col, click_col, out_dir, tuned_cfg, best_params["weight_decay"])

    logger.info(f"Best logloss: {result['best_value']:.6f}")
    logger.info(f"Best params: {json.dumps(best_params, ensure_ascii=False)}")
    logger.info(f"Saved artifacts to: {out_dir_p.resolve()}")

    return result

if __name__ == "__main__":
    run_optuna(
        mode="gkf",
        out_dir="pytorch/optuna_results/gkf",
        n_splits=5,
        trials=80,
        timeout=None,
    )
