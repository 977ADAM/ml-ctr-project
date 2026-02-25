import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from lightgbm import LGBMClassifier

def add_ctr_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["impressions"] = df["impressions"].astype(int)
    df["clicks"] = df["clicks"].astype(int)
    df["ctr"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0.0)
    return df

def make_features(df: pd.DataFrame, target_cols=("clicks", "impressions")):
    # Простейший авто-отбор: все кроме target_cols
    drop = set(target_cols)
    feat_cols = [c for c in df.columns if c not in drop]
    X = df[feat_cols].copy()
    return X, feat_cols

def prepare_target(df: pd.DataFrame):
    """
    Для агрегированных данных сделаем:
    y = 1 если был хотя бы 1 клик в группе, иначе 0,
    а вес = impressions (чтобы большие группы влияли сильнее).

    Это приближение, но на практике работает хорошо с бустингом.
    """
    y = (df["clicks"] > 0).astype(int).values
    w = df["impressions"].astype(float).values
    return y, w

def weighted_logloss(y_true, y_pred, sample_weight):
    # y_pred expected prob
    return log_loss(y_true, y_pred, sample_weight=sample_weight, labels=[0,1])

# -----------------------------
# Train
# -----------------------------
def train_ctr_model(
    df: pd.DataFrame,
    categorical_cols=None,
    model_path="ctr_lgbm.joblib",
    meta_path="ctr_lgbm_meta.json",
    random_state=42
):
    df = add_ctr_columns(df)

    # Убираем строки с нулевыми показами
    df = df[df["impressions"] > 0].reset_index(drop=True)

    X, feat_cols = make_features(df)
    y, w = prepare_target(df)

    # Если есть категории - приводим к category dtype (важно для LGBM)
    if categorical_cols:
        for c in categorical_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, w, test_size=0.2, random_state=random_state, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_samples=50,
        objective="binary",
        random_state=random_state,
    )

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="binary_logloss",
        callbacks=[]
    )

    # Оценка
    p_val = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p_val, sample_weight=w_val)
    wll = weighted_logloss(y_val, p_val, w_val)

    # Сохраняем
    joblib.dump(model, model_path)
    meta = {
        "feature_columns": feat_cols,
        "categorical_columns": categorical_cols or [],
        "metrics": {"val_auc_weighted": float(auc), "val_logloss_weighted": float(wll)},
        "notes": "Aggregated CTR: y=(clicks>0), weights=impressions. Predicts P(at least one click)."
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return model, meta

# -----------------------------
# Predict
# -----------------------------
def load_ctr_model(model_path="ctr_lgbm.joblib", meta_path="ctr_lgbm_meta.json"):
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

def predict_ctr(df: pd.DataFrame, model, meta: dict) -> pd.DataFrame:
    df = df.copy()

    X = df[meta["feature_columns"]].copy()
    for c in meta.get("categorical_columns", []):
        if c in X.columns:
            X[c] = X[c].astype("category")

    # Это вероятность "в группе будет хотя бы 1 клик".
    # Для маленьких impressions это близко к CTR, для больших — нет.
    # Ниже покажу как привести к CTR.
    p_any_click = model.predict_proba(X)[:, 1]
    df["p_any_click"] = p_any_click

    # Аппроксимация CTR из p_any_click при заданных impressions:
    # Если предположить клики ~ Binomial(n=impr, p=ctr),
    # то P(clicks>=1) = 1 - (1-ctr)^impr  => ctr = 1 - (1 - P_any)^(1/impr)
    if "impressions" in df.columns:
        impr = np.maximum(df["impressions"].astype(float).values, 1.0)
        df["ctr_pred"] = 1.0 - np.power(1.0 - np.clip(p_any_click, 1e-9, 1-1e-9), 1.0 / impr)
    else:
        df["ctr_pred"] = np.nan

    return df