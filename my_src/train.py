import json
from datetime import datetime, timezone
import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

try:
    from .config import config
    from .schema import FEATURE_SCHEMA
except ImportError:
    from config import config
    from schema import FEATURE_SCHEMA


def load_data(path, target_col=config.target) -> str:
    df = pd.read_csv(path)

    y = df[target_col].astype(int).values

    X = df.drop(columns=[target_col])

    return X, y

def build_model():
    model = CatBoostClassifier(
        iterations=config.iterations,
        depth=config.depth,
        learning_rate=config.learning_rate,
        loss_function=config.loss_function,
        eval_metric=config.eval_metric,
        random_seed=config.random_seed,
        verbose=config.verbose
    )
    return model


def save_model(model, model_path=config.model_path):

    joblib.dump(model, model_path)

    metadata = {
        "version": config.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(config.metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def main():

    X, y = load_data(config.dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )

    cat_features = FEATURE_SCHEMA.categorical

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_features
    )

    test_pool = Pool(
        X_test,
        y_test,
        cat_features=cat_features
    )

    model = build_model()

    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=config.early_stopping_rounds,
        use_best_model=True
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"AUC: {auc:.4f}")
    print(f"LogLoss: {logloss:.4f}")

    save_model(model)

if __name__ == "__main__":
    main()