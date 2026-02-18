# ==========================================
# 1. Импорт библиотек
# ==========================================
import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ==========================================
# 2. Загрузка данных
# ==========================================
df = pd.read_csv("dataset.csv")

# Удаляем ID
df = df.drop(columns=["ID кампании", "ID баннера"])

y = df["CTR"]
X = df.drop(columns=["CTR"])

cat_features = ["Тип баннера", "Тип устройства"]

# ==========================================
# 3. Настройка K-Fold
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 4. Objective функция для Optuna
# ==========================================
def objective(trial):

    params = {
        "iterations": trial.suggest_int("iterations", 300, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": False
    }

    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**params)

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=False
        )

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# ==========================================
# 5. Запуск оптимизации
# ==========================================
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)  # можно увеличить до 50–100

print("\nЛучшие параметры:")
print(study.best_params)

print("\nЛучший RMSE:")
print(study.best_value)

# ==========================================
# 6. Финальное обучение с лучшими параметрами
# ==========================================
best_params = study.best_params
best_params.update({
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": 100
})

final_model = CatBoostRegressor(**best_params)

final_model.fit(
    X,
    y,
    cat_features=cat_features
)

final_model.save_model("catboost_ctr_optuna.pkl")

print("\nМодель с Optuna сохранена как catboost_ctr_optuna.pkl")
