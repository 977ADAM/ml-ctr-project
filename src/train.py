# ==========================================
# 1. Импорт библиотек
# ==========================================

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 2. Загрузка данных
# ==========================================
df = pd.read_csv("dataset.csv")

# ==========================================
# 3. Подготовка данных
# ==========================================

# Удаляем ID (неинформативные признаки)
df = df.drop(columns=["ID кампании", "ID баннера"])

# Целевая переменная
y = df["CTR"]

# Признаки
X = df.drop(columns=["CTR"])

# Категориальные признаки
cat_features = ["Тип баннера", "Тип устройства"]

# ==========================================
# 4. K-Fold кросс-валидация
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
mae_scores = []
r2_scores = []

fold = 1

for train_idx, val_idx in kf.split(X):
    print(f"\n========== Fold {fold} ==========")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100
    )

    # Предсказание
    y_pred = model.predict(X_val)

    # Метрики
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R2:   {r2:.4f}")

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

    fold += 1

# ==========================================
# 5. Итоговые результаты CV
# ==========================================
print("\n========== Итог по 5-Fold ==========")
print(f"Средний RMSE: {np.mean(rmse_scores):.6f}")
print(f"Средний MAE:  {np.mean(mae_scores):.6f}")
print(f"Средний R2:   {np.mean(r2_scores):.4f}")

# ==========================================
# 6. Финальное обучение на всех данных
# ==========================================
final_model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=100
)

final_model.fit(
    X,
    y,
    cat_features=cat_features
)

# ==========================================
# 7. Сохранение модели
# ==========================================
final_model.save_model("catboost_ctr_model.pkl")

print("\nМодель сохранена как catboost_ctr_model.pkl")
