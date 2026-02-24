import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import joblib

df = pd.read_csv("data/dataset.csv")

TARGET = "CTR"

# Удаляем строки без таргета
df = df.dropna(subset=[TARGET])

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Определяем категориальные признаки
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Заполняем пропуски
X[cat_features] = X[cat_features].fillna("missing")
X[num_features] = X[num_features].fillna(0)

# =========================
# 3. Train / Validation split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

# =========================
# 4. Обучение модели
# =========================
model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    early_stopping_rounds=200,
    verbose=200
)

model.fit(train_pool, eval_set=val_pool)

# =========================
# 5. Оценка качества
# =========================
preds = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, preds))
r2 = r2_score(y_val, preds)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# =========================
# 6. Сохранение модели
# =========================
model.save_model("ctr_model.cbm")
joblib.dump(cat_features, "cat_features.pkl")

print("Модель сохранена.")