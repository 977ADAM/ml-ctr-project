# =========================
# 1. Импорт библиотек
# =========================
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 2. Загрузка данных
# =========================
df = pd.read_csv("dataset.csv")

# =========================
# 3. Подготовка данных
# =========================

# Убираем ID (они не несут полезной информации)
df = df.drop(columns=["ID кампании", "ID баннера"])

# Целевая переменная
y = df["CTR"]

# Признаки
X = df.drop(columns=["CTR"])

# Категориальные признаки
cat_features = ["Тип баннера", "Тип устройства"]

# =========================
# 4. Разделение на train/test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Обучение модели
# =========================
model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    early_stopping_rounds=100
)

# =========================
# 6. Оценка качества
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R2: {r2:.4f}")
