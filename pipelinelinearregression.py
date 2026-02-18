import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ==========================
# 1. Загрузка данных
# ==========================

DATA_PATH = "dataset.csv"
TARGET_COLUMN = "CTR"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# Проверка наличия таргета
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

# Убедимся, что CTR числовой
df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

# ==========================
# 2. Разделение признаков
# ==========================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# ==========================
# 3. Определение типов признаков
# ==========================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ==========================
# 4. Препроцессинг
# ==========================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ==========================
# 5. Pipeline
# ==========================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# ==========================
# 6. Train / Test split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 7. Обучение
# ==========================

model.fit(X_train, y_train)

# ==========================
# 8. Оценка
# ==========================

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel performance:")
print(f"RMSE: {rmse:.6f}")
print(f"R2: {r2:.6f}")

# ==========================
# 9. Сохранение
# ==========================

MODEL_PATH = "linear_regression_ctr.pkl"
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
