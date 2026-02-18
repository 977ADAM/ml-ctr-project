# ================================
# TRAIN LINEAR REGRESSION MODELS
# ================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------
# 1. Load dataset
# ------------------------------

DATA_PATH = "/mnt/data/dataset.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------------------
# 2. Define targets
# ------------------------------

TARGET_CTR = "CTR"
TARGET_CLICKS = "Clicks"  # или "Переходы"

if TARGET_CTR not in df.columns or TARGET_CLICKS not in df.columns:
    raise ValueError("Проверь названия целевых колонок!")

# ------------------------------
# 3. Split features / targets
# ------------------------------

X = df.drop([TARGET_CTR, TARGET_CLICKS], axis=1)
y_ctr = df[TARGET_CTR]
y_clicks = df[TARGET_CLICKS]

# ------------------------------
# 4. Identify feature types
# ------------------------------

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ------------------------------
# 5. Preprocessing
# ------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ------------------------------
# 6. Build pipelines
# ------------------------------

pipeline_ctr = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

pipeline_clicks = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

# ------------------------------
# 7. Train/Test split
# ------------------------------

X_train, X_test, y_ctr_train, y_ctr_test = train_test_split(
    X, y_ctr, test_size=0.2, random_state=42
)

_, _, y_clicks_train, y_clicks_test = train_test_split(
    X, y_clicks, test_size=0.2, random_state=42
)

# ------------------------------
# 8. Train models
# ------------------------------

pipeline_ctr.fit(X_train, y_ctr_train)
pipeline_clicks.fit(X_train, y_clicks_train)

# ------------------------------
# 9. Evaluate
# ------------------------------

def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    print(f"\n{name} metrics:")
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R2:", r2_score(y_test, preds))

evaluate(pipeline_ctr, X_test, y_ctr_test, "CTR")
evaluate(pipeline_clicks, X_test, y_clicks_test, "Clicks")

# ------------------------------
# 10. Save models
# ------------------------------

with open("linear_regression_ctr.pkl", "wb") as f:
    pickle.dump(pipeline_ctr, f)

with open("linear_regression_clicks.pkl", "wb") as f:
    pickle.dump(pipeline_clicks, f)

print("\nModels saved:")
print(" - linear_regression_ctr.pkl")
print(" - linear_regression_clicks.pkl")

# ------------------------------
# 11. Example inference
# ------------------------------

# пример предсказания
sample = X_test.iloc[:5]
ctr_pred = pipeline_ctr.predict(sample)
clicks_pred = pipeline_clicks.predict(sample)

print("\nSample predictions:")
print("CTR:", ctr_pred)
print("Clicks:", clicks_pred)
