import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

df = pd.read_csv('../data/dataset.csv')

target_col = "CTR"
leakage_cols = "Переходы"

y = df[target_col]

X = df.drop(columns=[target_col, leakage_cols])

cat_cols = ['Тип баннера', 'Тип устройства']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_cols,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=False,
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=200,
    use_best_model=True,
    verbose=200
)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

r2 = r2_score(y_test, preds)

print(f"mae: {mae}")
print(f"r2: {r2}")

model.save_model("models/model.cbm")