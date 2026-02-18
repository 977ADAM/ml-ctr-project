import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.catboost
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .config import config
    from .schema import FEATURE_SCHEMA
except ImportError:
    from config import config
    from schema import FEATURE_SCHEMA

st.set_page_config(page_title="CTR Model Trainer", layout="wide")

st.title("📊 CTR Prediction — CatBoost Trainer")

st.sidebar.header("⚙ Настройки модели")

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, config.test_size, 0.05)
iterations = st.sidebar.slider("Iterations", 500, 5000, config.iterations, 100)
depth = st.sidebar.slider("Depth", 4, 10, config.depth)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, config.learning_rate)

uploaded_file = st.file_uploader("📂 Загрузите CSV файл", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Данные")
    st.dataframe(df.head())

    if config.target not in df.columns:
        st.error(f"Target column '{config.target}' not found in dataset")
        st.stop()

    df = df.drop(columns=FEATURE_SCHEMA.drop_columns, errors="ignore")

    y = df[config.target]
    X = df.drop(columns=[config.target])

    cat_features = [c for c in FEATURE_SCHEMA.categorical if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.random_state
    )

    if st.button("🚀 Обучить модель"):
        with st.spinner("Обучение модели..."):
            mlflow.set_experiment("CTR_Streamlit_Experiment")

            model = CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                loss_function=config.loss_function,
                eval_metric=config.eval_metric,
                random_seed=config.random_seed,
                verbose=False
            )

            model.fit(
                X_train,
                y_train,
                cat_features=cat_features,
                eval_set=(X_test, y_test),
                early_stopping_rounds=config.early_stopping_rounds,
                verbose=False
            )

            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            with mlflow.start_run():
                mlflow.log_params({
                    "iterations": iterations,
                    "depth": depth,
                    "learning_rate": learning_rate,
                    "test_size": test_size,
                    "loss_function": config.loss_function,
                    "eval_metric": config.eval_metric,
                })
                mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
                mlflow.catboost.log_model(model, "catboost_model")

        st.success("✅ Обучение завершено!")

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.6f}")
        col2.metric("RMSE", f"{rmse:.6f}")
        col3.metric("R2", f"{r2:.4f}")

        # График
        st.subheader("📈 Фактические vs Предсказанные")
        chart_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        })

        st.line_chart(chart_df.reset_index(drop=True))

else:
    st.info("Загрузите CSV файл для начала работы.")
