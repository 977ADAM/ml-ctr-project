import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==============================
# 1. Модель (та же архитектура)
# ==============================

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),

            nn.ReLU(),
            nn.Linear(128, 64),

            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)


# ==============================
# 2. Загрузка модели
# ==============================

@st.cache_resource
def load_model():
    checkpoint = torch.load("ctr_regression_model.pth", map_location="cpu", weights_only=False)

    model = RegressionModel(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return (
        model,
        checkpoint["y_scaler"],
        checkpoint["X_scaler"],
        checkpoint["input_dim"],
        checkpoint["feature_columns"]
    )


model, y_scaler, X_scaler, input_dim, feature_columns = load_model()

st.title("CTR Regression Model")
st.write("Загрузите CSV-файл для предсказания CTR и Переходов")

st.write(feature_columns)
# ==============================
# 3. Загрузка файла
# ==============================

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.write("Исходные данные:")
    st.dataframe(df.head())

    # Кодирование категориальных
    X = pd.get_dummies(df, drop_first=True)

    # Выравнивание признаков
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    # оставляем только нужные
    X = X[feature_columns]

    # масштабирование
    X = X_scaler.transform(X)

    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        outputs = model(inputs).numpy()

    # Возвращаем в исходный масштаб
    predictions = y_scaler.inverse_transform(outputs)

    result_df = df.copy()
    result_df["Predicted_CTR"] = predictions[:, 0]
    result_df["Predicted_Переходы"] = predictions[:, 1]

    st.write("Предсказания:")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )
