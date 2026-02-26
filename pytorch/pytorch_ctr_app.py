import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path

# Импортируем функции из вашего файла
try:
    from pytorch_ctr_model import encode_row
    from inference import load_model
    from utils import sigmoid_np
except ImportError:
    from .pytorch_ctr_model import encode_row
    from .inference import load_model
    from .utils import sigmoid_np

    
st.set_page_config(page_title="CTR Predictor", layout="wide")

st.title("📊 CTR Model Test Interface")

MODEL_DIR = "pytorch/models"
MODEL_NAME = "model.pt"
META_NAME = "meta.json"


@st.cache_resource
def load_ctr_model():
    model, meta, device = load_model(model_dir=MODEL_DIR, meta_name=META_NAME, model_name=MODEL_NAME)
    return model, meta, device


model, meta, device = load_ctr_model()

st.sidebar.header("Настройки")
mode = st.sidebar.radio("Режим:", ["Одна запись", "Batch CSV"])


# ===============================
# 🔹 ОДНА ЗАПИСЬ
# ===============================
if mode == "Одна запись":

    st.header("Введите параметры")

    row = {}

    cols = st.columns(len(meta["cat_cols"]))

    for i, col in enumerate(meta["cat_cols"]):
        classes = meta["mappings"][col]["classes"][1:]  # без UNK
        row[col] = cols[i].selectbox(col, classes)

    impressions = st.number_input("Показы", min_value=1, value=1000)

    if st.button("Предсказать CTR"):
        x = encode_row(row, meta)
        xb = torch.tensor([x], dtype=torch.long).to(device)

        with torch.no_grad():
            logit = model(xb).cpu().numpy()
            ctr = sigmoid_np(logit)[0]

        expected_clicks = ctr * impressions

        st.success(f"🎯 CTR: {ctr:.4%}")
        st.info(f"Ожидаемые клики: {expected_clicks:.2f}")


# ===============================
# 🔹 BATCH CSV
# ===============================
if mode == "Batch CSV":

    st.header("Загрузите CSV файл")

    uploaded_file = st.file_uploader("CSV файл", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        required_cols = meta["cat_cols"]

        if not all(col in df.columns for col in required_cols):
            st.error(f"В CSV должны быть колонки: {required_cols}")
        else:
            rows = df[required_cols].to_dict(orient="records")

            X = np.stack([encode_row(r, meta) for r in rows])
            xb = torch.tensor(X, dtype=torch.long).to(device)

            with torch.no_grad():
                logits = model(xb).cpu().numpy()
                ctr = sigmoid_np(logits)

            df["Predicted_CTR"] = ctr
            if "Показы" in df.columns:
                df["Expected_Clicks"] = df["Predicted_CTR"] * df["Показы"]

            st.dataframe(df)

            st.subheader("📈 Распределение CTR")
            st.bar_chart(df["Predicted_CTR"])

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Скачать результаты",
                csv,
                "predictions.csv",
                "text/csv"
            )