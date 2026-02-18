import pandas as pd
from catboost import CatBoostRegressor
import streamlit as st
try:
    from .config import config
except ImportError:
    from config import config

df = pd.read_csv(config.dataset)

st.caption(f"Model version: {config.version}")
st.divider()

model = CatBoostRegressor()
model.load_model("model.cbm")

banner_type = st.selectbox(
    'Тип баннера', options=df["Тип баннера"].unique()
)
divace_type = st.selectbox(
    "Тип устройства", options=df["Тип устройства"].unique()
)


if st.button('Predict'):
    with st.spinner("Выполняется анализ транзакции..."):
        pass
