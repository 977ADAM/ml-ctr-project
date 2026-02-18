import streamlit as st
try:
    from .config import config
except ImportError:
    from config import config


st.caption(f"Model version: {config.version}")
st.divider()

banner_type = st.selectbox(
    'Тип баннера'
)
divace_type = st.selectbox(
    "Тип устройства"
)


if st.button('Predict'):
    with st.spinner("Выполняется анализ транзакции..."):
        pass
