from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

MODEL_PATH = Path("models/model.cbm")
DEFAULT_DATA_PATH = Path("data/dataset.csv")
FEATURE_COLUMNS = ["ID кампании", "ID баннера", "Тип баннера", "Тип устройства", "Показы"]


@st.cache_resource
def load_model(model_path: Path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model


@st.cache_data
def load_reference_data(data_path: Path) -> pd.DataFrame:
    if data_path.exists():
        return pd.read_csv(data_path)
    return pd.DataFrame()


def make_decision(pred_ctr: float, cost_per_impression: float, click_value: float) -> dict[str, float | str]:
    expected_value = pred_ctr * click_value
    margin = expected_value - cost_per_impression
    decision = "Покупать показ" if margin >= 0 else "Не покупать"
    return {
        "expected_value": expected_value,
        "margin": margin,
        "max_cpm": expected_value * 1000,
        "decision": decision,
    }


def main() -> None:
    st.set_page_config(page_title="CTR Decision App", layout="wide")
    st.title("CTR Model: прогноз и решение о покупке показа")

    if not MODEL_PATH.exists():
        st.error(
            "Модель не найдена. Сначала обучите ее командой: `venv/bin/python src/train.py`"
        )
        st.stop()

    model = load_model(MODEL_PATH)
    ref_df = load_reference_data(DEFAULT_DATA_PATH)

    st.sidebar.header("Экономика показа")
    cost_per_impression = st.sidebar.number_input(
        "Стоимость 1 показа", min_value=0.0, value=0.002, step=0.0001, format="%.6f"
    )
    click_value = st.sidebar.number_input(
        "Ценность 1 клика", min_value=0.0, value=1.0, step=0.1, format="%.4f"
    )
    st.sidebar.caption(
        "Правило: покупаем, если `predicted_ctr * ценность_клика >= стоимость_показа`."
    )

    tab_single = st.tabs(["Один показ"])

    with tab_single:
        st.subheader("Ручной прогноз")

        default_campaign = int(ref_df["ID кампании"].mode().iloc[0]) if not ref_df.empty else 0
        default_banner = int(ref_df["ID баннера"].mode().iloc[0]) if not ref_df.empty else 0
        banner_types = sorted(ref_df["Тип баннера"].dropna().unique().tolist()) if not ref_df.empty else ["interactive"]
        device_types = sorted(ref_df["Тип устройства"].dropna().unique().tolist()) if not ref_df.empty else ["Смартфон"]
        default_shows = int(ref_df["Показы"].median()) if not ref_df.empty else 1000

        col1, col2, col3 = st.columns(3)
        with col1:
            campaign_id = st.number_input("ID кампании", min_value=0, value=default_campaign, step=1)
            banner_type = st.selectbox("Тип баннера", options=banner_types)
        with col2:
            banner_id = st.number_input("ID баннера", min_value=0, value=default_banner, step=1)
            device_type = st.selectbox("Тип устройства", options=device_types)
        with col3:
            shows = st.number_input("Показы", min_value=1, value=default_shows, step=1)

        if st.button("Рассчитать", type="primary"):
            input_df = pd.DataFrame(
                [
                    {
                        "ID кампании": int(campaign_id),
                        "ID баннера": int(banner_id),
                        "Тип баннера": banner_type,
                        "Тип устройства": device_type,
                        "Показы": int(shows),
                    }
                ]
            )

            pred_ctr = model.predict(input_df)[0]
            pred_ctr = max(0, min(1, pred_ctr))
            predicted_clicks = shows * pred_ctr
            business = make_decision(pred_ctr, cost_per_impression, click_value)

            m1, m2, m3 = st.columns(3)
            m1.metric("Вероятность клика (CTR)", f"{pred_ctr:.4%}")
            m2.metric("Ожидаемые клики", f"{predicted_clicks:.2f}")
            m3.metric("Макс CPM", f"{business['max_cpm']:.4f}")

            st.write(f"Ожидаемая ценность показа: **{business['expected_value']:.6f}**")
            st.write(f"Маржа показа: **{business['margin']:.6f}**")
            if business["decision"] == "Покупать показ":
                st.success("Решение: Покупать показ")
            else:
                st.error("Решение: Не покупать")

if __name__ == "__main__":
    main()
