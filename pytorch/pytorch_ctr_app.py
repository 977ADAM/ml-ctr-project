import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Импортируем функции из вашего файла
try:
    from inference import load_model, encode_row
    from utils import sigmoid_np
except ImportError:
    from .inference import load_model, encode_row
    from .utils import sigmoid_np


st.set_page_config(page_title="CTR Predictor", layout="wide")
st.title("📊 CTR Model Test Interface")

@st.cache_resource
def load_ctr_model():
    model, meta, device = load_model()
    return model, meta, device

model, meta, device = load_ctr_model()

# ===============================
# Helpers: экономика
# ===============================
def economics_from_ctr(
    ctr: np.ndarray,
    impressions: np.ndarray,
    cpc: np.ndarray,
    cpm: np.ndarray,
):
    """
    Возвращает dict с метриками векторно.
    ctr in [0..1]
    impressions >= 0
    cpc, cpm >= 0
    """
    impressions = impressions.astype(float)
    cpc = cpc.astype(float)
    cpm = cpm.astype(float)

    expected_clicks = ctr * impressions
    revenue = expected_clicks * cpc
    cost = impressions / 1000.0 * cpm
    profit = revenue - cost

    # ROI = profit / cost
    roi = np.where(cost > 0, profit / cost, 0.0)

    # Break-even CTR: при заданных cpm и cpc
    break_even_ctr = np.where(cpc > 0, (cpm / 1000.0) / cpc, 0.0)

    # Break-even CPM: максимальный CPM при котором profit=0 (для данного ctr и cpc)
    # profit = ctr*imp*cpc - imp/1000*cpm => cpm_be = ctr*cpc*1000
    break_even_cpm = ctr * cpc * 1000.0

    # eCPM (ожидаемый доход на 1000 показов)
    ecpm = ctr * cpc * 1000.0

    # Маржинальность
    margin = np.where(revenue > 0, profit / revenue, 0.0)

    return {
        "Expected_Clicks": expected_clicks,
        "Revenue": revenue,
        "Cost": cost,
        "Profit": profit,
        "ROI": roi,
        "BreakEven_CTR": break_even_ctr,
        "BreakEven_CPM": break_even_cpm,
        "eCPM": ecpm,
        "Margin": margin,
    }

def recommended_cpm_cap(ctr: np.ndarray, cpc: np.ndarray, target_roi: float):
    """
    Из формулы ROI = (ctr*cpc*1000 / cpm) - 1
    ROI >= target_roi  =>  cpm <= ctr*cpc*1000 / (1+target_roi)
    """
    denom = 1.0 + float(target_roi)
    return (ctr * cpc * 1000.0) / denom

def action_label(profit: np.ndarray, roi: np.ndarray, target_roi: float):
    """
    Простая логика действий:
    - profit <= 0: Stop
    - profit > 0, но roi < target: Reduce bid
    - иначе: OK
    """
    out = np.full(profit.shape, "OK", dtype=object)
    out = np.where(profit <= 0, "STOP", out)
    out = np.where((profit > 0) & (roi < target_roi), "REDUCE_BID", out)
    return out


st.sidebar.header("Настройки")
mode = st.sidebar.radio("Режим:", ["Одна запись", "Batch CSV"])

st.sidebar.subheader("Цели по экономике")
target_roi = st.sidebar.number_input("Target ROI (например 0.2 = 20%)", min_value=0.0, value=0.2, step=0.05)

# ===============================
# 🔹 ОДНА ЗАПИСЬ
# ===============================
if mode == "Одна запись":
    st.header("Введите параметры")

    row = {}
    cols = st.columns(max(1, len(meta["cat_cols"])))

    for i, col in enumerate(meta["cat_cols"]):
        classes = meta["mappings"][col]["classes"][1:]  # без UNK
        row[col] = cols[i % len(cols)].selectbox(col, classes)

    impressions = st.number_input("Показы", min_value=1, value=1000)
    cpc = st.number_input("CPC (доход за клик)", min_value=0.0, value=10.0)
    cpm = st.number_input("CPM (стоимость 1000 показов)", min_value=0.0, value=200.0)

    if st.button("Предсказать CTR"):
        x = encode_row(row, meta)
        xb = torch.tensor([x], dtype=torch.long).to(device)

        with torch.no_grad():
            logit = model(xb).cpu().numpy()
            ctr = float(sigmoid_np(logit)[0])

        econ = economics_from_ctr(
            ctr=np.array([ctr]),
            impressions=np.array([impressions], dtype=float),
            cpc=np.array([cpc], dtype=float),
            cpm=np.array([cpm], dtype=float),
        )

        expected_clicks = float(econ["Expected_Clicks"][0])
        revenue = float(econ["Revenue"][0])
        cost = float(econ["Cost"][0])
        profit = float(econ["Profit"][0])
        roi = float(econ["ROI"][0])
        break_even_ctr = float(econ["BreakEven_CTR"][0])
        break_even_cpm = float(econ["BreakEven_CPM"][0])
        ecpm = float(econ["eCPM"][0])
        margin = float(econ["Margin"][0])

        # Рекомендованный CPM cap под target ROI
        cpm_cap = float(recommended_cpm_cap(np.array([ctr]), np.array([cpc]), target_roi)[0])

        st.success(f"🎯 CTR: {ctr:.4%}")
        st.info(f"Ожидаемые клики: {expected_clicks:.2f}")

        st.subheader("💰 Экономика")
        col1, col2, col3 = st.columns(3)
        col1.metric("Доход", f"{revenue:.2f}")
        col2.metric("Расход", f"{cost:.2f}")
        col3.metric("Прибыль", f"{profit:.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("ROI", f"{roi:.2%}")
        col5.metric("Break-even CTR", f"{break_even_ctr:.4%}")
        col6.metric("eCPM", f"{ecpm:.2f}")

        col7, col8, col9 = st.columns(3)
        col7.metric("Break-even CPM", f"{break_even_cpm:.2f}")
        col8.metric(f"Реком. CPM cap (ROI≥{target_roi:.0%})", f"{cpm_cap:.2f}")
        col9.metric("Margin", f"{margin:.2%}")

        # Вердикт
        if profit <= 0:
            st.error("⛔ Убыточно: рекомендую STOP или снижение CPM.")
        elif roi < target_roi:
            st.warning("⚠ Прибыльно, но ROI ниже цели: рекомендую REDUCE BID (снизить CPM).")
        else:
            st.success("✅ ОК: кампания проходит по прибыли и целевому ROI.")

        if ctr < break_even_ctr:
            st.warning("⚠ CTR ниже точки безубыточности при текущих CPC/CPM.")
        else:
            st.info("CTR выше точки безубыточности.")

# ===============================
# 🔹 BATCH CSV
# ===============================
if mode == "Batch CSV":
    st.header("Загрузите CSV файл")
    uploaded_file = st.file_uploader("CSV файл", type="csv")

    st.caption("Для экономических метрик в CSV нужны колонки: Показы, CPC, CPM (можно русские/англ. варианты).")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        required_cols = meta["cat_cols"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"В CSV должны быть колонки (категориальные фичи): {required_cols}")
        else:
            rows = df[required_cols].to_dict(orient="records")
            X = np.stack([encode_row(r, meta) for r in rows])
            xb = torch.tensor(X, dtype=torch.long).to(device)

            with torch.no_grad():
                logits = model(xb).cpu().numpy()
                ctr = sigmoid_np(logits).astype(float)

            df["Predicted_CTR"] = ctr

            # Попробуем найти колонки показов/цен (поддержка рус/англ)
            def pick_col(candidates):
                for c in candidates:
                    if c in df.columns:
                        return c
                return None

            col_imp = pick_col(["Показы", "Impressions", "impressions", "shows"])
            col_cpc = pick_col(["CPC", "cpc"])
            col_cpm = pick_col(["CPM", "cpm"])

            with st.expander("Фильтры и отображение", expanded=True):
                show_only = st.selectbox("Показывать строки:", ["Все", "Только OK", "Только REDUCE_BID", "Только STOP"])
                topn = st.number_input("Top-N по прибыли", min_value=5, max_value=500, value=50, step=5)

            # Экономика если есть входные колонки
            if col_imp and col_cpc and col_cpm:
                econ = economics_from_ctr(
                    ctr=df["Predicted_CTR"].to_numpy(float),
                    impressions=df[col_imp].to_numpy(float),
                    cpc=df[col_cpc].to_numpy(float),
                    cpm=df[col_cpm].to_numpy(float),
                )

                for k, v in econ.items():
                    df[k] = v

                df["Rec_CPM_Cap_TargetROI"] = recommended_cpm_cap(
                    df["Predicted_CTR"].to_numpy(float),
                    df[col_cpc].to_numpy(float),
                    target_roi,
                )

                df["Action"] = action_label(df["Profit"].to_numpy(float), df["ROI"].to_numpy(float), target_roi)

                # Агрегаты
                st.subheader("💰 Итоги по экономике")
                total_revenue = float(df["Revenue"].sum())
                total_cost = float(df["Cost"].sum())
                total_profit = float(df["Profit"].sum())
                total_roi = (total_profit / total_cost) if total_cost > 0 else 0.0

                ok_share = float((df["Action"] == "OK").mean())
                reduce_share = float((df["Action"] == "REDUCE_BID").mean())
                stop_share = float((df["Action"] == "STOP").mean())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Общий доход", f"{total_revenue:.2f}")
                c2.metric("Общий расход", f"{total_cost:.2f}")
                c3.metric("Общая прибыль", f"{total_profit:.2f}")
                c4.metric("Общий ROI", f"{total_roi:.2%}")

                c5, c6, c7 = st.columns(3)
                c5.metric("Доля OK", f"{ok_share:.2%}")
                c6.metric("Доля REDUCE_BID", f"{reduce_share:.2%}")
                c7.metric("Доля STOP", f"{stop_share:.2%}")

                st.caption(
                    "Интерпретация: "
                    "STOP — убыточно; REDUCE_BID — прибыльно, но ROI ниже цели; OK — проходит по цели."
                )

            # Фильтрация таблицы
            df_view = df
            if "Action" in df.columns:
                if show_only == "Только OK":
                    df_view = df[df["Action"] == "OK"]
                elif show_only == "Только REDUCE_BID":
                    df_view = df[df["Action"] == "REDUCE_BID"]
                elif show_only == "Только STOP":
                    df_view = df[df["Action"] == "STOP"]

            st.subheader("📋 Результаты")
            st.dataframe(df_view, use_container_width=True)

            # Топ по прибыли (если посчитана)
            if "Profit" in df.columns:
                st.subheader(f"🏆 Top-{int(topn)} по прибыли")
                df_top = df.sort_values("Profit", ascending=False).head(int(topn))
                st.dataframe(df_top, use_container_width=True)

            st.subheader("📈 Распределение CTR")
            fig, ax = plt.subplots()
            ax.hist(df["Predicted_CTR"], bins=30)
            ax.set_xlabel("CTR")
            ax.set_ylabel("Количество")
            ax.set_title("Гистограмма распределения CTR")
            st.pyplot(fig)

            # Доп графики, если есть экономика
            if "Profit" in df.columns:
                st.subheader("📉 Profit распределение")
                fig2, ax2 = plt.subplots()
                ax2.hist(df["Profit"].to_numpy(float), bins=30)
                ax2.set_xlabel("Profit")
                ax2.set_ylabel("Количество")
                ax2.set_title("Гистограмма распределения Profit")
                st.pyplot(fig2)

            # Скачать
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Скачать результаты", csv, "predictions_with_econ.csv", "text/csv")