#!/usr/bin/env python3
"""
Uzum Bank — Churn Analytics Dashboard
Run: streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_loader import load_data

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Uzum Bank · Churn Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Column name helpers
# Column names changed in model.py:
#   churn_proba      → dormant_30d_proba
#   predicted_churn  → predicted_dormant
#   cnt_m0           → cnt_early
#   n_cats_m0        → n_cats_early
#   activated_m0     → activated_early
# We resolve dynamically so the dashboard works with both old and new CSVs.
# ---------------------------------------------------------------------------

def _col(scored, new_name, old_name):
    """Return new_name if present, else old_name (backwards compat)."""
    return new_name if new_name in scored.columns else old_name

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Загрузка данных…")
def load_all():
    df      = load_data(fallback_path="data/uzum_hackathon_dataset.csv")
    scored  = pd.read_csv("results/scored_cards.csv")  if Path("results/scored_cards.csv").exists()  else None
    metrics = json.load(open("results/model_metrics.json")) if Path("results/model_metrics.json").exists() else None
    diag    = json.load(open("results/diagnostics.json"))   if Path("results/diagnostics.json").exists()   else None
    bonus   = pd.read_csv("results/bonus_candidates.csv")  if Path("results/bonus_candidates.csv").exists() else None
    return df, scored, metrics, diag, bonus


df, scored, metrics, diag, bonus = load_all()

# Resolve column names once
if scored is not None:
    COL_PROBA    = _col(scored, 'dormant_30d_proba', 'churn_proba')
    COL_PREDICT  = _col(scored, 'predicted_dormant',  'predicted_churn')
    COL_CNT      = _col(scored, 'cnt_early',           'cnt_m0')
    COL_NCATS    = _col(scored, 'n_cats_early',         'n_cats_m0')
    COL_ACTIVATED = _col(scored, 'activated_early',     'activated_m0')
else:
    COL_PROBA = COL_PREDICT = COL_CNT = COL_NCATS = COL_ACTIVATED = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("""
<div style="padding: 8px 0 4px;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <span style="font-size:1.6rem;">🏦</span>
        <span style="font-size:1.1rem;font-weight:700;color:#ffffff;letter-spacing:-0.01em;">Uzum Bank</span>
    </div>
    <div style="font-size:0.78rem;color:#595b66;font-weight:500;letter-spacing:0.06em;text-transform:uppercase;">Churn Analytics</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.divider()

page = st.sidebar.radio(
    "Раздел",
    ["Обзор", "Модель", "Скоринг карт", "Бонусы", "Регионы"],
    label_visibility="collapsed",
)

st.sidebar.divider()
if scored is not None:
    total    = len(scored)
    at_risk  = scored[COL_PREDICT].sum()
    st.sidebar.metric("Всего карт",         f"{total:,}")
    st.sidebar.metric("Под риском оттока",  f"{at_risk:,}",
                      delta=f"{100*at_risk/total:.1f}%", delta_color="inverse")

if metrics:
    st.sidebar.metric("AUC-ROC модели", f"{metrics['auc_roc']:.3f}")

# ---------------------------------------------------------------------------
# Helper colours
# ---------------------------------------------------------------------------

RISK_COLORS = {
    "CRITICAL": "#cc0000",
    "HIGH":     "#ff8800",
    "MEDIUM":   "#e56f00",
    "LOW":      "#008a32",
}

SEGMENT_COLORS = {
    "Stable active":           "#00ad3a",
    "Late activation, stable": "#7f4dff",
    "Fast start / fast sleep": "#ff8800",
    "Early start, unstable":   "#cc0000",
    "One-time use":            "#7000ff",
    "Never transacted":        "#4d4f59",
    "Other":                   "#dee0e5",
}

# Feature labels for model.py output (new column names)
FEATURE_LABELS = {
    # Activation date
    "creation_day":         "День создания карты",
    "creation_dow":         "День недели создания",
    "creation_is_weekend":  "Создана в выходной",
    # Early window behavior
    "activated_early":      "Активирован (0–14 дней)",
    "activated_week1":      "Активирован (неделя 1)",
    "cnt_early":            "Кол-во транзакций (0–14 дней)",
    "cnt_week1":            "Кол-во транзакций (неделя 1)",
    "amt_early":            "Сумма транзакций (0–14 дней)",
    "n_cats_early":         "Разнообразие MCC (0–14 дней)",
    "n_cats_week1":         "Разнообразие MCC (неделя 1)",
    "days_to_first_txn":    "Дней до первой транзакции",
    # First transaction channel
    "first_txn_online":     "Первый канал: онлайн",
    "first_txn_offline":    "Первый канал: офлайн",
    "first_txn_transfer":   "Первый канал: перевод",
    "first_txn_cash":       "Первый канал: наличные",
    "first_txn_other":      "Первый канал: другое",
    # Channel mix
    "online_share":         "Доля онлайн-платежей",
    "offline_share":        "Доля офлайн-платежей",
    "transfer_share":       "Доля переводов",
    "cash_share":           "Доля снятия наличных",
    # Legacy (old model.py) — kept for backwards compat
    "cnt_m0":               "Кол-во транзакций (мес. 0)",
    "activated_m0":         "Активирован в мес. 0",
    "amt_m0":               "Сумма транзакций (мес. 0)",
    "n_cats_m0":            "Кол-во категорий",
}

# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================

if page == "Обзор":
    st.title("Аналитика оттока карт")
    st.caption("Дашборд по предсказанию 'засыпания' дебетовых карт · Uzum Bank")

    total_cards = diag['total_cards']          if diag else 0
    ever        = diag['ever_txn_cards']       if diag else 0
    rate        = diag['ever_txn_rate_pct']    if diag else 0
    active_n    = int(diag['total_cards'] * diag['is_active_target_pct'] / 100) if diag else 0
    pct         = diag['is_active_target_pct'] if diag else 0
    churned     = diag['churned_cards']        if diag else 0
    auc         = metrics['auc_roc']           if metrics else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Карт всего",      f"{total_cards:,}")
    with k2: st.metric("Платили хоть раз", f"{ever:,}", delta=f"{rate:.1f}%")
    with k3: st.metric("Активных карт",   f"{active_n:,}", delta=f"{pct:.1f}%")
    with k4: st.metric("Засыпающих",      f"{churned:,}")
    with k5: st.metric("AUC-ROC",         f"{auc:.3f}")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Поведенческие сегменты")
        if diag:
            seg_df = pd.DataFrame(
                [(k, v) for k, v in diag["segments"].items()],
                columns=["segment", "count"]
            ).sort_values("count", ascending=True)

            colors = [SEGMENT_COLORS.get(s, "#bdc3c7") for s in seg_df["segment"]]
            fig = go.Figure(go.Bar(
                x=seg_df["count"], y=seg_df["segment"],
                orientation="h",
                marker_color=colors,
                text=seg_df["count"].apply(lambda v: f"{v:,}"),
                textposition="outside",
            ))
            fig.update_layout(
                margin=dict(l=0, r=60, t=20, b=0), height=320, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#dee0e5"),
            )
            st.plotly_chart(fig)

    with col_right:
        st.subheader("Распределение по уровням риска")
        if scored is not None:
            risk_counts = scored["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "count"]
            order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            risk_counts["risk_level"] = pd.Categorical(
                risk_counts["risk_level"], categories=order, ordered=True
            )
            risk_counts = risk_counts.sort_values("risk_level")

            fig = go.Figure(go.Pie(
                labels=risk_counts["risk_level"],
                values=risk_counts["count"],
                hole=0.55,
                marker_colors=[RISK_COLORS[r] for r in risk_counts["risk_level"]],
                textinfo="label+percent",
                textfont_size=13,
                textfont_color="#4d4f59",
            ))
            fig.update_layout(
                margin=dict(l=0, r=0, t=20, b=0), height=320,
                showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig)

    st.subheader("Распределение вероятности засыпания")
    if scored is not None:
        fig = px.histogram(
            scored, x=COL_PROBA, nbins=50,
            color_discrete_sequence=["#7f4dff"],
            labels={COL_PROBA: "P(dormant)", "count": "Карт"},
        )
        thr = metrics["threshold"] if metrics else 0.5
        fig.add_vline(x=thr, line_dash="dash", line_color="#cc0000",
                      annotation_text=f"Порог {thr:.2f}", annotation_position="top right")
        fig.update_layout(
            height=260, margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig)


# ===========================================================================
# PAGE 2 — MODEL
# ===========================================================================

elif page == "Модель":
    st.title("Модель предсказания засыпания")

    if not metrics:
        st.warning("Запустите `python3 model.py` для генерации метрик.")
        st.stop()

    st.subheader("Качество модели (Decision Tree, test set)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC",   f"{metrics['auc_roc']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall",    f"{metrics['recall']:.3f}")
    m4.metric("F1-score",  f"{metrics['f1']:.3f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Важность признаков")
        fi = pd.DataFrame(
            [(k, v) for k, v in metrics["feature_importance"].items()],
            columns=["feature", "importance"]
        ).sort_values("importance")
        fi["label"] = fi["feature"].map(FEATURE_LABELS).fillna(fi["feature"])

        fig = go.Figure(go.Bar(
            x=fi["importance"], y=fi["label"],
            orientation="h",
            marker_color=["#7f4dff" if v > 0.01 else "#dee0e5" for v in fi["importance"]],
            text=fi["importance"].apply(lambda v: f"{v:.3f}"),
            textposition="outside",
        ))
        fig.update_layout(
            height=420, margin=dict(l=0, r=60, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, fi["importance"].max() * 1.25]),
        )
        st.plotly_chart(fig)

    with col2:
        st.subheader("Обоснование порога классификации")
        thr = metrics["threshold"]
        p, r, f = metrics["precision"], metrics["recall"], metrics["f1"]

        st.markdown(f"""
**Выбранный порог: `{thr:.3f}`**

При этом пороге:
- **Precision** {p:.3f} — из всех предсказанных засыпаний {p*100:.1f}% реальные
- **Recall** {r:.3f} — модель ловит {r*100:.1f}% всех реальных засыпаний
- **F1** {f:.3f} — баланс точности и полноты

**Почему именно F1?**

В задаче предотвращения засыпания:
- **FN** (пропустить уходящего) = потеря клиента → дорого
- **FP** (лишний бонус активному) = малые затраты → дёшево

Поэтому **Recall важнее Precision**, но F1 выбран как базовый оптимум.
Снижение порога повышает Recall за счёт Precision — допустимо,
если стоимость триггера мала по сравнению с LTV карты.
        """)

        # Read real PR curve from saved metrics; fall back to a single point.
        pr_curve = metrics.get("pr_curve") or []
        if pr_curve:
            thresholds_demo = [pt["threshold"] for pt in pr_curve]
            precisions_demo = [pt["precision"] for pt in pr_curve]
            recalls_demo    = [pt["recall"]    for pt in pr_curve]
        else:
            thresholds_demo = [thr]
            precisions_demo = [p]
            recalls_demo    = [r]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds_demo, y=precisions_demo,
                                 name="Precision", line=dict(color="#7f4dff", width=2)))
        fig.add_trace(go.Scatter(x=thresholds_demo, y=recalls_demo,
                                 name="Recall", line=dict(color="#cc0000", width=2)))
        fig.add_vline(x=thr, line_dash="dash", line_color="#00ad3a",
                      annotation_text="Выбранный порог", annotation_position="top left")
        fig.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.15),
            xaxis_title="Порог", yaxis_title="Метрика",
        )
        st.plotly_chart(fig)

    # Confusion matrix — read directly from saved JSON (no hardcoded math)
    st.subheader("Матрица ошибок (test set)")
    if "confusion_matrix" in metrics:
        cm   = metrics["confusion_matrix"]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    else:
        # Fallback estimate for old metrics files that lack confusion_matrix key
        total_test = sum(metrics.get("confusion_matrix", {}).values()) or 3494
        churn_rate = metrics.get("churn_rate", metrics.get("dormancy_rate", 0.5))
        tp = int(round(metrics["recall"]    * total_test * churn_rate))
        fp = int(round((1 - metrics["precision"]) / max(metrics["precision"], 1e-9) * tp))
        fn = int(round(total_test * churn_rate - tp))
        tn = total_test - tp - fp - fn

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Факт: остался", "Факт: засыпание"],
        columns=["Предсказан: остался", "Предсказан: засыпание"],
    )
    fig = px.imshow(
        cm_df, text_auto=True,
        color_continuous_scale=["#f2f4f7", "#7f4dff", "#7000ff"],
        labels=dict(color="Карт"),
    )
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)


# ===========================================================================
# PAGE 3 — SCORING
# ===========================================================================

elif page == "Скоринг карт":
    st.title("Скоринг карт по риску засыпания")

    if scored is None:
        st.warning("Запустите `python3 model.py` для генерации scored_cards.csv")
        st.stop()

    f1, f2, f3 = st.columns(3)
    with f1:
        risk_filter = st.multiselect(
            "Уровень риска",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH"],
        )
    with f2:
        region_filter = st.multiselect(
            "Регион",
            sorted(scored["kiosk_name"].unique()),
            default=[],
        )
    with f3:
        proba_range = st.slider("P(dormant) диапазон", 0.0, 1.0, (0.0, 1.0), 0.01)

    mask = (
        scored["risk_level"].isin(risk_filter) &
        scored[COL_PROBA].between(*proba_range)
    )
    if region_filter:
        mask &= scored["kiosk_name"].isin(region_filter)

    filtered = scored[mask].copy()
    st.caption(f"Показано: **{len(filtered):,}** из {len(scored):,} карт")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Карт", f"{len(filtered):,}")
    c2.metric("Средний P(dormant)", f"{filtered[COL_PROBA].mean():.3f}")
    c3.metric("Не активированы", f"{int((filtered[COL_ACTIVATED] == 0).sum()):,}")
    c4.metric("Ср. транзакций (0–14 дн.)", f"{filtered[COL_CNT].mean():.1f}")

    st.subheader("P(dormant) vs транзакции в первые 14 дней")
    fig = px.scatter(
        filtered.sample(min(3000, len(filtered)), random_state=42),
        x=COL_CNT, y=COL_PROBA,
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        opacity=0.5,
        labels={COL_CNT: "Транзакций (0–14 дней)", COL_PROBA: "P(dormant)"},
        hover_data=["card_id", "kiosk_name", COL_NCATS],
    )
    if metrics:
        fig.add_hline(y=metrics["threshold"], line_dash="dash", line_color="#7000ff",
                      annotation_text="Порог", annotation_position="right")
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)

    st.subheader("Таблица карт")

    def _style_risk(val):
        colors_map = {"CRITICAL": "#fff5f5", "HIGH": "#fff8f0",
                      "MEDIUM": "#fffdf0", "LOW": "#f5fbf6"}
        return f"color: {colors_map.get(val, '')}"

    display_cols = ["card_id", "kiosk_name", COL_PROBA, "risk_level",
                    COL_CNT, COL_NCATS, COL_ACTIVATED]
    display_cols = [c for c in display_cols if c in filtered.columns]

    display_df = (
        filtered[display_cols]
        .sort_values(COL_PROBA, ascending=False)
        .head(500)
        .rename(columns={
            "card_id":       "Карта",
            "kiosk_name":    "Регион",
            COL_PROBA:       "P(dormant)",
            "risk_level":    "Риск",
            COL_CNT:         "Транзакций",
            COL_NCATS:       "Категорий",
            COL_ACTIVATED:   "Активирован",
        })
    )
    st.dataframe(display_df.style.applymap(_style_risk, subset=["Риск"]), height=400)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Скачать отфильтрованные карты (CSV)", csv_bytes,
                       "filtered_cards.csv", "text/csv")


# ===========================================================================
# PAGE 4 — BONUSES
# ===========================================================================

elif page == "Бонусы":
    st.title("Рекомендации по бонусам")

    if bonus is None or len(bonus) == 0:
        st.info("Запустите `python3 bonus_logic.py` для генерации рекомендаций.")
        if scored is not None:
            st.subheader("Предпросмотр: карты HIGH/CRITICAL риска")
            preview = scored[scored["risk_level"].isin(["CRITICAL", "HIGH"])].head(10)
            st.dataframe(preview)
        st.stop()

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Кандидатов на бонус", f"{len(bonus):,}")
    b2.metric("CRITICAL", f"{(bonus['risk_level']=='CRITICAL').sum():,}")
    b3.metric("HIGH",     f"{(bonus['risk_level']=='HIGH').sum():,}")
    b4.metric("Ср. P(dormant)", f"{bonus['dormant_proba'].mean():.3f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Топ категории среди кандидатов")
        cat_counts = bonus["top_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig = px.bar(cat_counts, x="count", y="category", orientation="h",
                     color_discrete_sequence=["#7f4dff"],
                     labels={"count": "Карт", "category": ""})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig)

    with col2:
        st.subheader("Когда и как отправить")
        timing_data = (
            bonus.groupby(["risk_level", "channel", "send_day"])
            .size().reset_index(name="n")
        )
        order_map = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        timing_data["order"] = timing_data["risk_level"].map(order_map)
        timing_data = timing_data.sort_values("order")

        fig = px.bar(timing_data, x="risk_level", y="n", color="channel",
                     labels={"n": "Карт", "risk_level": "Уровень риска"},
                     color_discrete_sequence=["#7000ff", "#7f4dff", "#cc0000", "#ff8800"])
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(title="Канал"))
        st.plotly_chart(fig)

    st.subheader("Примеры персонализированных сообщений")
    top5 = bonus.sort_values("dormant_proba", ascending=False).head(5)
    for _, row in top5.iterrows():
        risk_color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(row["risk_level"], "⚪")
        with st.expander(f"{risk_color} {row['card_id']} — P(dormant)={row['dormant_proba']:.2f} | {row['top_category']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Регион:** {row.get('kiosk_name', '—')}")
                st.markdown(f"**Риск:** {row['risk_level']}")
                st.markdown(f"**P(dormant):** {row['dormant_proba']:.3f}")
                st.markdown(f"**Категория:** {row['top_category']}")
            with c2:
                st.info(row["message"])
                st.markdown(f"**Бонус:** {row['bonus_sums']} сум")
                st.markdown(f"**Отправить:** день {row['send_day']} · {row['channel']}")

    st.subheader("Полный список кандидатов")
    st.dataframe(
        bonus.sort_values("dormant_proba", ascending=False)
             [["card_id", "kiosk_name", "dormant_proba", "risk_level",
               "top_category", "bonus_sums", "send_day", "channel", "message"]],
        height=350,
    )
    st.download_button(
        "Скачать список бонусов (CSV)",
        bonus.to_csv(index=False).encode("utf-8"),
        "bonus_candidates.csv", "text/csv",
    )


# ===========================================================================
# PAGE 5 — REGIONS
# ===========================================================================

elif page == "Регионы":
    st.title("Региональный анализ")

    if scored is None:
        st.warning("Нет данных scored_cards.csv")
        st.stop()

    region_stats = (
        scored.groupby("kiosk_name")
        .agg(
            n_cards         = ("card_id", "count"),
            avg_churn_proba = (COL_PROBA, "mean"),
            pct_critical    = ("risk_level", lambda x: (x == "CRITICAL").mean() * 100),
            pct_high        = ("risk_level", lambda x: (x == "HIGH").mean() * 100),
            pct_at_risk     = (COL_PREDICT, "mean"),
            avg_cnt         = (COL_CNT, "mean"),
        )
        .reset_index()
        .sort_values("avg_churn_proba", ascending=False)
    )
    region_stats["pct_at_risk"] *= 100

    st.subheader("Регионы по среднему P(dormant)")
    fig = px.bar(
        region_stats.sort_values("avg_churn_proba"),
        x="avg_churn_proba", y="kiosk_name", orientation="h",
        color="avg_churn_proba",
        color_continuous_scale=["#00ad3a", "#7f4dff", "#cc0000"],
        labels={"avg_churn_proba": "Ср. P(dormant)", "kiosk_name": "Регион"},
        text=region_stats.sort_values("avg_churn_proba")["avg_churn_proba"].apply(lambda v: f"{v:.3f}"),
    )
    fig.update_layout(
        height=max(300, len(region_stats) * 28),
        margin=dict(l=0, r=60, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig)

    st.subheader("Карты vs риск засыпания по регионам")
    fig = px.scatter(
        region_stats,
        x="avg_cnt", y="avg_churn_proba",
        size="n_cards", color="pct_at_risk",
        color_continuous_scale=["#00ad3a", "#7f4dff", "#cc0000"],
        hover_name="kiosk_name",
        labels={
            "avg_cnt":          "Ср. транзакций (0–14 дней)",
            "avg_churn_proba":  "Ср. P(dormant)",
            "pct_at_risk":      "% под риском",
            "n_cards":          "Карт",
        },
        size_max=50,
    )
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(tickfont=dict(color="#4d4f59")),
    )
    st.plotly_chart(fig)

    st.subheader("Детальная таблица по регионам")
    _region_renamed = region_stats.rename(columns={
        "kiosk_name":       "Регион",
        "n_cards":          "Карт",
        "avg_churn_proba":  "Ср. P(dormant)",
        "pct_critical":     "% CRITICAL",
        "pct_high":         "% HIGH",
        "pct_at_risk":      "% под риском",
        "avg_cnt":          "Ср. транзакций",
    })
    _region_styled = _region_renamed.style.format({
        "Ср. P(dormant)": "{:.3f}",
        "% CRITICAL":     "{:.1f}%",
        "% HIGH":         "{:.1f}%",
        "% под риском":   "{:.1f}%",
        "Ср. транзакций": "{:.2f}",
    }).background_gradient(subset=["Ср. P(dormant)"], cmap="Purples")
    st.dataframe(_region_styled, height=420)