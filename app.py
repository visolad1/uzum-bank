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
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Root & global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f0f2f5; }

    /* ── Main content area ── */
    .main .block-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-top: 1rem;
        box-shadow: 0 2px 12px rgba(31,32,38,0.06);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #1f2026 !important;
        border-right: none !important;
    }
    [data-testid="stSidebar"] * { color: #c2c5cc !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #ffffff !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #7f4dff !important; font-size: 1.4rem !important; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #8b8e99 !important; }
    [data-testid="stSidebar"] [data-testid="metric-container"] {
        background: rgba(127,77,255,0.1) !important;
        border-left: 3px solid #7f4dff !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stRadio > div { gap: 4px; }
    [data-testid="stSidebar"] .stRadio label {
        color: #a6a9b2 !important;
        padding: 8px 12px;
        border-radius: 8px;
        transition: background 0.15s;
    }
    [data-testid="stSidebar"] .stRadio label:hover { background: rgba(127,77,255,0.15) !important; color: #ffffff !important; }

    /* ── KPI cards ── */
    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border-left: 4px solid #7000ff;
        box-shadow: 0 2px 8px rgba(112,0,255,0.10);
        margin-bottom: 8px;
    }
    .kpi-label { font-size: 0.78rem; font-weight: 500; color: #8b8e99; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .kpi-value { font-size: 1.9rem; font-weight: 700; color: #1f2026; line-height: 1; }
    .kpi-sub   { font-size: 0.82rem; color: #7f4dff; font-weight: 500; margin-top: 4px; }

    /* ── Risk labels ── */
    .risk-CRITICAL { color: #cc0000; font-weight: 700; }
    .risk-HIGH     { color: #e56f00; font-weight: 700; }
    .risk-MEDIUM   { color: #cc7700; font-weight: 700; }
    .risk-LOW      { color: #008a32; font-weight: 700; }

    /* ── Streamlit native metric widget (non-KPI pages) ── */
    [data-testid="metric-container"] {
        background: #fafafa;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid #7f4dff;
        box-shadow: 0 1px 4px rgba(127,77,255,0.08);
    }
    [data-testid="stMetricLabel"]  { color: #8b8e99 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"]  { color: #1f2026 !important; font-weight: 700; }
    [data-testid="stMetricDelta"]  { color: #7f4dff !important; }

    /* ── Headings ── */
    h1 { color: #1f2026 !important; font-weight: 700 !important; letter-spacing: -0.02em; }
    h2, h3 { color: #2a2b33 !important; font-weight: 600 !important; }
    .stCaption, caption { color: #8b8e99 !important; }

    /* ── Buttons ── */
    .stDownloadButton > button, .stButton > button {
        background: #7000ff !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.4rem 1.2rem !important;
        transition: background 0.15s !important;
    }
    .stDownloadButton > button:hover, .stButton > button:hover {
        background: #7f4dff !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; padding: 8px 18px; color: #8b8e99; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #7000ff !important; border-bottom: 3px solid #7000ff !important; font-weight: 600 !important; }

    /* ── Divider ── */
    hr { border-color: #edeff2 !important; }

    /* ── Expander ── */
    details summary { color: #7000ff !important; font-weight: 600; }
    details { border: 1px solid #edeff2 !important; border-radius: 10px !important; background: #fafafa !important; }

    /* ── Selectbox / multiselect ── */
    [data-testid="stMultiSelect"] span[aria-selected="true"] { background-color: #7f4dff !important; }
    [data-baseweb="select"] { border-radius: 8px !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden; }

    /* ── Slider accent ── */
    [data-testid="stSlider"] [role="slider"] { background: #7000ff !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Загрузка данных…")
def load_all():
    df = load_data(fallback_path="data/uzum_hackathon_dataset.csv")
    scored = pd.read_csv("results/scored_cards.csv") if Path("results/scored_cards.csv").exists() else None
    metrics = json.load(open("results/model_metrics.json")) if Path("results/model_metrics.json").exists() else None
    diag = json.load(open("results/diagnostics.json")) if Path("results/diagnostics.json").exists() else None
    bonus = pd.read_csv("results/bonus_candidates.csv") if Path("results/bonus_candidates.csv").exists() else None
    return df, scored, metrics, diag, bonus


df, scored, metrics, diag, bonus = load_all()

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
    total = len(scored)
    at_risk = scored["predicted_churn"].sum()
    st.sidebar.metric("Всего карт", f"{total:,}")
    st.sidebar.metric("Под риском оттока", f"{at_risk:,}", delta=f"{100*at_risk/total:.1f}%", delta_color="inverse")

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

# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================

if page == "Обзор":
    st.title("Аналитика оттока карт")
    st.caption("Дашборд по предсказанию 'засыпания' дебетовых карт · Uzum Bank")

    # ── KPI row (HTML cards — no truncation) ────────────────────────────────
    def kpi(label, value, sub=""):
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        return f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>"""

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        total_cards = diag['total_cards'] if diag else 0
        st.markdown(kpi("Карт всего", f"{total_cards:,}"), unsafe_allow_html=True)
    with k2:
        ever = diag['ever_txn_cards'] if diag else 0
        rate = diag['ever_txn_rate_pct'] if diag else 0
        st.markdown(kpi("Платили хоть раз", f"{ever:,}", f"↑ {rate:.1f}% от всех"), unsafe_allow_html=True)
    with k3:
        active_n = int(diag['total_cards'] * diag['is_active_target_pct'] / 100) if diag else 0
        pct = diag['is_active_target_pct'] if diag else 0
        st.markdown(kpi("Активных (is_active=1)", f"{active_n:,}", f"{pct:.1f}%"), unsafe_allow_html=True)
    with k4:
        churned = diag['churned_cards'] if diag else 0
        st.markdown(kpi("Засыпающих (churn)", f"{churned:,}"), unsafe_allow_html=True)
    with k5:
        auc = metrics['auc_roc'] if metrics else 0
        st.markdown(kpi("AUC-ROC модели", f"{auc:.3f}", "Decision Tree"), unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Сегменты карт ────────────────────────────────────────────────────────
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
                margin=dict(l=0, r=60, t=20, b=0),
                height=320, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#dee0e5"),
            )
            st.plotly_chart(fig, width="stretch")

    # ── Risk distribution ────────────────────────────────────────────────────
    with col_right:
        st.subheader("Распределение по уровням риска")
        if scored is not None:
            risk_counts = scored["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "count"]
            order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            risk_counts["risk_level"] = pd.Categorical(risk_counts["risk_level"], categories=order, ordered=True)
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
                margin=dict(l=0, r=0, t=20, b=0),
                height=320,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")

    # ── Churn proba distribution ──────────────────────────────────────────────
    st.subheader("Распределение вероятности оттока")
    if scored is not None:
        fig = px.histogram(
            scored, x="churn_proba", nbins=50,
            color_discrete_sequence=["#7f4dff"],
            labels={"churn_proba": "P(churn)", "count": "Карт"},
        )
        thr = metrics["threshold"] if metrics else 0.5
        fig.add_vline(x=thr, line_dash="dash", line_color="#cc0000",
                      annotation_text=f"Порог {thr:.2f}", annotation_position="top right")
        fig.update_layout(
            height=260, margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")


# ===========================================================================
# PAGE 2 — MODEL
# ===========================================================================

elif page == "Модель":
    st.title("Модель предсказания оттока")

    if not metrics:
        st.warning("Запустите `python3 model.py` для генерации метрик.")
        st.stop()

    # ── Метрики ──────────────────────────────────────────────────────────────
    st.subheader("Качество модели (Decision Tree, test set)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC",   f"{metrics['auc_roc']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall",    f"{metrics['recall']:.3f}")
    m4.metric("F1-score",  f"{metrics['f1']:.3f}")

    st.divider()

    col1, col2 = st.columns(2)

    # ── Feature importance ───────────────────────────────────────────────────
    with col1:
        st.subheader("Важность признаков")
        fi = pd.DataFrame(
            [(k, v) for k, v in metrics["feature_importance"].items()],
            columns=["feature", "importance"]
        ).sort_values("importance")

        FEATURE_LABELS = {
            "cnt_m0":         "Кол-во транзакций (мес. 0)",
            "activated_m0":   "Активирован в мес. 0",
            "creation_day":   "День создания карты",
            "amt_m0":         "Сумма транзакций (мес. 0)",
            "online_share":   "Доля онлайн-платежей",
            "n_cats_m0":      "Кол-во категорий",
            "offline_share":  "Доля офлайн-платежей",
            "transfer_share": "Доля переводов",
            "cash_share":     "Доля снятия наличных",
        }
        fi["label"] = fi["feature"].map(FEATURE_LABELS).fillna(fi["feature"])

        fig = go.Figure(go.Bar(
            x=fi["importance"], y=fi["label"],
            orientation="h",
            marker_color=["#7f4dff" if v > 0.01 else "#dee0e5" for v in fi["importance"]],
            text=fi["importance"].apply(lambda v: f"{v:.3f}"),
            textposition="outside",
        ))
        fig.update_layout(
            height=350, margin=dict(l=0, r=60, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, fi["importance"].max() * 1.25]),
        )
        st.plotly_chart(fig, width="stretch")

    # ── Threshold обоснование ────────────────────────────────────────────────
    with col2:
        st.subheader("Обоснование порога классификации")
        thr = metrics["threshold"]
        p, r, f = metrics["precision"], metrics["recall"], metrics["f1"]

        st.markdown(f"""
**Выбранный порог: `{thr:.3f}`**

При этом пороге:
- **Precision** {p:.3f} — из всех предсказанных оттоков {p*100:.1f}% реальные
- **Recall** {r:.3f} — модель ловит {r*100:.1f}% всех реальных оттоков
- **F1** {f:.3f} — баланс точности и полноты

**Почему именно F1?**

В задаче предотвращения оттока:
- FN (пропустить уходящего) = потеря клиента → дорого
- FP (лишний бонус активному) = малые затраты → дёшево

Поэтому **Recall важнее Precision**, но F1 выбран как базовый оптимум.
При желании снизить порог до `0.30` — Recall вырастет до **98.5%**
при Precision **68.2%** (больше бонусов, меньше пропущенных).
        """)

        # Визуализация precision/recall tradeoff
        thresholds_demo = [0.274, 0.328, 0.475, 0.498, 0.549, 0.658, 0.845]
        precisions_demo  = [0.682, 0.691, 0.724, 0.746, 0.764, 0.772, 0.905]
        recalls_demo     = [0.985, 0.977, 0.937, 0.902, 0.858, 0.809, 0.115]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds_demo, y=precisions_demo,
                                 name="Precision", line=dict(color="#7f4dff", width=2)))
        fig.add_trace(go.Scatter(x=thresholds_demo, y=recalls_demo,
                                 name="Recall", line=dict(color="#cc0000", width=2)))
        fig.add_vline(x=thr, line_dash="dash", line_color="#00ad3a",
                      annotation_text="Выбранный порог", annotation_position="top left")
        fig.update_layout(
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.15),
            xaxis_title="Порог", yaxis_title="Метрика",
        )
        st.plotly_chart(fig, width="stretch")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.subheader("Матрица ошибок (test set)")
    total_test = 3494
    tp = int(round(metrics["recall"]    * total_test * metrics["churn_rate"]))
    fp = int(round((1 - metrics["precision"]) / metrics["precision"] * tp))
    fn = int(round(total_test * metrics["churn_rate"] - tp))
    tn = total_test - tp - fp - fn

    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Факт: остался", "Факт: ушёл"],
        columns=["Предсказан: остался", "Предсказан: ушёл"],
    )
    fig = px.imshow(
        cm_df, text_auto=True, color_continuous_scale=["#f2f4f7", "#7f4dff", "#7000ff"],
        labels=dict(color="Карт"),
    )
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")


# ===========================================================================
# PAGE 3 — SCORING
# ===========================================================================

elif page == "Скоринг карт":
    st.title("Скоринг карт по риску оттока")

    if scored is None:
        st.warning("Запустите `python3 model.py` для генерации scored_cards.csv")
        st.stop()

    # ── Filters ──────────────────────────────────────────────────────────────
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
        proba_range = st.slider("P(churn) диапазон", 0.0, 1.0, (0.0, 1.0), 0.01)

    mask = (
        scored["risk_level"].isin(risk_filter) &
        scored["churn_proba"].between(*proba_range)
    )
    if region_filter:
        mask &= scored["kiosk_name"].isin(region_filter)

    filtered = scored[mask].copy()
    st.caption(f"Показано: **{len(filtered):,}** из {len(scored):,} карт")

    # ── KPIs for filtered ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Карт", f"{len(filtered):,}")
    c2.metric("Средний P(churn)", f"{filtered['churn_proba'].mean():.3f}")
    c3.metric("Не активированы (мес. 0)", f"{int((filtered['activated_m0']==0).sum()):,}")
    c4.metric("Ср. транзакций (мес. 0)", f"{filtered['cnt_m0'].mean():.1f}")

    # ── Scatter: P(churn) vs cnt_m0 ─────────────────────────────────────────
    st.subheader("P(churn) vs транзакции в 1-й месяц")
    fig = px.scatter(
        filtered.sample(min(3000, len(filtered)), random_state=42),
        x="cnt_m0", y="churn_proba",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        opacity=0.5,
        labels={"cnt_m0": "Транзакций (мес. 0)", "churn_proba": "P(churn)"},
        hover_data=["card_id", "kiosk_name", "n_cats_m0"],
    )
    fig.add_hline(y=metrics["threshold"], line_dash="dash", line_color="#7000ff",
                  annotation_text="Порог", annotation_position="right")
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    # ── Table ────────────────────────────────────────────────────────────────
    st.subheader("Таблица карт")

    def _style_risk(val):
        colors_map = {"CRITICAL": "#fff5f5", "HIGH": "#fff8f0",
                      "MEDIUM": "#fffdf0", "LOW": "#f5fbf6"}
        return f"background-color: {colors_map.get(val, '')}"

    display_df = (
        filtered[["card_id", "kiosk_name", "churn_proba", "risk_level",
                   "cnt_m0", "n_cats_m0", "activated_m0"]]
        .sort_values("churn_proba", ascending=False)
        .head(500)
        .rename(columns={
            "card_id":      "Карта",
            "kiosk_name":   "Регион",
            "churn_proba":  "P(churn)",
            "risk_level":   "Риск",
            "cnt_m0":       "Транзакций м.0",
            "n_cats_m0":    "Категорий м.0",
            "activated_m0": "Активирован м.0",
        })
    )
    st.dataframe(
        display_df.style
            .map(_style_risk, subset=["Риск"])
            .format({"P(churn)": "{:.3f}"}),
        use_container_width=True, height=420,
    )

    # Download
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

        # Show heuristic preview from scored cards
        if scored is not None:
            st.subheader("Предпросмотр: карты HIGH/CRITICAL риска")
            preview = scored[scored["risk_level"].isin(["CRITICAL", "HIGH"])].head(10)
            st.dataframe(preview, width="stretch")
        st.stop()

    # ── KPIs ─────────────────────────────────────────────────────────────────
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Кандидатов на бонус", f"{len(bonus):,}")
    b2.metric("CRITICAL", f"{(bonus['risk_level']=='CRITICAL').sum():,}")
    b3.metric("HIGH",     f"{(bonus['risk_level']=='HIGH').sum():,}")
    b4.metric("Ср. P(churn)", f"{bonus['churn_proba'].mean():.3f}")

    st.divider()
    col1, col2 = st.columns(2)

    # ── Топ категории ─────────────────────────────────────────────────────────
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
        st.plotly_chart(fig, width="stretch")

    # ── Каналы отправки ───────────────────────────────────────────────────────
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
        st.plotly_chart(fig, width="stretch")

    # ── Sample messages ───────────────────────────────────────────────────────
    st.subheader("Примеры персонализированных сообщений")
    top5 = bonus.sort_values("churn_proba", ascending=False).head(5)
    for _, row in top5.iterrows():
        risk_color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(row["risk_level"], "⚪")
        with st.expander(f"{risk_color} {row['card_id']} — P(churn)={row['churn_proba']:.2f} | {row['top_category']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Регион:** {row.get('kiosk_name', '—')}")
                st.markdown(f"**Риск:** {row['risk_level']}")
                st.markdown(f"**P(churn):** {row['churn_proba']:.3f}")
                st.markdown(f"**Категория:** {row['top_category']}")
            with c2:
                st.info(row["message"])
                st.markdown(f"**Бонус:** {row['bonus_sums']} сум")
                st.markdown(f"**Отправить:** день {row['send_day']} · {row['channel']}")

    # ── Full table ────────────────────────────────────────────────────────────
    st.subheader("Полный список кандидатов")
    st.dataframe(
        bonus.sort_values("churn_proba", ascending=False)
             [["card_id", "kiosk_name", "churn_proba", "risk_level",
               "top_category", "bonus_sums", "send_day", "channel", "message"]],
        use_container_width=True, height=350,
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
            n_cards=("card_id", "count"),
            avg_churn_proba=("churn_proba", "mean"),
            pct_critical=("risk_level", lambda x: (x == "CRITICAL").mean() * 100),
            pct_high=("risk_level", lambda x: (x == "HIGH").mean() * 100),
            pct_at_risk=("predicted_churn", "mean"),
            avg_cnt_m0=("cnt_m0", "mean"),
        )
        .reset_index()
        .sort_values("avg_churn_proba", ascending=False)
    )
    region_stats["pct_at_risk"] *= 100

    # ── Top regions bar ───────────────────────────────────────────────────────
    st.subheader("Регионы по среднему P(churn)")
    fig = px.bar(
        region_stats.sort_values("avg_churn_proba"),
        x="avg_churn_proba", y="kiosk_name", orientation="h",
        color="avg_churn_proba",
        color_continuous_scale=["#00ad3a", "#7f4dff", "#cc0000"],
        labels={"avg_churn_proba": "Ср. P(churn)", "kiosk_name": "Регион"},
        text=region_stats.sort_values("avg_churn_proba")["avg_churn_proba"].apply(lambda v: f"{v:.3f}"),
    )
    fig.update_layout(
        height=max(300, len(region_stats) * 28),
        margin=dict(l=0, r=60, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width="stretch")

    # ── Bubble: n_cards vs avg_churn_proba ───────────────────────────────────
    st.subheader("Карты vs риск оттока по регионам")
    fig = px.scatter(
        region_stats,
        x="avg_cnt_m0", y="avg_churn_proba",
        size="n_cards", color="pct_at_risk",
        color_continuous_scale=["#00ad3a", "#7f4dff", "#cc0000"],
        hover_name="kiosk_name",
        labels={
            "avg_cnt_m0":      "Ср. транзакций (мес. 0)",
            "avg_churn_proba": "Ср. P(churn)",
            "pct_at_risk":     "% под риском",
            "n_cards":         "Карт",
        },
        size_max=50,
    )
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(tickfont=dict(color="#4d4f59")),
    )
    st.plotly_chart(fig, width="stretch")

    # ── Table ────────────────────────────────────────────────────────────────
    st.subheader("Детальная таблица по регионам")
    _region_renamed = region_stats.rename(columns={
        "kiosk_name":       "Регион",
        "n_cards":          "Карт",
        "avg_churn_proba":  "Ср. P(churn)",
        "pct_critical":     "% CRITICAL",
        "pct_high":         "% HIGH",
        "pct_at_risk":      "% под риском",
        "avg_cnt_m0":       "Ср. транзакций м.0",
    })
    _region_styled = _region_renamed.style.format({
        "Ср. P(churn)": "{:.3f}",
        "% CRITICAL":   "{:.1f}%",
        "% HIGH":       "{:.1f}%",
        "% под риском": "{:.1f}%",
        "Ср. транзакций м.0": "{:.2f}",
    }).background_gradient(subset=["Ср. P(churn)"], cmap="Purples")
    st.dataframe(_region_styled, width="stretch", height=420)
