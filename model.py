#!/usr/bin/env python3
"""
Decision Tree model predicting card dormancy (churn).

Dataset has monthly granularity. month_of_life=0 (first calendar month
of card life) is used as the proxy for 'first 7-14 days' behavior.

Features  : activation speed, transaction volume, category diversity,
            payment channel mix — all from month_of_life=0.
Target    : churn = 1 - is_active  (static label per card).
Output    : churn probability per card + results/scored_cards.csv
            for downstream bonus_logic.py.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
)
import json
from pathlib import Path
import argparse
from data_loader import load_data

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Category constants (actual names from dataset)
# ---------------------------------------------------------------------------

CAT_ONLINE        = 'Онлайн оплаты'
CAT_OFFLINE       = 'Оффлайн оплаты'
CAT_TRANSFER_IN   = 'Перевод внутри страны - пополнение'
CAT_TRANSFER_OUT  = 'Перевод внутри страны - списание'
CAT_ATM_CAPITAL   = 'Пополнение через АТМ - Kапитал Банк'
CAT_ATM_CASH_CAP  = 'Снятие наличных в АТМ - Kапитал Банк'
CAT_ATM_CASH_OTH  = 'Снятие наличных в АТМ - Другие банки'
CAT_NASIYA        = 'Кэш кредит Nasiya - пополнение'
CAT_RETURNS       = 'Возвраты/отмены - Оплаты'
CAT_CROSS_BORDER  = 'Трансгран - пополнение'
CAT_OTHER         = 'Остальное'

ALL_CATEGORIES = [
    CAT_ONLINE, CAT_OFFLINE, CAT_TRANSFER_IN, CAT_TRANSFER_OUT,
    CAT_ATM_CAPITAL, CAT_ATM_CASH_CAP, CAT_ATM_CASH_OTH,
    CAT_NASIYA, CAT_RETURNS, CAT_CROSS_BORDER, CAT_OTHER,
]

# Features used for training
FEATURE_COLS = [
    'creation_day',       # day of month card was created (proxy: days available in month 0)
    'activated_m0',       # 1 if any transaction in month 0
    'cnt_m0',             # total transaction count in month 0
    'amt_m0',             # total amount in month 0
    'n_cats_m0',          # distinct categories used in month 0
    'online_share',       # share of cnt: online payments
    'offline_share',      # share of cnt: offline/POS payments
    'transfer_share',     # share of cnt: transfers (in + out)
    'cash_share',         # share of cnt: cash withdrawals
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def extract_features(df):
    """
    Build one row per card from month_of_life=0 data.

    month_of_life=0 is the first calendar month after card creation —
    the closest monthly proxy to 'first 7-14 days' available in the data.
    Cards created late in the month have fewer effective days in month 0
    (captured via creation_day feature).

    Target: churn = 1 - is_active.
    """
    m0 = df[df['month_of_life'] == 0].copy()

    # Pivot category transaction counts for month 0
    cat_pivot = m0.pivot_table(
        index='card_id', columns='category', values='cnt',
        aggfunc='sum', fill_value=0,
    )
    for cat in ALL_CATEGORIES:          # ensure all columns exist
        if cat not in cat_pivot.columns:
            cat_pivot[cat] = 0

    # Card-level totals in month 0
    totals_m0 = m0.groupby('card_id').agg(
        cnt_m0=('cnt', 'sum'),
        amt_m0=('amt', 'sum'),
        n_cats_m0=('cnt', lambda x: (x > 0).sum()),
    )

    # Card metadata (from full df — every card has month 0 rows)
    meta = df.groupby('card_id').agg(
        creation_day=('card_creation_date', lambda x: x.iloc[0].day),
        kiosk_name=('kiosk_name', 'first'),
        is_active=('is_active', 'max'),
    )

    features = (
        meta
        .join(totals_m0, how='left')
        .join(cat_pivot,  how='left')
        .fillna(0)
    )

    # Channel-share features (safe division: clip denominator at 1)
    safe_cnt = features['cnt_m0'].clip(lower=1)

    features['activated_m0']   = (features['cnt_m0'] > 0).astype(int)
    features['online_share']   = features[CAT_ONLINE] / safe_cnt
    features['offline_share']  = features[CAT_OFFLINE] / safe_cnt
    features['transfer_share'] = (
        features[CAT_TRANSFER_IN] + features[CAT_TRANSFER_OUT]
    ) / safe_cnt
    features['cash_share'] = (
        features[CAT_ATM_CASH_CAP] + features[CAT_ATM_CASH_OTH]
    ) / safe_cnt

    features['churn'] = 1 - features['is_active']

    return features.reset_index()


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def select_threshold(y_test, y_proba):
    """
    Pick classification threshold via Precision-Recall curve analysis.

    Rationale
    ---------
    In churn prevention the cost of a False Negative (missed churner who
    receives no bonus and leaves) exceeds the cost of a False Positive
    (unnecessary bonus sent). We therefore optimize for F1, which gives
    equal weight to Precision and Recall, and document the tradeoff so
    the business can shift the threshold toward higher Recall if the
    per-bonus cost is low.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    denom = precisions[:-1] + recalls[:-1]
    f1_scores = np.where(denom > 0,
                         2 * precisions[:-1] * recalls[:-1] / denom, 0)
    best_idx = int(np.argmax(f1_scores))
    best_t   = float(thresholds[best_idx])

    print(f"\nThreshold selection — Precision / Recall tradeoff:")
    print(f"   {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"   {'-'*45}")
    for pct in [10, 25, 40, 50, 60, 75, 90]:
        t   = float(np.percentile(thresholds, pct))
        idx = min(int(np.searchsorted(thresholds, t)), len(precisions) - 2)
        p, r = precisions[idx], recalls[idx]
        f    = 2 * p * r / (p + r) if p + r > 0 else 0
        mark = '  <-- selected' if abs(t - best_t) < 0.02 else ''
        print(f"   {t:>10.3f}  {p:>10.3f}  {r:>8.3f}  {f:>8.3f}{mark}")

    print(f"\n   Optimal threshold : {best_t:.3f}")
    print(f"   At this point     : Precision={precisions[best_idx]:.3f}  "
          f"Recall={recalls[best_idx]:.3f}  F1={f1_scores[best_idx]:.3f}")
    print(f"   Note: lower threshold → higher Recall (catches more churners")
    print(f"         at cost of more false alarms; acceptable when bonus is cheap).")

    return best_t


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(features_df):
    """Train Decision Tree, print metrics, return model + test artifacts."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    X = features_df[FEATURE_COLS]
    y = features_df['churn']

    print(f"\nTotal cards : {len(features_df):,}")
    print(f"Churn rate  : {y.mean():.1%}  (churn=1 means is_active=0)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',   # corrects for equal 50/50 split, robust to imbalance
        random_state=42,
    )
    dt.fit(X_train, y_train)

    y_proba_train = dt.predict_proba(X_train)[:, 1]
    y_proba_test  = dt.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, y_proba_train)
    auc_test  = roc_auc_score(y_test,  y_proba_test)
    print(f"\nAUC-ROC : train={auc_train:.3f}  test={auc_test:.3f}")

    threshold = select_threshold(y_test, y_proba_test)

    y_pred = (y_proba_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = f1_score(y_test, y_pred)

    print(f"\nTest metrics at threshold={threshold:.3f}:")
    print(f"   Precision : {precision:.3f}")
    print(f"   Recall    : {recall:.3f}")
    print(f"   F1-score  : {f1:.3f}")
    print(f"   AUC-ROC   : {auc_test:.3f}")

    print(f"\nConfusion matrix (test):")
    print(f"   TN={tn:,}  FP={fp:,}")
    print(f"   FN={fn:,}  TP={tp:,}")

    importance_df = pd.DataFrame({
        'feature':    FEATURE_COLS,
        'importance': dt.feature_importances_,
    }).sort_values('importance', ascending=False)

    print(f"\nFeature importance:")
    for _, row in importance_df.iterrows():
        bar = '#' * int(row['importance'] * 40)
        print(f"   {row['feature']:<22}  {row['importance']:.3f}  {bar}")

    return dt, X_test, y_test, y_proba_test, threshold, importance_df


# ---------------------------------------------------------------------------
# Scoring & output
# ---------------------------------------------------------------------------

def score_all_cards(dt, features_df, threshold):
    """Add churn_proba, predicted_churn, risk_level to every card."""
    scored = features_df.copy()
    scored['churn_proba']     = dt.predict_proba(scored[FEATURE_COLS])[:, 1]
    scored['predicted_churn'] = (scored['churn_proba'] >= threshold).astype(int)

    def _risk(p):
        if p >= 0.70: return 'CRITICAL'
        if p >= 0.50: return 'HIGH'
        if p >= 0.30: return 'MEDIUM'
        return 'LOW'

    scored['risk_level'] = scored['churn_proba'].apply(_risk)
    return scored


def save_results(scored_df, X_test, y_test, y_proba_test, threshold, importance_df):
    """Save metrics JSON and scored_cards CSV for bonus_logic.py."""
    Path('results').mkdir(exist_ok=True)

    y_pred = (y_proba_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        'auc_roc':               round(float(roc_auc_score(y_test, y_proba_test)), 4),
        'threshold':             round(float(threshold), 4),
        'precision':             round(float(precision), 4),
        'recall':                round(float(recall), 4),
        'f1':                    round(float(f1_score(y_test, y_pred)), 4),
        'total_cards':           int(len(scored_df)),
        'churn_rate':            round(float(scored_df['churn'].mean()), 4),
        'predicted_churn_cards': int(scored_df['predicted_churn'].sum()),
        'feature_importance': {
            row['feature']: round(float(row['importance']), 4)
            for _, row in importance_df.iterrows()
        },
    }

    with open('results/model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved  → results/model_metrics.json")

    out_cols = [
        'card_id', 'kiosk_name', 'churn_proba', 'predicted_churn',
        'risk_level', 'cnt_m0', 'n_cats_m0', 'activated_m0',
    ]
    scored_df[out_cols].to_csv('results/scored_cards.csv', index=False)
    print(f"Scored cards   → results/scored_cards.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Uzum Bank churn prediction model')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to CSV data file')
    args = parser.parse_args()

    print("\nUZUM BANK: Churn Prediction Model")
    print("=" * 70)

    df = load_data(args.data_path, fallback_path='data/uzum_hackathon_dataset.csv')
    if df is None:
        return

    print(f"Rows    : {len(df):,}")
    print(f"Cards   : {df['card_id'].nunique():,}")
    print(f"Months  : {df['month_of_life'].max() + 1} (month_of_life 0–{df['month_of_life'].max()})")
    print(f"is_active=1: {df.groupby('card_id')['is_active'].first().mean():.1%}")

    print("\nExtracting features from month_of_life=0...")
    features_df = extract_features(df)
    print(f"Cards with features : {len(features_df):,}")
    print(f"Activated in month 0: {features_df['activated_m0'].mean():.1%}")

    dt, X_test, y_test, y_proba_test, threshold, importance_df = train_model(features_df)

    scored_df = score_all_cards(dt, features_df, threshold)
    save_results(scored_df, X_test, y_test, y_proba_test, threshold, importance_df)

    print(f"\nDone.")


if __name__ == '__main__':
    main()
