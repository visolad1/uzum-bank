#!/usr/bin/env python3
"""
Decision Tree model predicting card dormancy (churn).

Task spec (4.3):
    - Input window  : first 7–14 days of card life
    - Target        : card goes dormant in the NEXT 30 days
                      (no transactions in days 15–44 after activation)
    - Features      : activation date properties, transaction count in
                      first week, MCC diversity, channel of first transaction
    - Output        : churn probability per card
                      + trigger recommendation (type + send day)
                      + results/scored_cards.csv for downstream bonus_logic.py

Metrics: Precision / Recall / AUC-ROC with threshold justification.
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

# Channel groupings for first-transaction channel feature
CHANNEL_ONLINE   = [CAT_ONLINE]
CHANNEL_OFFLINE  = [CAT_OFFLINE]
CHANNEL_TRANSFER = [CAT_TRANSFER_IN, CAT_TRANSFER_OUT]
CHANNEL_CASH     = [CAT_ATM_CASH_CAP, CAT_ATM_CASH_OTH]
CHANNEL_OTHER    = [CAT_ATM_CAPITAL, CAT_NASIYA, CAT_RETURNS, CAT_CROSS_BORDER, CAT_OTHER]

# Features used for training — strictly aligned with spec 4.3:
#   «дата активации, количество транзакций в первую неделю,
#    разнообразие MCC, канал первой транзакции»
FEATURE_COLS = [
    # 1. Activation date properties (дата активации)
    'creation_dow',           # day of week (0=Monday … 6=Sunday)
    'creation_is_weekend',    # 1 if card created on Sat/Sun
    'days_to_first_txn',      # days from card creation to first transaction
    'activated_week1',        # 1 if any transaction in first 7 days

    # 2. Transaction count in first week (количество транзакций в первую неделю)
    'cnt_week1',              # transaction count in first 7 days
    'cnt_early',              # transaction count in first 14 days (extended)

    # 3. MCC diversity (разнообразие MCC)
    'n_cats_week1',           # distinct categories used in first 7 days
    'n_cats_early',           # distinct categories used in first 14 days

    # 4. Channel of first transaction (канал первой транзакции, one-hot)
    'first_txn_online',
    'first_txn_offline',
    'first_txn_transfer',
    'first_txn_cash',
    'first_txn_other',
]

# NOTE: `amt_early` (transaction volume) is intentionally excluded — the spec
# specifies COUNT, not amount. Including it dominated the tree at the expense
# of the spec-mandated features. `creation_day` (day-of-month) is excluded as
# a likely calendar artifact rather than a behavioural signal.


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _get_channel(category):
    """Map a category name to a high-level channel string."""
    if category in CHANNEL_ONLINE:   return 'online'
    if category in CHANNEL_OFFLINE:  return 'offline'
    if category in CHANNEL_TRANSFER: return 'transfer'
    if category in CHANNEL_CASH:     return 'cash'
    return 'other'


def extract_features(df):
    """
    Build one row per card from the raw transaction log.

    Observation window : days 0–14 from card creation  (features)
    Target window      : days 15–44 from card creation  (label)

    Target definition
    -----------------
    dormant_30d = 1  →  card had NO transactions in the 15–44 day window
    dormant_30d = 0  →  card had AT LEAST ONE transaction in that window

    Cards that have no data beyond day 14 are excluded from training
    (we cannot label them) but are still scored for production use.
    """
    df = df.copy()

    df['card_creation_date'] = pd.to_datetime(df['card_creation_date'])
    df['month']              = pd.to_datetime(df['month'])
    df['days_from_creation'] = (df['month'] - df['card_creation_date']).dt.days

    # ── 1. OBSERVATION WINDOW: days 0–14 ──────────────────────────────────
    early = df[(df['days_from_creation'] >= 0) & (df['days_from_creation'] <= 14)].copy()

    # Category pivot (cnt per category in early window)
    cat_pivot = early.pivot_table(
        index='card_id', columns='category', values='cnt',
        aggfunc='sum', fill_value=0
    )
    for cat in ALL_CATEGORIES:
        if cat not in cat_pivot.columns:
            cat_pivot[cat] = 0

    # Week-1 subset (days 0–6) — explicit spec window
    week1 = early[early['days_from_creation'] <= 6]
    week1_active = week1[week1['cnt'] > 0]
    week1_agg = week1.groupby('card_id').agg(
        cnt_week1    = ('cnt', 'sum'),
    )
    week1_cats = (
        week1_active.groupby('card_id')['category'].nunique().rename('n_cats_week1')
    )

    # Base aggregations over 0–14 day extended observation window
    early_active = early[early['cnt'] > 0]
    base = early.groupby('card_id').agg(
        cnt_early    = ('cnt', 'sum'),
        amt_early    = ('amt', 'sum'),                      # kept for reporting only
        creation_day = ('card_creation_date', lambda x: x.iloc[0].day),
        creation_dow = ('card_creation_date', lambda x: x.iloc[0].dayofweek),
        kiosk_name   = ('kiosk_name', 'first'),
        is_active    = ('is_active', 'max'),
    )
    early_cats = (
        early_active.groupby('card_id')['category'].nunique().rename('n_cats_early')
    )

    # First transaction channel
    # Find the earliest row (min days_from_creation) with cnt > 0
    txn_rows = early[early['cnt'] > 0].copy()
    txn_rows = txn_rows.sort_values('days_from_creation')
    first_txn = (
        txn_rows.groupby('card_id')
        .agg(
            first_txn_category  = ('category', 'first'),
            days_to_first_txn   = ('days_from_creation', 'min'),
        )
    )

    # ── 2. TARGET WINDOW: days 15–44 ──────────────────────────────────────
    target_window = df[(df['days_from_creation'] >= 15) & (df['days_from_creation'] <= 44)]
    has_txn_in_target = (
        target_window[target_window['cnt'] > 0]
        .groupby('card_id')['cnt']
        .sum()
        .rename('cnt_target')
        .gt(0)
        .astype(int)
        .rename('active_in_target')
    )

    # ── 3. ASSEMBLE ───────────────────────────────────────────────────────
    features = (
        base
        .join(cat_pivot, how='left')
        .join(week1_agg, how='left')
        .join(week1_cats, how='left')
        .join(early_cats, how='left')
        .join(first_txn, how='left')
        .join(has_txn_in_target, how='left')
        .fillna(0)
    )

    # ── 4. DERIVED FEATURES ───────────────────────────────────────────────
    features['creation_is_weekend'] = (features['creation_dow'] >= 5).astype(int)
    features['activated_early']     = (features['cnt_early'] > 0).astype(int)
    features['activated_week1']     = (features['cnt_week1'] > 0).astype(int)

    # First transaction channel: one-hot encoding
    first_channel = features['first_txn_category'].apply(
        lambda c: _get_channel(c) if isinstance(c, str) else 'none'
    )
    for ch in ['online', 'offline', 'transfer', 'cash', 'other']:
        features[f'first_txn_{ch}'] = (first_channel == ch).astype(int)

    # ── 5. TARGET ─────────────────────────────────────────────────────────
    # dormant_30d = 1 if card did NOT transact in the 15–44 day window
    # Cards with no data in target window are treated as dormant (conservative)
    features['dormant_30d'] = (features['active_in_target'] == 0).astype(int)

    # Flag cards that have no target-window data at all (cannot be labeled reliably)
    cards_with_target_data = set(target_window['card_id'].unique())
    features['has_target_data'] = features.index.isin(cards_with_target_data).astype(int)

    return features.reset_index()


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def select_threshold(y_test, y_proba):
    """
    Pick classification threshold via Precision-Recall curve analysis.

    Business rationale
    ------------------
    False Negative (missed dormant card → no trigger sent → customer leaves):
        Cost = lost long-term revenue from an inactive card.
    False Positive (active card flagged → unnecessary trigger sent):
        Cost = small bonus/notification spend.

    Since FN cost > FP cost, we accept slightly lower Precision to gain
    higher Recall.  We optimize F1 as a balanced baseline and present the
    full tradeoff table so the business can shift the threshold lower if the
    per-trigger cost is low relative to lifetime card value.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    denom    = precisions[:-1] + recalls[:-1]
    f1_scores = np.where(denom > 0,
                         2 * precisions[:-1] * recalls[:-1] / denom, 0)
    best_idx  = int(np.argmax(f1_scores))
    best_t    = float(thresholds[best_idx])

    print(f"\nThreshold selection — Precision / Recall tradeoff:")
    print(f"   {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"   {'-'*47}")
    for pct in [10, 25, 40, 50, 60, 75, 90]:
        t   = float(np.percentile(thresholds, pct))
        idx = min(int(np.searchsorted(thresholds, t)), len(precisions) - 2)
        p, r = precisions[idx], recalls[idx]
        f    = 2 * p * r / (p + r) if p + r > 0 else 0
        mark = '  <-- selected (best F1)' if abs(t - best_t) < 0.02 else ''
        print(f"   {t:>10.3f}  {p:>10.3f}  {r:>8.3f}  {f:>8.3f}{mark}")

    print(f"\n   Optimal threshold : {best_t:.3f}")
    print(f"   At this point     : Precision={precisions[best_idx]:.3f}  "
          f"Recall={recalls[best_idx]:.3f}  F1={f1_scores[best_idx]:.3f}")
    print(f"\n   Interpretation:")
    print(f"   → Lower threshold  = higher Recall (catch more dormant cards)")
    print(f"     acceptable when trigger cost is low vs card lifetime value.")
    print(f"   → Higher threshold = higher Precision (fewer wasted triggers)")
    print(f"     use when bonus budget is tight.")

    return best_t


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(features_df):
    """Train Decision Tree, print metrics, return model + test artifacts."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    # Only train on cards where we have target-window data
    labeled = features_df[features_df['has_target_data'] == 1].copy()
    unlabeled = features_df[features_df['has_target_data'] == 0].copy()

    X = labeled[FEATURE_COLS]
    y = labeled['dormant_30d']

    print(f"\nTotal cards          : {len(features_df):,}")
    print(f"Cards with target data (labeled): {len(labeled):,}")
    print(f"Cards without target data (score-only): {len(unlabeled):,}")
    print(f"Dormancy rate (days 15–44): {y.mean():.1%}  (1 = no transactions in window)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    dt = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',   # inversely weighted by class frequency
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
    print(f"   TN = correctly identified as NOT dormant")
    print(f"   FP = active card flagged (unnecessary trigger cost)")
    print(f"   FN = dormant card missed (customer leaves without intervention)")
    print(f"   TP = dormant card caught (trigger sent, potential save)")

    importance_df = pd.DataFrame({
        'feature':    FEATURE_COLS,
        'importance': dt.feature_importances_,
    }).sort_values('importance', ascending=False)

    print(f"\nFeature importance:")
    for _, row in importance_df.iterrows():
        bar = '#' * int(row['importance'] * 40)
        print(f"   {row['feature']:<25}  {row['importance']:.3f}  {bar}")

    return dt, X_test, y_test, y_proba_test, threshold, importance_df


# ---------------------------------------------------------------------------
# Trigger recommendation logic
# ---------------------------------------------------------------------------

def assign_trigger(row):
    """
    Rule-based trigger recommendation on top of churn score.

    Logic
    -----
    Trigger type is based on the customer's first transaction channel and
    category diversity — we meet them where they already are.
    Send day is calibrated to reach the customer before the 30-day dormancy
    window closes (days 15–44), so we send between day 10 and day 18.

    Returns (trigger_type, trigger_message_ru, send_day)
    """
    p = row['dormant_30d_proba']

    if p < 0.30:
        return 'none', '—', None

    # Determine primary channel preference
    if row['first_txn_online'] == 1:
        channel = 'online'
    elif row['first_txn_offline'] == 1:
        channel = 'offline'
    elif row['first_txn_transfer'] == 1:
        channel = 'transfer'
    elif row['first_txn_cash'] == 1:
        channel = 'cash'
    else:
        channel = 'other'

    n_cats   = int(row['n_cats_early'])
    cnt      = int(row['cnt_early'])

    # ── HIGH RISK (p ≥ 0.70) ─────────────────────────────────────────────
    if p >= 0.70:
        if channel == 'online':
            return (
                'push_cashback',
                'Получите 5% кэшбэк на онлайн-покупки — только 3 дня!',
                10
            )
        if channel == 'offline':
            return (
                'push_cashback',
                'Двойной кэшбэк в магазинах партнёров — используйте карту сегодня!',
                10
            )
        if channel == 'cash':
            return (
                'push_bonus',
                'Совершите безналичную оплату и получите бонус 500 сум.',
                10
            )
        # transfer / other
        return (
            'push_generic',
            'Специальное предложение для вас — откройте приложение!',
            10
        )

    # ── MEDIUM-HIGH RISK (0.50 ≤ p < 0.70) ──────────────────────────────
    if p >= 0.50:
        if n_cats <= 1:
            return (
                'email_discovery',
                'Откройте новые возможности карты: оплата, переводы и кэшбэк.',
                12
            )
        if channel == 'online':
            return (
                'push_reminder',
                'Не забудьте про кэшбэк за онлайн-покупки в этом месяце!',
                12
            )
        return (
            'push_reminder',
            'Ваша карта ждёт — совершите любую операцию и получите бонус.',
            12
        )

    # ── MEDIUM RISK (0.30 ≤ p < 0.50) ───────────────────────────────────
    if cnt == 0:
        # Never transacted in early window — activation push
        return (
            'sms_activation',
            'Активируйте карту: совершите первую покупку и получите подарок.',
            14
        )
    return (
        'email_nurture',
        'Советы по использованию карты и персональные предложения.',
        18
    )


# ---------------------------------------------------------------------------
# Scoring & output
# ---------------------------------------------------------------------------

def score_all_cards(dt, features_df, threshold):
    """Add dormancy probability, prediction, risk level, and trigger to every card."""
    scored = features_df.copy()
    scored['dormant_30d_proba']  = dt.predict_proba(scored[FEATURE_COLS])[:, 1]
    scored['predicted_dormant']  = (scored['dormant_30d_proba'] >= threshold).astype(int)

    def _risk(p):
        if p >= 0.70: return 'CRITICAL'
        if p >= 0.50: return 'HIGH'
        if p >= 0.30: return 'MEDIUM'
        return 'LOW'

    scored['risk_level'] = scored['dormant_30d_proba'].apply(_risk)

    trigger_cols = scored.apply(assign_trigger, axis=1, result_type='expand')
    trigger_cols.columns = ['trigger_type', 'trigger_message', 'trigger_send_day']
    scored = pd.concat([scored, trigger_cols], axis=1)

    return scored


def _build_pr_curve(y_test, y_proba_test, n_points=20):
    """Sample (threshold, precision, recall) points from the real PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_test)
    if len(thresholds) == 0:
        return []
    idxs = np.linspace(0, len(thresholds) - 1, num=min(n_points, len(thresholds))).astype(int)
    return [
        {
            'threshold': round(float(thresholds[i]), 4),
            'precision': round(float(precisions[i]), 4),
            'recall':    round(float(recalls[i]), 4),
        }
        for i in idxs
    ]


def save_results(scored_df, X_test, y_test, y_proba_test, threshold, importance_df):
    """Save metrics JSON and scored_cards CSV."""
    Path('results').mkdir(exist_ok=True)

    y_pred = (y_proba_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        'model'                  : 'DecisionTree — dormancy in days 15–44',
        'observation_window_days': '0–14',
        'target_window_days'     : '15–44',
        'auc_roc'                : round(float(roc_auc_score(y_test, y_proba_test)), 4),
        'threshold'              : round(float(threshold), 4),
        'precision'              : round(float(precision), 4),
        'recall'                 : round(float(recall), 4),
        'f1'                     : round(float(f1_score(y_test, y_pred)), 4),
        'confusion_matrix'       : {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'pr_curve'               : _build_pr_curve(y_test, y_proba_test),
        'total_cards'            : int(len(scored_df)),
        'labeled_cards'          : int(scored_df['has_target_data'].sum()),
        'dormancy_rate'          : round(float(scored_df.loc[scored_df['has_target_data']==1, 'dormant_30d'].mean()), 4),
        'predicted_dormant_cards': int(scored_df['predicted_dormant'].sum()),
        'trigger_distribution'   : scored_df['trigger_type'].value_counts().to_dict(),
        'feature_importance'     : {
            row['feature']: round(float(row['importance']), 4)
            for _, row in importance_df.iterrows()
        },
    }

    with open('results/model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved  → results/model_metrics.json")

    out_cols = [
        'card_id', 'kiosk_name',
        'dormant_30d_proba', 'predicted_dormant', 'risk_level',
        'trigger_type', 'trigger_message', 'trigger_send_day',
        # raw features for downstream use
        'cnt_early', 'cnt_week1', 'n_cats_early', 'activated_early',
        'first_txn_online', 'first_txn_offline', 'first_txn_transfer',
        'first_txn_cash', 'first_txn_other',
        'days_to_first_txn',
        'creation_day', 'creation_dow', 'creation_is_weekend',
    ]
    # keep only columns that exist (safety)
    out_cols = [c for c in out_cols if c in scored_df.columns]
    scored_df[out_cols].to_csv('results/scored_cards.csv', index=False)
    print(f"Scored cards   → results/scored_cards.csv")

    # Trigger summary
    print(f"\nTrigger distribution:")
    for ttype, count in scored_df['trigger_type'].value_counts().items():
        print(f"   {ttype:<20} : {count:,} cards")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Uzum Bank dormancy prediction model')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to CSV data file')
    args = parser.parse_args()

    print("\nUZUM BANK: Dormancy Prediction Model (days 15–44 after activation)")
    print("=" * 70)

    df = load_data(args.data_path, fallback_path='data/uzum_hackathon_dataset.csv')
    if df is None:
        return

    print(f"Rows    : {len(df):,}")
    print(f"Cards   : {df['card_id'].nunique():,}")
    print(f"is_active=1: {df.groupby('card_id')['is_active'].first().mean():.1%}")

    print("\nExtracting features from first 14 days of card life...")
    features_df = extract_features(df)

    labeled   = features_df[features_df['has_target_data'] == 1]
    unlabeled = features_df[features_df['has_target_data'] == 0]
    print(f"Cards with features            : {len(features_df):,}")
    print(f"  → Labeled (have target data) : {len(labeled):,}")
    print(f"  → Score-only (no target data): {len(unlabeled):,}")
    print(f"Activated in first 14 days     : {features_df['activated_early'].mean():.1%}")
    print(f"Dormancy rate (days 15–44)     : {labeled['dormant_30d'].mean():.1%}")

    dt, X_test, y_test, y_proba_test, threshold, importance_df = train_model(features_df)

    scored_df = score_all_cards(dt, features_df, threshold)
    save_results(scored_df, X_test, y_test, y_proba_test, threshold, importance_df)

    print(f"\nDone.")


if __name__ == '__main__':
    main()