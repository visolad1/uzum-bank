#!/usr/bin/env python3
"""
Bonus / trigger recommendation logic for churn prevention.

Pipeline
--------
1. Load raw data (for top-category lookup and recent-activity check).
2. Load churn scores from results/scored_cards.csv (produced by model.py)
   or fall back to heuristic scoring.
3. For each predicted-dormant card that is still recently active, assign:
   - a personalised bonus offer based on top MCC category
   - a send timing and channel based on risk level
4. Export to results/bonus_candidates.csv + .json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
from data_loader import load_data


# ---------------------------------------------------------------------------
# Bonus catalogue  (keyed by actual dataset category names)
# ---------------------------------------------------------------------------

BONUS_OFFERS = {
    'Онлайн оплаты': {
        'message':    'Кэшбэк 4% на онлайн-оплаты — совершайте покупки онлайн!',
        'bonus_sums': 200,
        'trigger':    'cashback_online',
    },
    'Оффлайн оплаты': {
        'message':    'Кэшбэк 3% на оплаты в магазинах — платите картой!',
        'bonus_sums': 150,
        'trigger':    'cashback_offline',
    },
    'Перевод внутри страны - пополнение': {
        'message':    'Бонус 1% на входящие переводы — получайте выгоду!',
        'bonus_sums': 100,
        'trigger':    'transfer_bonus',
    },
    'Перевод внутри страны - списание': {
        'message':    'Бонус 1% на переводы другим — отправляйте с выгодой!',
        'bonus_sums': 100,
        'trigger':    'transfer_bonus',
    },
    'Пополнение через АТМ - Kапитал Банк': {
        'message':    'Без комиссии на пополнение через банкомат Капитал Банк!',
        'bonus_sums': 0,
        'trigger':    'atm_fee_waiver',
    },
    'Снятие наличных в АТМ - Kапитал Банк': {
        'message':    'Без комиссии на снятие в банкоматах Капитал Банк!',
        'bonus_sums': 0,
        'trigger':    'atm_fee_waiver',
    },
    'Снятие наличных в АТМ - Другие банки': {
        'message':    'Кэшбэк 1% при снятии в других банкоматах!',
        'bonus_sums': 50,
        'trigger':    'atm_cashback',
    },
    'Кэш кредит Nasiya - пополнение': {
        'message':    'Бонус 500 сум за регулярное погашение Nasiya!',
        'bonus_sums': 500,
        'trigger':    'nasiya_loyalty',
    },
    'Трансгран - пополнение': {
        'message':    'Специальный курс на международные переводы!',
        'bonus_sums': 200,
        'trigger':    'cross_border_offer',
    },
    'Возвраты/отмены - Оплаты': {
        'message':    'Кэшбэк 2% на все оплаты — больше транзакций, больше выгоды!',
        'bonus_sums': 100,
        'trigger':    'general_cashback',
    },
    'Остальное': {
        'message':    'Кэшбэк 2% на любые операции картой!',
        'bonus_sums': 100,
        'trigger':    'general_cashback',
    },
}

DEFAULT_OFFER = {
    'message':    'Кэшбэк 2% на все операции — активируйте карту!',
    'bonus_sums': 100,
    'trigger':    'general_cashback',
}

# Send timing per risk level
SEND_TIMING = {
    'CRITICAL': {'send_day': 1,  'channel': 'push + sms'},
    'HIGH':     {'send_day': 3,  'channel': 'push'},
    'MEDIUM':   {'send_day': 7,  'channel': 'in-app'},
    'LOW':      {'send_day': 14, 'channel': 'in-app'},
}


# ---------------------------------------------------------------------------
# Scoring fallback (when scored_cards.csv is not available)
# ---------------------------------------------------------------------------

def heuristic_score(df):
    """
    Rule-based risk scoring aligned with the model's 0–14 day window.
    Used when model.py has not been run yet.
    """
    df = df.copy()
    df['days_from_creation'] = (
        pd.to_datetime(df['month']) - pd.to_datetime(df['card_creation_date'])
    ).dt.days

    early = df[(df['days_from_creation'] >= 0) & (df['days_from_creation'] <= 14)]
    early_active = early[early['cnt'] > 0]

    base = early.groupby('card_id').agg(
        cnt_early=('cnt', 'sum'),
    ).reset_index()
    cats = (
        early_active.groupby('card_id')['category'].nunique()
        .rename('n_cats_early').reset_index()
    )

    meta = df.groupby('card_id').agg(
        creation_dow=('card_creation_date', lambda x: x.iloc[0].dayofweek),
        kiosk_name=('kiosk_name', 'first'),
        is_active=('is_active', 'max'),
    ).reset_index()

    scored = (
        meta.merge(base, on='card_id', how='left')
            .merge(cats, on='card_id', how='left')
            .fillna(0)
    )
    scored['activated_early'] = (scored['cnt_early'] > 0).astype(int)

    def _score(row):
        s = 0.0
        if row['activated_early'] == 0: s += 0.5
        if row['cnt_early'] < 2:        s += 0.3
        if row['n_cats_early'] < 2:     s += 0.2
        return min(s, 1.0)

    def _risk(p):
        if p >= 0.70: return 'CRITICAL'
        if p >= 0.50: return 'HIGH'
        if p >= 0.30: return 'MEDIUM'
        return 'LOW'

    scored['dormant_30d_proba'] = scored.apply(_score, axis=1)
    scored['predicted_dormant'] = (scored['dormant_30d_proba'] >= 0.5).astype(int)
    scored['risk_level']        = scored['dormant_30d_proba'].apply(_risk)
    return scored


# ---------------------------------------------------------------------------
# Pre-computation helpers (vectorized — no per-card loops)
# ---------------------------------------------------------------------------

def precompute_top_categories(df):
    """
    Return a Series {card_id: top_category} based on highest total cnt.
    Vectorized — runs once over the full dataframe.
    """
    return (
        df[df['cnt'] > 0]
        .groupby(['card_id', 'category'])['cnt']
        .sum()
        .reset_index()
        .sort_values('cnt', ascending=False)
        .groupby('card_id')['category']
        .first()
    )


def precompute_recently_active(df, n_months=2):
    """
    Return a set of card_ids that had at least one transaction (cnt > 0)
    in the last n_months of their observed history.
    Vectorized — runs once.
    """
    max_mol = df.groupby('card_id')['month_of_life'].max().rename('max_mol')
    df_with_max = df.merge(max_mol, on='card_id')

    recent = df_with_max[
        (df_with_max['month_of_life'] >= df_with_max['max_mol'] - n_months + 1) &
        (df_with_max['cnt'] > 0)
    ]
    return set(recent['card_id'].unique())


# ---------------------------------------------------------------------------
# Core recommendation builder
# ---------------------------------------------------------------------------

def build_recommendations(df, scored):
    """
    For each predicted-dormant card that is still recently active,
    return a personalised bonus offer + send timing.
    """
    print("\n" + "=" * 70)
    print("BONUS TRIGGER RECOMMENDATIONS")
    print("=" * 70)

    # Support both model.py output and heuristic fallback column names
    proba_col   = 'dormant_30d_proba' if 'dormant_30d_proba' in scored.columns else 'churn_proba'
    predict_col = 'predicted_dormant'  if 'predicted_dormant'  in scored.columns else 'predicted_churn'

    at_risk = scored[scored[predict_col] == 1].copy()
    print(f"\nPredicted-dormant cards : {len(at_risk):,} / {len(scored):,}")

    # Pre-compute lookups — O(n) instead of O(n * cards)
    print("Pre-computing top categories and recent activity...")
    top_cats       = precompute_top_categories(df)
    recently_active = precompute_recently_active(df, n_months=2)

    candidates = []
    skipped = 0
    for _, row in at_risk.iterrows():
        card_id = row['card_id']

        # Only target cards that still have recent activity (reachable)
        if card_id not in recently_active:
            skipped += 1
            continue

        top_cat = top_cats.get(card_id)
        offer   = BONUS_OFFERS.get(top_cat, DEFAULT_OFFER) if top_cat else DEFAULT_OFFER
        timing  = SEND_TIMING.get(str(row['risk_level']), SEND_TIMING['MEDIUM'])

        candidates.append({
            'card_id':      card_id,
            'kiosk_name':   row.get('kiosk_name', ''),
            'dormant_proba': round(float(row[proba_col]), 3),
            'risk_level':   row['risk_level'],
            'top_category': top_cat or 'N/A',
            'trigger':      offer['trigger'],
            'message':      offer['message'],
            'bonus_sums':   offer['bonus_sums'],
            'send_day':     timing['send_day'],
            'channel':      timing['channel'],
        })

    print(f"Skipped (not recently active): {skipped:,}")

    result_df = (
        pd.DataFrame(candidates)
        .sort_values('dormant_proba', ascending=False)
        .reset_index(drop=True)
    )

    print(f"Recently-active at-risk cards (bonus candidates) : {len(result_df):,}")

    if len(result_df) > 0:
        print(f"\nBy risk level:")
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            n = (result_df['risk_level'] == level).sum()
            if n > 0:
                t = SEND_TIMING[level]
                print(f"   {level:<10}: {n:4,} cards "
                      f"→ send day {t['send_day']}, channel: {t['channel']}")

        print(f"\nTop categories among at-risk cards:")
        for cat, n in result_df['top_category'].value_counts().head(5).items():
            print(f"   {cat}: {n} ({100*n/len(result_df):.1f}%)")

        print(f"\nSample recommendations (top 3 by dormancy risk):")
        for _, r in result_df.head(3).iterrows():
            print(f"\n   Card     : {r['card_id']}  |  Risk: {r['risk_level']} ({r['dormant_proba']:.2f})")
            print(f"   Category : {r['top_category']}")
            print(f"   Message  : {r['message']}")
            print(f"   Bonus    : {r['bonus_sums']} сум")
            print(f"   Send     : day {r['send_day']} via {r['channel']}")

    return result_df


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_recommendations(recs_df):
    Path('results').mkdir(exist_ok=True)
    recs_df.to_csv('results/bonus_candidates.csv', index=False)
    recs_df.to_json('results/bonus_candidates.json', orient='records',
                    force_ascii=False, indent=2)
    print(f"\nResults saved:")
    print(f"   results/bonus_candidates.csv  ({len(recs_df):,} records)")
    print(f"   results/bonus_candidates.json ({len(recs_df):,} records)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Bonus trigger recommendations for at-risk Uzum Bank cards'
    )
    parser.add_argument('--data-path',   type=str, default=None)
    parser.add_argument('--scores-path', type=str, default='results/scored_cards.csv',
                        help='scored_cards.csv from model.py (optional, falls back to heuristic)')
    args = parser.parse_args()

    print("\nUZUM BANK: Bonus Trigger Logic")
    print("=" * 70)

    df = load_data(args.data_path, fallback_path='data/uzum_hackathon_dataset.csv')
    if df is None:
        return

    scores_path = Path(args.scores_path)
    if scores_path.exists():
        print(f"Using model scores from {scores_path}")
        scored = pd.read_csv(scores_path)
        # Merge kiosk_name if absent
        if 'kiosk_name' not in scored.columns:
            meta = df.groupby('card_id')['kiosk_name'].first().reset_index()
            scored = scored.merge(meta, on='card_id', how='left')
    else:
        print("Scored cards not found — using heuristic scoring.")
        scored = heuristic_score(df)

    recs = build_recommendations(df, scored)

    if len(recs) > 0:
        export_recommendations(recs)
    else:
        print("\nNo bonus candidates found.")

    print("\nDone.")


if __name__ == '__main__':
    main()