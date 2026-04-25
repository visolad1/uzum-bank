#!/usr/bin/env python3
"""
Diagnostic analysis of Uzum Bank debit card data.
Dataset: monthly aggregates per category (cnt, amt, is_active).

Answers the following questions:
- In which month of card life do customers most often stop transacting?
- Which MCC categories are entry-level and which indicate a mature user?
- Are there segments that activate quickly but go dormant fast?
- Which region correlates with long-term activity?
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Warning: matplotlib/seaborn not installed. Charts will not be generated.")


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_data(data_path):
    """Load CSV and derive month_of_life for each row."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    df['month'] = pd.to_datetime(df['month'])
    df['card_creation_date'] = pd.to_datetime(df['card_creation_date'])

    # month_of_life: 0 = issuance month, 1 = next month, etc.
    creation_period = df['card_creation_date'].dt.to_period('M')
    month_period = df['month'].dt.to_period('M')
    df['month_of_life'] = (month_period - creation_period).apply(lambda x: x.n)

    return df


def build_card_monthly(df):
    """
    Aggregate to card-month level (sum across categories).

    Note: is_active is a static target label per card (same across all months);
    cnt/amt reflect actual monthly transaction activity.
    """
    card_month = (
        df.groupby(['card_id', 'kiosk_name', 'card_creation_date', 'month', 'month_of_life'])
        .agg(
            total_cnt=('cnt', 'sum'),
            total_amt=('amt', 'sum'),
            n_categories_used=('cnt', lambda x: (x > 0).sum()),
            is_active_target=('is_active', 'max'),   # static target label
        )
        .reset_index()
    )
    # True activity in this month: at least one transaction
    card_month['txn_active'] = (card_month['total_cnt'] > 0).astype(int)
    return card_month


def build_card_summary(card_month):
    """Build a per-card summary: activation, retention, region."""
    txn_active_rows = card_month[card_month['txn_active'] == 1]

    n_obs = card_month.groupby('card_id')['month'].count().rename('n_months_observed')
    n_txn_active = card_month.groupby('card_id')['txn_active'].sum().rename('n_months_with_txn')
    is_active_target = card_month.groupby('card_id')['is_active_target'].first().rename('is_active_target')

    first_txn_mol = txn_active_rows.groupby('card_id')['month_of_life'].min().rename('first_txn_mol')
    last_txn_mol = txn_active_rows.groupby('card_id')['month_of_life'].max().rename('last_txn_mol')
    avg_cats = txn_active_rows.groupby('card_id')['n_categories_used'].mean().rename('avg_categories')

    meta = card_month.groupby('card_id')[['kiosk_name', 'card_creation_date']].first()

    summary = pd.concat([meta, n_obs, n_txn_active, is_active_target, first_txn_mol, last_txn_mol, avg_cats], axis=1).reset_index()
    summary['ever_txn'] = summary['n_months_with_txn'] > 0
    summary['txn_rate'] = summary['n_months_with_txn'] / summary['n_months_observed']
    return summary


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_activation(card_month, summary):
    """
    Q1: In which month of card life do customers most often stop transacting?
    Uses actual transactions (cnt > 0), not the static is_active label.
    """
    print("\n" + "=" * 70)
    print("1.  ACTIVATION & CHURN ANALYSIS (based on cnt > 0)")
    print("=" * 70)

    total = len(summary)
    ever_txn = summary['ever_txn'].sum()
    print(f"\nTotal cards:              {total:,}")
    print(f"Cards with transactions:  {ever_txn:,} ({100 * ever_txn / total:.1f}%)")
    print(f"Target is_active=1:       {int(summary['is_active_target'].sum()):,} "
          f"({100 * summary['is_active_target'].mean():.1f}%)")

    txn_summary = summary[summary['ever_txn']].copy()
    print(f"\nFirst transactional month of card life:")
    print(f"   Mean:    {txn_summary['first_txn_mol'].mean():.2f}")
    print(f"   Median:  {txn_summary['first_txn_mol'].median():.0f}")
    print(f"   Min/Max: {txn_summary['first_txn_mol'].min()} / {txn_summary['first_txn_mol'].max()}")

    # Dormant: had transactions but last_txn_mol < max observed mol
    max_mol = card_month.groupby('card_id')['month_of_life'].max().rename('max_mol')
    churn_data = txn_summary.merge(max_mol, on='card_id')
    churned = churn_data[churn_data['last_txn_mol'] < churn_data['max_mol']].copy()

    print(f"\nDormant cards (had transactions, then stopped): {len(churned):,}")
    if len(churned) > 0:
        last_dist = churned['last_txn_mol'].value_counts().sort_index()
        print(f"\nLast active month of life (distribution):")
        for mol, cnt in last_dist.items():
            pct = 100 * cnt / len(churned)
            bar = '#' * int(pct / 3)
            print(f"   Month {mol}: {cnt:4d} cards ({pct:5.1f}%) {bar}")

    # Correlation between txn_rate and is_active target
    print(f"\nCorrelation between % transactional months and is_active target:")
    for bucket in [(0, 0.01), (0.01, 0.4), (0.4, 0.7), (0.7, 1.01)]:
        subset = summary[(summary['txn_rate'] >= bucket[0]) & (summary['txn_rate'] < bucket[1])]
        if len(subset) > 0:
            target_rate = 100 * subset['is_active_target'].mean()
            label = f"{int(bucket[0]*100)}-{int(bucket[1]*100)}%"
            print(f"   Txn months {label:>8}: {target_rate:.1f}% is_active=1 ({len(subset)} cards)")

    return txn_summary, churned


def analyze_categories(df, card_month):
    """
    Q2: Which categories are entry-level and which indicate a mature user?
    """
    print("\n" + "=" * 70)
    print("2.  MCC CATEGORY ANALYSIS")
    print("=" * 70)

    # First month with any transaction (cnt > 0 in at least one category)
    first_txn_mol = (
        card_month[card_month['txn_active'] == 1]
        .groupby('card_id')['month_of_life'].min()
        .reset_index()
        .rename(columns={'month_of_life': 'first_txn_mol'})
    )
    n_ever_txn = len(first_txn_mol)

    # Categories used in the first transactional month
    df_with_first = df.merge(first_txn_mol, on='card_id')
    entry_rows = df_with_first[
        (df_with_first['month_of_life'] == df_with_first['first_txn_mol']) &
        (df_with_first['cnt'] > 0)
    ]
    entry_cats = entry_rows.groupby('category')['card_id'].nunique().sort_values(ascending=False)

    print(f"\nENTRY CATEGORIES (first transactional month, top 5):")
    for cat, cnt in entry_cats.head(5).items():
        pct = 100 * cnt / n_ever_txn
        print(f"   {cat}: {cnt} cards ({pct:.1f}%)")

    # Mature categories: appear 2+ months after first transaction
    late_rows = df_with_first[
        (df_with_first['month_of_life'] >= df_with_first['first_txn_mol'] + 2) &
        (df_with_first['cnt'] > 0)
    ]
    if len(late_rows) > 0:
        late_cats = late_rows.groupby('category')['card_id'].nunique().sort_values(ascending=False)
        n_late_cards = late_rows['card_id'].nunique()
        print(f"\nMATURE CATEGORIES (appear 2+ months after first txn, top 5):")
        for cat, cnt in late_cats.head(5).items():
            pct = 100 * cnt / n_late_cards
            print(f"   {cat}: {cnt} cards ({pct:.1f}%)")

    # Exclusively late: only appear after the first month, never in it
    if len(late_rows) > 0:
        entry_card_cat = set(zip(entry_rows['card_id'], entry_rows['category']))
        late_rows_copy = late_rows.copy()
        late_rows_copy['in_entry'] = late_rows_copy.apply(
            lambda r: (r['card_id'], r['category']) in entry_card_cat, axis=1
        )
        only_late = late_rows_copy[~late_rows_copy['in_entry']]
        if len(only_late) > 0:
            only_late_cats = only_late.groupby('category')['card_id'].nunique().sort_values(ascending=False)
            print(f"\nEXCLUSIVELY MATURE (never appear in first month):")
            for cat, cnt in only_late_cats.items():
                pct = 100 * cnt / n_late_cards
                print(f"   {cat}: {cnt} cards ({pct:.1f}%)")

    # Category diversity vs is_active target
    card_target = card_month.groupby('card_id')[['is_active_target']].first()
    cat_per_card = (
        df_with_first[df_with_first['cnt'] > 0]
        .groupby('card_id')['category'].nunique()
        .rename('total_cats')
        .reset_index()
    )
    cat_target = cat_per_card.merge(card_target, on='card_id')
    cat_target['bucket'] = pd.cut(cat_target['total_cats'], bins=[0, 1, 2, 3, 20],
                                   labels=['1', '2', '3', '4+'], right=True)

    print(f"\nis_active=1 rate by number of categories used:")
    for b in ['1', '2', '3', '4+']:
        subset = cat_target[cat_target['bucket'] == b]
        if len(subset) > 0:
            rate = 100 * subset['is_active_target'].mean()
            print(f"   {b} categories: {rate:.1f}% is_active=1 ({len(subset)} cards)")

    return first_txn_mol, cat_target


def analyze_channels(card_month, summary):
    """
    Q4: Which region correlates with long-term activity?
    """
    print("\n" + "=" * 70)
    print("3.  REGION ANALYSIS (kiosk_name)")
    print("=" * 70)

    region_stats = (
        summary.groupby('kiosk_name')
        .agg(
            n_cards=('card_id', 'count'),
            pct_ever_txn=('ever_txn', 'mean'),
            avg_txn_rate=('txn_rate', 'mean'),
            pct_is_active=('is_active_target', 'mean'),
        )
        .reset_index()
    )
    region_stats['pct_ever_txn'] *= 100
    region_stats['avg_txn_rate'] *= 100
    region_stats['pct_is_active'] *= 100
    region_stats = region_stats.sort_values('avg_txn_rate', ascending=False)

    print(f"\nTop 10 regions by % transactional months:")
    print(f"   {'Region':<15} {'Cards':>6} {'Ever transacted':>15} {'% txn months':>13} {'is_active=1':>11}")
    print(f"   {'-'*15} {'-'*6} {'-'*15} {'-'*13} {'-'*11}")
    for _, row in region_stats.head(10).iterrows():
        print(f"   {row['kiosk_name']:<15} {int(row['n_cards']):>6} "
              f"{row['pct_ever_txn']:>14.1f}% {row['avg_txn_rate']:>12.1f}% "
              f"{row['pct_is_active']:>10.1f}%")

    return region_stats


def identify_segments(card_month, summary):
    """
    Q3: Behavioral segments based on activation and churn patterns (cnt-based).
    """
    print("\n" + "=" * 70)
    print("4.  BEHAVIORAL SEGMENTS")
    print("=" * 70)

    max_mol = card_month.groupby('card_id')['month_of_life'].max().rename('max_mol')
    cards = summary.merge(max_mol, on='card_id')

    def get_segment(row):
        if not row['ever_txn']:
            return 'Never transacted'
        mol = row['first_txn_mol']
        n_txn = row['n_months_with_txn']
        n_obs = row['n_months_observed']
        last = row['last_txn_mol']
        max_m = row['max_mol']

        fast_start = mol <= 0
        high_stability = n_txn / n_obs >= 0.6
        dropped = last < max_m  # was active but stopped before last observed month

        if fast_start and dropped and n_txn <= 2:
            return 'Fast start / fast sleep'
        if fast_start and high_stability:
            return 'Stable active'
        if fast_start and not high_stability:
            return 'Early start, unstable'
        if mol >= 2 and high_stability:
            return 'Late activation, stable'
        if n_txn == 1:
            return 'One-time use'
        return 'Other'

    cards['segment'] = cards.apply(get_segment, axis=1)

    print(f"\nCustomer segmentation:")
    seg_stats = (
        cards.groupby('segment')
        .agg(
            n=('card_id', 'count'),
            avg_txn_months=('n_months_with_txn', 'mean'),
            pct_is_active=('is_active_target', 'mean'),
        )
        .sort_values('n', ascending=False)
    )
    for seg, row in seg_stats.iterrows():
        print(f"   [{seg}]")
        print(f"      Cards: {int(row['n'])}, "
              f"Avg txn months: {row['avg_txn_months']:.1f}, "
              f"is_active=1: {row['pct_is_active']*100:.1f}%")

    return cards


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_report(card_month, summary, cards_df, region_stats):
    if not PLOT_AVAILABLE:
        print("\nVisualization unavailable (install matplotlib and seaborn).")
        return

    print("\nGenerating charts...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Uzum Bank: Diagnostics of Debit Card Activity', fontsize=15, fontweight='bold')

    # 1. Первый транзакционный месяц жизни
    txn_cards = summary[summary['ever_txn']]
    mol_dist = txn_cards['first_txn_mol'].value_counts().sort_index()
    axes[0, 0].bar(mol_dist.index, mol_dist.values, color='steelblue')
    axes[0, 0].set_xlabel('Month of card life')
    axes[0, 0].set_ylabel('Number of cards')
    axes[0, 0].set_title('First transaction: month of card life')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. Регионы: % транзакционных месяцев (топ-15)
    top_regions = region_stats.head(15)
    axes[0, 1].barh(top_regions['kiosk_name'], top_regions['avg_txn_rate'], color='coral')
    axes[0, 1].set_xlabel('% months with transactions')
    axes[0, 1].set_title('Regions: avg % transactional months')
    axes[0, 1].grid(axis='x', alpha=0.3)

    # 3. Сегменты — количество карт
    seg_counts = cards_df['segment'].value_counts()
    axes[1, 0].barh(seg_counts.index, seg_counts.values, color='mediumseagreen')
    axes[1, 0].set_xlabel('Number of cards')
    axes[1, 0].set_title('Behavioral segments')
    axes[1, 0].grid(axis='x', alpha=0.3)

    # 4. % карт с транзакциями по месяцам жизни
    mol_activity = (
        card_month.groupby('month_of_life')['txn_active']
        .mean()
        .reset_index()
    )
    mol_activity['txn_active'] *= 100
    axes[1, 1].plot(mol_activity['month_of_life'], mol_activity['txn_active'],
                    marker='o', linewidth=2, color='darkorange')
    axes[1, 1].set_xlabel('Month of card life')
    axes[1, 1].set_ylabel('% cards with transactions')
    axes[1, 1].set_title('Transaction activity by month of life')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/diagnostics.png', dpi=150, bbox_inches='tight')
    print("Chart saved to results/diagnostics.png")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Uzum Bank debit card diagnostic analysis')
    parser.add_argument('--data-path', type=str,
                        default='data/uzum_hackathon_dataset.csv',
                        help='Path to the CSV data file')
    args = parser.parse_args()

    print("\nUZUM BANK: Debit Card Activity Diagnostics")
    print("=" * 70)

    df = load_data(args.data_path)
    print(f"\nRows loaded:    {len(df):,}")
    print(f"Unique cards:   {df['card_id'].nunique():,}")
    print(f"Period:         {df['month'].min().strftime('%Y-%m')} - {df['month'].max().strftime('%Y-%m')}")
    print(f"Categories:     {df['category'].nunique()}")
    print(f"Regions:        {df['kiosk_name'].nunique()}")

    card_month = build_card_monthly(df)
    summary = build_card_summary(card_month)

    active_summary, churned = analyze_activation(card_month, summary)
    first_txn_mol, cat_target = analyze_categories(df, card_month)
    region_stats = analyze_channels(card_month, summary)
    cards_df = identify_segments(card_month, summary)

    if PLOT_AVAILABLE:
        create_report(card_month, summary, cards_df, region_stats)

    # Save results
    Path('results').mkdir(exist_ok=True)

    diagnostics = {
        'total_cards': int(summary['card_id'].nunique()),
        'ever_txn_cards': int(summary['ever_txn'].sum()),
        'ever_txn_rate_pct': float(100 * summary['ever_txn'].mean()),
        'is_active_target_pct': float(100 * summary['is_active_target'].mean()),
        'churned_cards': int(len(churned)),
        'segments': cards_df['segment'].value_counts().to_dict(),
    }

    with open('results/diagnostics.json', 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to results/")
    print(f"\nDiagnostics complete.")


if __name__ == '__main__':
    main()
