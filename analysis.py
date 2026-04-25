#!/usr/bin/env python3
"""
Диагностика данных о дебетовых картах Uzum Bank.
Датасет: помесячные агрегаты по категориям (cnt, amt, is_active).

Отвечает на вопросы:
- На каком месяце жизни карты клиент чаще перестаёт платить?
- Какие MCC-категории входные, какие — признак зрелого пользователя?
- Есть ли сегменты, которые активируются быстро, но быстро засыпают?
- Какой регион коррелирует с долгосрочной активностью?
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
    print("⚠️  matplotlib/seaborn не установлены. Графики не будут созданы.")


# ---------------------------------------------------------------------------
# Загрузка и подготовка
# ---------------------------------------------------------------------------

def load_data(data_path):
    """Загружает CSV и добавляет month_of_life."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {data_path}")

    print(f"📂 Загружаю данные из {data_path}...")
    df = pd.read_csv(data_path)

    df['month'] = pd.to_datetime(df['month'])
    df['card_creation_date'] = pd.to_datetime(df['card_creation_date'])

    # Месяц жизни карты: 0 = месяц выпуска, 1 = следующий месяц и т.д.
    creation_period = df['card_creation_date'].dt.to_period('M')
    month_period = df['month'].dt.to_period('M')
    df['month_of_life'] = (month_period - creation_period).apply(lambda x: x.n)

    return df


def build_card_monthly(df):
    """
    Агрегирует до уровня карта-месяц (суммирует по категориям).

    Примечание: is_active — статичный таргет-флаг карты (одинаков для всех
    месяцев), cnt/amt — реальная помесячная активность.
    """
    card_month = (
        df.groupby(['card_id', 'kiosk_name', 'card_creation_date', 'month', 'month_of_life'])
        .agg(
            total_cnt=('cnt', 'sum'),
            total_amt=('amt', 'sum'),
            n_categories_used=('cnt', lambda x: (x > 0).sum()),
            is_active_target=('is_active', 'max'),   # таргет-флаг (статичный)
        )
        .reset_index()
    )
    # Реальная активность в этом месяце: хотя бы одна транзакция
    card_month['txn_active'] = (card_month['total_cnt'] > 0).astype(int)
    return card_month


def build_card_summary(card_month):
    """Сводка по каждой карте: активация, удержание, регион."""
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
# Анализы
# ---------------------------------------------------------------------------

def analyze_activation(card_month, summary):
    """
    Q1: На каком месяце жизни карты клиент чаще перестаёт платить?
    Используем реальные транзакции (cnt > 0), не статичный is_active.
    """
    print("\n" + "=" * 70)
    print("1️⃣  АНАЛИЗ АКТИВАЦИИ И ОТТОКА (по cnt > 0)")
    print("=" * 70)

    total = len(summary)
    ever_txn = summary['ever_txn'].sum()
    print(f"\n📊 Карт всего:              {total:,}")
    print(f"📊 Совершили транзакцию:    {ever_txn:,} ({100 * ever_txn / total:.1f}%)")
    print(f"📊 Таргет is_active=1:      {int(summary['is_active_target'].sum()):,} "
          f"({100 * summary['is_active_target'].mean():.1f}%)")

    txn_summary = summary[summary['ever_txn']].copy()
    print(f"\n⏱️  Первый месяц жизни с транзакцией:")
    print(f"   Среднее:  {txn_summary['first_txn_mol'].mean():.2f}")
    print(f"   Медиана:  {txn_summary['first_txn_mol'].median():.0f}")
    print(f"   Мин/Макс: {txn_summary['first_txn_mol'].min()} / {txn_summary['first_txn_mol'].max()}")

    # Уснувшие: были транзакции, но last_txn_mol < max наблюдаемого mol
    max_mol = card_month.groupby('card_id')['month_of_life'].max().rename('max_mol')
    churn_data = txn_summary.merge(max_mol, on='card_id')
    churned = churn_data[churn_data['last_txn_mol'] < churn_data['max_mol']].copy()

    print(f"\n📉 Уснувших карт (были транзакции, потом прекратились): {len(churned):,}")
    if len(churned) > 0:
        last_dist = churned['last_txn_mol'].value_counts().sort_index()
        print(f"\n🔻 Последний активный месяц жизни (распределение):")
        for mol, cnt in last_dist.items():
            pct = 100 * cnt / len(churned)
            bar = '█' * int(pct / 3)
            print(f"   Месяц {mol}: {cnt:4d} карт ({pct:5.1f}%) {bar}")

    # Корреляция is_active с txn_rate
    print(f"\n🎯 Связь между % активных месяцев и таргетом is_active:")
    for bucket in [(0, 0.01), (0.01, 0.4), (0.4, 0.7), (0.7, 1.01)]:
        subset = summary[(summary['txn_rate'] >= bucket[0]) & (summary['txn_rate'] < bucket[1])]
        if len(subset) > 0:
            target_rate = 100 * subset['is_active_target'].mean()
            label = f"{int(bucket[0]*100)}-{int(bucket[1]*100)}%"
            print(f"   Транзакций {label:>8} мес.: {target_rate:.1f}% is_active=1 ({len(subset)} карт)")

    return txn_summary, churned


def analyze_categories(df, card_month):
    """
    Q2: Какие категории входные, какие — зрелые?
    """
    print("\n" + "=" * 70)
    print("2️⃣  АНАЛИЗ MCC-КАТЕГОРИЙ")
    print("=" * 70)

    # Первый месяц с транзакциями (cnt > 0 хоть в одной категории)
    first_txn_mol = (
        card_month[card_month['txn_active'] == 1]
        .groupby('card_id')['month_of_life'].min()
        .reset_index()
        .rename(columns={'month_of_life': 'first_txn_mol'})
    )
    n_ever_txn = len(first_txn_mol)

    # Категории в первый транзакционный месяц
    df_with_first = df.merge(first_txn_mol, on='card_id')
    entry_rows = df_with_first[
        (df_with_first['month_of_life'] == df_with_first['first_txn_mol']) &
        (df_with_first['cnt'] > 0)
    ]
    entry_cats = entry_rows.groupby('category')['card_id'].nunique().sort_values(ascending=False)

    print(f"\n📍 ВХОДНЫЕ КАТЕГОРИИ (первый транзакционный месяц, топ-5):")
    for cat, cnt in entry_cats.head(5).items():
        pct = 100 * cnt / n_ever_txn
        print(f"   {cat}: {cnt} карт ({pct:.1f}%)")

    # Зрелые категории: появляются через 2+ месяца после первой транзакции
    late_rows = df_with_first[
        (df_with_first['month_of_life'] >= df_with_first['first_txn_mol'] + 2) &
        (df_with_first['cnt'] > 0)
    ]
    if len(late_rows) > 0:
        late_cats = late_rows.groupby('category')['card_id'].nunique().sort_values(ascending=False)
        n_late_cards = late_rows['card_id'].nunique()
        print(f"\n👑 ЗРЕЛЫЕ КАТЕГОРИИ (появляются 2+ мес. после старта, топ-5):")
        for cat, cnt in late_cats.head(5).items():
            pct = 100 * cnt / n_late_cards
            print(f"   {cat}: {cnt} карт ({pct:.1f}%)")

    # Уникально-поздние: только в поздних месяцах, не в первом
    if len(late_rows) > 0:
        entry_card_cat = set(zip(entry_rows['card_id'], entry_rows['category']))
        late_rows_copy = late_rows.copy()
        late_rows_copy['in_entry'] = late_rows_copy.apply(
            lambda r: (r['card_id'], r['category']) in entry_card_cat, axis=1
        )
        only_late = late_rows_copy[~late_rows_copy['in_entry']]
        if len(only_late) > 0:
            only_late_cats = only_late.groupby('category')['card_id'].nunique().sort_values(ascending=False)
            print(f"\n🔮 ЭКСКЛЮЗИВНО ЗРЕЛЫЕ (появляются только после первого мес.):")
            for cat, cnt in only_late_cats.items():
                pct = 100 * cnt / n_late_cards
                print(f"   {cat}: {cnt} карт ({pct:.1f}%)")

    # Диверсификация vs is_active target
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

    print(f"\n🎯 is_active=1 по числу использованных категорий:")
    for b in ['1', '2', '3', '4+']:
        subset = cat_target[cat_target['bucket'] == b]
        if len(subset) > 0:
            rate = 100 * subset['is_active_target'].mean()
            print(f"   {b} категорий: {rate:.1f}% is_active=1 ({len(subset)} карт)")

    return first_txn_mol, cat_target


def analyze_channels(card_month, summary):
    """
    Q4: Какой регион коррелирует с долгосрочной активностью?
    """
    print("\n" + "=" * 70)
    print("3️⃣  АНАЛИЗ РЕГИОНОВ (kiosk_name)")
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

    print(f"\n🏆 Топ-10 регионов по % транзакционных месяцев:")
    print(f"   {'Регион':<15} {'Карт':>6} {'С транзакцией':>14} {'% тр.мес.':>10} {'is_active=1':>11}")
    print(f"   {'-'*15} {'-'*6} {'-'*14} {'-'*10} {'-'*11}")
    for _, row in region_stats.head(10).iterrows():
        print(f"   {row['kiosk_name']:<15} {int(row['n_cards']):>6} "
              f"{row['pct_ever_txn']:>13.1f}% {row['avg_txn_rate']:>9.1f}% "
              f"{row['pct_is_active']:>10.1f}%")

    return region_stats


def identify_segments(card_month, summary):
    """
    Q3: Сегменты по поведению активации и оттока (на основе cnt).
    """
    print("\n" + "=" * 70)
    print("4️⃣  ПОВЕДЕНЧЕСКИЕ СЕГМЕНТЫ")
    print("=" * 70)

    max_mol = card_month.groupby('card_id')['month_of_life'].max().rename('max_mol')
    cards = summary.merge(max_mol, on='card_id')

    def get_segment(row):
        if not row['ever_txn']:
            return 'Никогда не транзакционировал'
        mol = row['first_txn_mol']
        n_txn = row['n_months_with_txn']
        n_obs = row['n_months_observed']
        last = row['last_txn_mol']
        max_m = row['max_mol']

        fast_start = mol <= 0
        high_stability = n_txn / n_obs >= 0.6
        dropped = last < max_m  # был активен, но последний мес. — не последний набл.

        if fast_start and dropped and n_txn <= 2:
            return 'Быстрый старт / быстрый сон'
        if fast_start and high_stability:
            return 'Стабильный активный'
        if fast_start and not high_stability:
            return 'Ранний старт, нестабильный'
        if mol >= 2 and high_stability:
            return 'Поздняя активация, стабильный'
        if n_txn == 1:
            return 'Разовое использование'
        return 'Прочее'

    cards['segment'] = cards.apply(get_segment, axis=1)

    print(f"\n📊 Сегментация клиентов:")
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
        print(f"      Карт: {int(row['n'])}, "
              f"Средн. тр.мес.: {row['avg_txn_months']:.1f}, "
              f"is_active=1: {row['pct_is_active']*100:.1f}%")

    return cards


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def create_report(card_month, summary, cards_df, region_stats):
    if not PLOT_AVAILABLE:
        print("\n⚠️  Визуализация недоступна.")
        return

    print("\n📊 Создаю визуализацию...")
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
    print("💾 График сохранён в results/diagnostics.png")
    plt.close()


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Диагностика данных карт Узум Банка')
    parser.add_argument('--data-path', type=str,
                        default='data/uzum_hackathon_dataset.csv',
                        help='Путь к CSV файлу с данными')
    args = parser.parse_args()

    print("\n🏦 УЗУМ БАНК: Диагностика активности дебетовых карт")
    print("=" * 70)

    df = load_data(args.data_path)
    print(f"\n✅ Загружено строк:    {len(df):,}")
    print(f"✅ Уникальных карт:   {df['card_id'].nunique():,}")
    print(f"✅ Период:            {df['month'].min().strftime('%Y-%m')} — {df['month'].max().strftime('%Y-%m')}")
    print(f"✅ Категорий:         {df['category'].nunique()}")
    print(f"✅ Регионов:          {df['kiosk_name'].nunique()}")

    card_month = build_card_monthly(df)
    summary = build_card_summary(card_month)

    active_summary, churned = analyze_activation(card_month, summary)
    first_txn_mol, cat_target = analyze_categories(df, card_month)
    region_stats = analyze_channels(card_month, summary)
    cards_df = identify_segments(card_month, summary)

    if PLOT_AVAILABLE:
        create_report(card_month, summary, cards_df, region_stats)

    # Сохраняем результаты
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

    print(f"\n💾 Результаты сохранены в results/")
    print(f"\n✅ Диагностика завершена!")


if __name__ == '__main__':
    main()
