#!/usr/bin/env python3
"""
Диагностика данных о дебетовых картах Uzum Bank.
Отвечает на вопросы:
- На каком дне жизни карты клиент чаще перестаёт платить?
- Какие MCC-категории входные, какие — признак зрелого пользователя?
- Есть ли сегменты, которые активируются быстро, но быстро засыпают?
- Какой канал коррелирует с долгосрочной активностью?
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import json
import argparse
from pathlib import Path

# Для визуализации (опционально)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("⚠️  matplotlib/seaborn не установлены. Графики не будут созданы.")


def generate_sample_data(n_cards=500):
    """
    Генерирует синтетический датасет для демонстрации.
    """
    np.random.seed(42)
    
    data = []
    channels = ['POS', 'Online', 'In-app']
    categories = ['Supermarket', 'Restaurant', 'Transport', 'Telecom', 'Online Shopping', 'Gas Station']
    
    issue_date = pd.Timestamp('2026-03-01')
    
    for card_id in range(n_cards):
        # Вероятность активации: 70%
        if np.random.random() > 0.7:
            continue  # Неактивная карта
        
        # День первой транзакции: в пределах 14 дней
        days_to_activation = np.random.randint(1, 15)
        first_txn_date = issue_date + timedelta(days=days_to_activation)
        
        # Количество транзакций в первую неделю (Пуассон)
        txn_week1 = np.random.poisson(2) + 1
        
        # Вероятность удержания зависит от быстроты активации и разнообразия
        retention_prob = 0.5
        if days_to_activation <= 3:
            retention_prob += 0.2  # Быстрая активация — лучше удержание
        if txn_week1 >= 3:
            retention_prob += 0.2  # Много трансакций — лучше удержание
        
        active_day30 = 1 if np.random.random() < retention_prob else 0
        
        # Информация о канале и категориях
        first_channel = np.random.choice(channels)
        n_categories = np.random.randint(1, 5) if active_day30 else np.random.randint(1, 2)
        
        has_cash_withdrawal = 1 if np.random.random() < 0.3 else 0
        is_transfer_only = 1 if np.random.random() < 0.1 and not has_cash_withdrawal else 0
        
        # Генерируем транзакции
        for txn_day in range(1, 31):
            txn_date = issue_date + timedelta(days=txn_day)
            
            # Вероятность транзакции
            if active_day30:
                prob_txn = 0.3 if txn_day <= 7 else 0.15
            else:
                prob_txn = 0.2 if txn_day <= 3 else 0.0
            
            if np.random.random() < prob_txn:
                channel = np.random.choice(channels)
                category = np.random.choice(categories, size=min(n_categories, 3))[0]
                
                data.append({
                    'card_id': f'CARD_{card_id:06d}',
                    'issue_date': issue_date.strftime('%Y-%m-%d'),
                    'first_transaction_date': first_txn_date.strftime('%Y-%m-%d'),
                    'transaction_date': txn_date.strftime('%Y-%m-%d'),
                    'channel': channel,
                    'mcc_category': category,
                    'is_cash_withdrawal': has_cash_withdrawal,
                    'is_transfer_only': is_transfer_only,
                    'active_day30': active_day30
                })
    
    return pd.DataFrame(data)


def load_data(data_path=None):
    """
    Загружает данные либо из файла, либо генерирует синтетические.
    """
    if data_path and Path(data_path).exists():
        print(f"📂 Загружаю данные из {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("⚙️  Генерирую синтетический датасет для демонстрации...")
        df = generate_sample_data(n_cards=500)
        # Сохраняю для справки
        Path('data').mkdir(exist_ok=True)
        df.to_csv('data/sample_data.csv', index=False)
        print("💾 Пример датасета сохранён в data/sample_data.csv")
    
    # Конвертируем даты
    date_cols = ['issue_date', 'first_transaction_date', 'transaction_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df


def analyze_activation(df):
    """
    Анализирует скорость активации карт и корреляцию с удержанием.
    """
    print("\n" + "="*70)
    print("1️⃣  АНАЛ��З АКТИВАЦИИ")
    print("="*70)
    
    # Карты с транзакциями
    activated = df[df['first_transaction_date'].notna()]['card_id'].nunique()
    total = df['card_id'].nunique()
    activation_rate = 100 * activated / total if total > 0 else 0
    
    print(f"\n📊 Активированы: {activated} из {total} карт ({activation_rate:.1f}%)")
    
    # Дни до активации
    df_active = df[df['first_transaction_date'].notna()].copy()
    df_active['days_to_activation'] = (df_active['first_transaction_date'] - df_active['issue_date']).dt.days
    
    print(f"\n⏱️  Дни до первой транзакции:")
    print(f"   Средний: {df_active['days_to_activation'].mean():.1f} дней")
    print(f"   Медиана: {df_active['days_to_activation'].median():.0f} дней")
    print(f"   Мин/Макс: {df_active['days_to_activation'].min()}/{df_active['days_to_activation'].max()} дней")
    
    # Активация и удержание
    retention_by_activation = df_active.groupby('days_to_activation')['active_day30'].agg(['sum', 'count', 'mean'])
    retention_by_activation.columns = ['retained', 'total', 'retention_rate']
    retention_by_activation['retention_rate'] = 100 * retention_by_activation['retention_rate']
    
    print(f"\n🎯 Удержание по дням активации:")
    for days in [1, 3, 7, 14]:
        if days in retention_by_activation.index:
            rate = retention_by_activation.loc[days, 'retention_rate']
            print(f"   День {days}: {rate:.1f}% активны на день 30")
    
    return df_active, retention_by_activation


def analyze_categories(df):
    """
    Анализирует роль MCC-категорий в удержании.
    """
    print("\n" + "="*70)
    print("2️⃣  АНАЛИЗ MCC-КАТЕГОРИЙ")
    print("="*70)
    
    # Какие категории чаще встречаются в первых транзакциях?
    df_active = df[df['first_transaction_date'].notna()].copy()
    
    # Первая категория каждой карты
    first_txn = df_active.sort_values('transaction_date').drop_duplicates('card_id')
    first_categories = first_txn.groupby('mcc_category').size().sort_values(ascending=False)
    
    print(f"\n📍 ВХОДНЫЕ КАТЕГОРИИ (первые транзакции):")
    for cat, count in first_categories.head(5).items():
        pct = 100 * count / len(first_txn)
        print(f"   {cat}: {count} ({pct:.1f}%)")
    
    # Диверсификация и удержание
    card_categories = df_active.groupby('card_id')['mcc_category'].nunique().reset_index()
    card_categories.columns = ['card_id', 'n_categories']
    
    card_active = df_active.groupby('card_id')['active_day30'].first().reset_index()
    card_data = card_categories.merge(card_active, on='card_id')
    
    print(f"\n🎯 Удержание по разнообразию категорий:")
    for n_cat in sorted(card_data['n_categories'].unique()):
        subset = card_data[card_data['n_categories'] == n_cat]
        retention = 100 * subset['active_day30'].mean()
        count = len(subset)
        print(f"   {n_cat} категорий: {retention:.1f}% удержания ({count} карт)")
    
    # Зрелые пользователи
    mature_threshold = 3
    mature_users = card_data[card_data['n_categories'] >= mature_threshold]
    mature_retention = 100 * mature_users['active_day30'].mean()
    
    print(f"\n👑 ЗРЕЛЫЕ ПОЛЬЗОВАТЕЛИ (3+ категорий): {mature_retention:.1f}% удержания")
    
    return card_data


def analyze_channels(df):
    """
    Анализирует влияние каналов платежа на удержание.
    """
    print("\n" + "="*70)
    print("3️⃣  АНАЛИЗ КАНАЛОВ ПЛАТЕЖА")
    print("="*70)
    
    df_active = df[df['first_transaction_date'].notna()].copy()
    
    # Канал первой транзакции
    first_txn = df_active.sort_values('transaction_date').drop_duplicates('card_id')
    
    print(f"\n🛒 Канал первой транзакции и удержание:")
    for channel in sorted(first_txn['channel'].unique()):
        subset = first_txn[first_txn['channel'] == channel]
        retention = 100 * subset['active_day30'].mean()
        count = len(subset)
        print(f"   {channel}: {retention:.1f}% удержания ({count} карт)")


def identify_segments(df):
    """
    Выявляет поведенческие сегменты клиентов.
    """
    print("\n" + "="*70)
    print("4️⃣  ПОВЕДЕНЧЕСКИЕ СЕГМЕНТЫ")
    print("="*70)
    
    df_active = df[df['first_transaction_date'].notna()].copy()
    df_active['days_to_activation'] = (df_active['first_transaction_date'] - df_active['issue_date']).dt.days
    
    # Характеристики по карте
    card_stats = []
    for card_id in df_active['card_id'].unique():
        card_df = df_active[df_active['card_id'] == card_id]
        
        days_to_activation = card_df['days_to_activation'].iloc[0]
        n_txn_week1 = len(card_df[card_df['transaction_date'] <= 
                                   card_df['first_transaction_date'] + timedelta(days=7)])
        n_categories = card_df['mcc_category'].nunique()
        active_day30 = card_df['active_day30'].iloc[0]
        
        card_stats.append({
            'card_id': card_id,
            'days_to_activation': days_to_activation,
            'txn_week1': n_txn_week1,
            'categories': n_categories,
            'active_day30': active_day30
        })
    
    cards_df = pd.DataFrame(card_stats)
    
    # Сегменты
    def get_segment(row):
        if row['active_day30'] == 0:
            if row['days_to_activation'] <= 3 and row['txn_week1'] >= 2:
                return '⚡ Быстрый старт, быстрый отток'
            else:
                return '❌ Медленный старт'
        else:
            if row['categories'] == 1:
                return '🔗 Один магазин'
            elif row['categories'] < 3:
                return '📍 Двухканальный'
            else:
                return '🌟 Многоканальный (зрелый)'
    
    cards_df['segment'] = cards_df.apply(get_segment, axis=1)
    
    print(f"\n📊 Сегментация клиентов:")
    for segment in cards_df['segment'].unique():
        subset = cards_df[cards_df['segment'] == segment]
        count = len(subset)
        retention = 100 * subset['active_day30'].mean()
        avg_txn = subset['txn_week1'].mean()
        print(f"   {segment}")
        print(f"      Карт: {count}, Удержание: {retention:.1f}%, Средн. трансакций/неделю: {avg_txn:.1f}")
    
    return cards_df


def create_report(df, cards_df):
    """
    Создаёт HTML отчёт с визуализацией (если доступна).
    """
    if not PLOT_AVAILABLE:
        print("\n⚠️  Визуализация недоступна (установите matplotlib и seaborn)")
        return
    
    print("\n📊 Создаю визуализацию...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Узум Банк: Диагностика активности дебетовых карт', fontsize=16, fontweight='bold')
    
    # График 1: Удержание по дням активации
    df_active = df[df['first_transaction_date'].notna()].copy()
    df_active['days_to_activation'] = (df_active['first_transaction_date'] - df_active['issue_date']).dt.days
    retention_by_day = df_active.groupby('days_to_activation')['active_day30'].mean() * 100
    
    axes[0, 0].bar(retention_by_day.index[:14], retention_by_day.values[:14], color='steelblue')
    axes[0, 0].set_xlabel('Дни до активации')
    axes[0, 0].set_ylabel('Удержание на день 30 (%)')
    axes[0, 0].set_title('Удержание по скорости активации')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # График 2: Удержание по категориям
    retention_by_cat = cards_df.groupby('categories')['active_day30'].mean() * 100
    axes[0, 1].plot(retention_by_cat.index, retention_by_cat.values, marker='o', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Число категорий')
    axes[0, 1].set_ylabel('Удержание (%)')
    axes[0, 1].set_title('Удержание по диверсификации')
    axes[0, 1].grid(alpha=0.3)
    
    # График 3: Распределение по сегментам
    segment_counts = cards_df['segment'].value_counts()
    axes[1, 0].barh(range(len(segment_counts)), segment_counts.values, color='coral')
    axes[1, 0].set_yticks(range(len(segment_counts)))
    axes[1, 0].set_yticklabels(segment_counts.index)
    axes[1, 0].set_xlabel('Количество карт')
    axes[1, 0].set_title('Распределение по сегментам')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # График 4: Удержание по сегментам
    segment_retention = cards_df.groupby('segment')['active_day30'].mean() * 100
    colors = ['red' if x < 40 else 'orange' if x < 70 else 'green' for x in segment_retention.values]
    axes[1, 1].barh(range(len(segment_retention)), segment_retention.values, color=colors)
    axes[1, 1].set_yticks(range(len(segment_retention)))
    axes[1, 1].set_yticklabels(segment_retention.index)
    axes[1, 1].set_xlabel('Удержание (%)')
    axes[1, 1].set_title('Удержание по сегментам')
    axes[1, 1].set_xlim([0, 100])
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/diagnostics.png', dpi=150, bbox_inches='tight')
    print("💾 График сохранён в results/diagnostics.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Диагностика данных карт Узум Банка')
    parser.add_argument('--data-path', type=str, default=None, help='Путь к CSV файлу с данными')
    args = parser.parse_args()
    
    print("\n🏦 УЗУМ БАНК: Диагностика активности дебетовых карт")
    print("="*70)
    
    # Загружаем данные
    df = load_data(args.data_path)
    print(f"\n✅ Загружено: {df['card_id'].nunique()} уникальных карт")
    print(f"✅ Транзакций: {len(df)}")
    
    # Анализ
    df_active, retention_by_activation = analyze_activation(df)
    card_data = analyze_categories(df)
    analyze_channels(df)
    cards_df = identify_segments(df)
    
    # Визуализация
    if PLOT_AVAILABLE:
        create_report(df, cards_df)
    
    # Сохраняем результаты
    Path('results').mkdir(exist_ok=True)
    
    # JSON с выводами
    diagnostics = {
        'total_cards': int(df['card_id'].nunique()),
        'activated_cards': int(df[df['first_transaction_date'].notna()]['card_id'].nunique()),
        'activation_rate_pct': float(100 * df[df['first_transaction_date'].notna()]['card_id'].nunique() / df['card_id'].nunique()),
        'avg_days_to_activation': float(df_active['days_to_activation'].mean()),
        'high_risk_segment': '⚡ Быстрый старт, быстрый отток',
        'high_risk_count': int(len(cards_df[cards_df['segment'] == '⚡ Быстрый старт, быстрый отток'])),
        'recommendation': 'Фокус на сегмент "Быстрый старт, быстрый отток" и "Один магазин"'
    }
    
    with open('results/diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены в results/")
    print(f"\n✅ Диагностика завершена!")


if __name__ == '__main__':
    main()
