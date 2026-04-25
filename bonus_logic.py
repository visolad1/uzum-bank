#!/usr/bin/env python3
"""
Логика для определения, кому отправлять персонализированный бонус.

Алгоритм:
1. Определи риск оттока по модели
2. Проверь, активен ли клиент в последние 7 дней
3. Найди его TOP MCC-категорию
4. Отправь персонализированный бонус
"""

import pandas as pd
import json
from datetime import timedelta
from pathlib import Path
import argparse


def load_data(data_path=None):
    """Загружает данные."""
    if data_path and Path(data_path).exists():
        print(f"📂 Загружаю данные из {data_path}...")
        df = pd.read_csv(data_path)
    else:
        if Path('data/sample_data.csv').exists():
            print(f"📂 Загружаю данные из data/sample_data.csv...")
            df = pd.read_csv('data/sample_data.csv')
        else:
            print("❌ Данные не найдены!")
            return None
    
    for col in ['issue_date', 'first_transaction_date', 'transaction_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df


def calculate_risk_score(df, card_id):
    """
    Вычисляет риск оттока для карты.
    Возвращает (risk_score, risk_level).
    """
    card_df = df[df['card_id'] == card_id]
    
    if card_df['first_transaction_date'].isna().all():
        return None, 'NOT_ACTIVATED'
    
    issue_date = card_df['issue_date'].iloc[0]
    first_txn_date = card_df['first_transaction_date'].iloc[0]
    
    # Основные признаки
    days_to_activation = (first_txn_date - issue_date).days
    
    # Транзакции в первую неделю
    week1_end = first_txn_date + timedelta(days=7)
    week1_txns = card_df[card_df['transaction_date'] <= week1_end]
    txn_count_week1 = len(week1_txns)
    
    # Категории в первую неделю
    category_count_week1 = week1_txns['mcc_category'].nunique()
    
    # Расчёт риска (простая модель)
    risk_score = 0.0
    
    # Медленная активация
    if days_to_activation > 3:
        risk_score += 0.3
    elif days_to_activation > 7:
        risk_score += 0.1
    
    # Мало транзакций
    if txn_count_week1 < 2:
        risk_score += 0.3
    elif txn_count_week1 < 3:
        risk_score += 0.1
    
    # Мало разнообразия
    if category_count_week1 < 2:
        risk_score += 0.2
    elif category_count_week1 < 3:
        risk_score += 0.1
    
    # Классификация
    if risk_score >= 0.6:
        risk_level = 'HIGH'
    elif risk_score >= 0.3:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    return min(risk_score, 1.0), risk_level


def get_top_category(df, card_id):
    """
    Определяет самую часто используемую MCC-категорию.
    """
    card_df = df[df['card_id'] == card_id]
    
    if len(card_df) == 0:
        return None
    
    category_counts = card_df['mcc_category'].value_counts()
    return category_counts.index[0] if len(category_counts) > 0 else None


def is_recently_active(df, card_id, days=7):
    """
    Проверяет, активна ли карта в последние N дней.
    """
    card_df = df[df['card_id'] == card_id]
    
    if len(card_df) == 0:
        return False
    
    last_txn_date = card_df['transaction_date'].max()
    analysis_date = card_df['transaction_date'].max()  # Берём последний день в данных
    
    if pd.isna(last_txn_date) or pd.isna(analysis_date):
        return False
    
    days_since_last_txn = (analysis_date - last_txn_date).days
    return days_since_last_txn <= days


def get_bonus_offer(category):
    """
    Возвращает персонализированное предложение бонуса для категории.
    """
    offers = {
        'Supermarket': {
            'message': '🏪 Кэшбэк 3% в супермаркетах — платите картой!',
            'value_sums': 200
        },
        'Restaurant': {
            'message': '🍽️  Кэшбэк 5% в ресторанах — неземные вкусы ждут!',
            'value_sums': 300
        },
        'Transport': {
            'message': '🚕 Кэшбэк 2% на транспорт — каждая поездка с бонусом!',
            'value_sums': 100
        },
        'Telecom': {
            'message': '📱 Бонус 2% на мобильную связь — всегда на связи!',
            'value_sums': 50
        },
        'Online Shopping': {
            'message': '🛒 Кэшбэк 4% на онлайн-покупки — добавьте вещей в корзину!',
            'value_sums': 200
        },
        'Gas Station': {
            'message': '⛽ Кэшбэк 2% на топливо — каждый литр с вознаграждением!',
            'value_sums': 100
        }
    }
    
    if category in offers:
        return offers[category]
    else:
        return {
            'message': f'💳 Кэшбэк 2% в категории {category}',
            'value_sums': 100
        }


def identify_bonus_candidates(df):
    """
    Определяет клиентов, которым нужно отправить бонус.
    Возвращает список с рекомендациями.
    """
    print("\n" + "="*70)
    print("🎁 ЛОГИКА ОТПРАВКИ БОНУСОВ")
    print("="*70)
    
    candidates = []
    
    for card_id in df['card_id'].unique():
        # Расчитаем риск
        risk_score, risk_level = calculate_risk_score(df, card_id)
        
        if risk_level == 'NOT_ACTIVATED':
            continue
        
        # Проверим активность
        active = is_recently_active(df, card_id, days=7)
        
        if not active:
            continue
        
        # HIGH RISK + ACTIVE = кандидат
        if risk_level == 'HIGH':
            top_category = get_top_category(df, card_id)
            
            if top_category:
                bonus_offer = get_bonus_offer(top_category)
                
                candidates.append({
                    'card_id': card_id,
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'top_category': top_category,
                    'message': bonus_offer['message'],
                    'bonus_value_sums': bonus_offer['value_sums'],
                    'send_bonus': True,
                    'trigger_date': pd.Timestamp.now().strftime('%Y-%m-%d')
                })
    
    candidates_df = pd.DataFrame(candidates)
    
    print(f"\n📊 Результаты:")
    print(f"   Всего карт: {df['card_id'].nunique()}")
    print(f"   Активных: {df[df['first_transaction_date'].notna()]['card_id'].nunique()}")
    print(f"   Кандидатов на бонус (HIGH RISK + ACTIVE): {len(candidates_df)}")
    
    if len(candidates_df) > 0:
        print(f"\n   Процент от активных: {100*len(candidates_df)/df[df['first_transaction_date'].notna()]['card_id'].nunique():.1f}%")
    
    return candidates_df


def export_bonus_list(candidates_df):
    """
    Экспортирует список кандидатов для отправки бонусов.
    """
    Path('results').mkdir(exist_ok=True)
    
    # Сохраняем в JSON для backend интеграции
    bonus_list = candidates_df.to_dict('records')
    
    with open('results/bonus_candidates.json', 'w') as f:
        json.dump(bonus_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Список бонусов сохранён в results/bonus_candidates.json")
    
    # Также CSV для отчётности
    candidates_df.to_csv('results/bonus_candidates.csv', index=False)
    print(f"💾 CSV отчёт сохранён в results/bonus_candidates.csv")
    
    # Примеры сообщений
    print(f"\n📧 Примеры сообщений для отправки:")
    for idx, row in candidates_df.head(3).iterrows():
        print(f"\n   Карта {row['card_id']}:")
        print(f"   → {row['message']}")
        print(f"   → Стоимость: {row['bonus_value_sums']} сум")
    
    if len(candidates_df) > 3:
        print(f"\n   ... и ещё {len(candidates_df) - 3} карт")


def main():
    parser = argparse.ArgumentParser(description='Логика отправки бонусов')
    parser.add_argument('--data-path', type=str, default=None)
    args = parser.parse_args()
    
    print("\n🎁 УЗУМ БАНК: Логика отправки целевых бонусов")
    print("="*70)
    
    df = load_data(args.data_path)
    if df is None:
        return
    
    # Определяем кандидатов
    candidates_df = identify_bonus_candidates(df)
    
    # Экспортируем
    if len(candidates_df) > 0:
        export_bonus_list(candidates_df)
    
    print(f"\n✅ Логика бонусов готова!")


if __name__ == '__main__':
    main()
