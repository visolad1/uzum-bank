#!/usr/bin/env python3
"""
Декишн три модель для предсказания оттока клиентов.

Модель анализирует поведение в первые 7-14 дней и предсказывает
вероятность того, что клиент перестанет пользоваться картой в следующие 30 дней.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score
)
import json
from pathlib import Path
import argparse

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


def load_data(data_path=None):
    """Загружает данные для обучения модели."""
    if data_path and Path(data_path).exists():
        print(f"📂 Загружаю данные из {data_path}...")
        df = pd.read_csv(data_path)
    else:
        # Использую sample_data если он существует
        if Path('data/sample_data.csv').exists():
            print(f"📂 Загружаю данные из data/sample_data.csv...")
            df = pd.read_csv('data/sample_data.csv')
        else:
            print("❌ Данные не найдены!")
            return None
    
    # Конвертируем даты
    for col in ['issue_date', 'first_transaction_date', 'transaction_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df


def extract_features(df):
    """
    Извлекает признаки для модели по первым 7-14 дням жизни карты.
    
    Признаки:
    - days_to_activation: дни до первой транзакции
    - txn_count_week1: число транзакций в первую неделю
    - category_count_week1: число уникальных категорий в неделю
    - channel_count_week1: число разных каналов в неделю
    - has_cash_withdrawal: был ли снят наличка
    - is_transfer_only: только переводы
    - pos_ratio: доля POS транзакций
    - online_ratio: доля Online
    - inapp_ratio: доля In-app
    """
    
    df_active = df[df['first_transaction_date'].notna()].copy()
    
    features_list = []
    
    for card_id in df_active['card_id'].unique():
        card_df = df_active[df_active['card_id'] == card_id].sort_values('transaction_date')
        
        # Базовая информация
        issue_date = card_df['issue_date'].iloc[0]
        first_txn_date = card_df['first_transaction_date'].iloc[0]
        days_to_activation = (first_txn_date - issue_date).days
        
        # Целевая переменная (активна ли на день 30)
        target = card_df['active_day30'].iloc[0]
        
        # Конец первой недели
        week1_end = first_txn_date + timedelta(days=7)
        week1_txns = card_df[card_df['transaction_date'] <= week1_end]
        
        txn_count_week1 = len(week1_txns)
        category_count_week1 = week1_txns['mcc_category'].nunique()
        channel_count_week1 = week1_txns['channel'].nunique()
        
        # Распределение по каналам
        total_week1 = len(week1_txns)
        channel_counts = week1_txns['channel'].value_counts()
        pos_ratio = channel_counts.get('POS', 0) / total_week1 if total_week1 > 0 else 0
        online_ratio = channel_counts.get('Online', 0) / total_week1 if total_week1 > 0 else 0
        inapp_ratio = channel_counts.get('In-app', 0) / total_week1 if total_week1 > 0 else 0
        
        # Специальное поведение
        has_cash_withdrawal = card_df['is_cash_withdrawal'].iloc[0]
        is_transfer_only = card_df['is_transfer_only'].iloc[0]
        
        features_list.append({
            'card_id': card_id,
            'days_to_activation': days_to_activation,
            'txn_count_week1': txn_count_week1,
            'category_count_week1': category_count_week1,
            'channel_count_week1': channel_count_week1,
            'pos_ratio': pos_ratio,
            'online_ratio': online_ratio,
            'inapp_ratio': inapp_ratio,
            'has_cash_withdrawal': has_cash_withdrawal,
            'is_transfer_only': is_transfer_only,
            'churn_day30': 1 - target  # 1 = ушёл, 0 = остался
        })
    
    return pd.DataFrame(features_list)


def train_model(features_df):
    """
    Обучает Decision Tree модель.
    """
    print("\n" + "="*70)
    print("🤖 ОБУЧЕНИЕ МОДЕЛИ")
    print("="*70)
    
    X = features_df[[
        'days_to_activation',
        'txn_count_week1',
        'category_count_week1',
        'channel_count_week1',
        'pos_ratio',
        'online_ratio',
        'inapp_ratio',
        'has_cash_withdrawal',
        'is_transfer_only'
    ]]
    
    y = features_df['churn_day30']
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Данные для обучения:")
    print(f"   Train: {len(X_train)} карт")
    print(f"   Test: {len(X_test)} карт")
    print(f"   Доля оттока в train: {y_train.mean():.1%}")
    print(f"   Доля оттока в test: {y_test.mean():.1%}")
    
    # Обучаем Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=5,  # Не слишком глубокая для интерпретируемости
        min_samples_split=20,  # Минимум 20 образцов для разделения
        min_samples_leaf=10,  # Минимум 10 в листе
        random_state=42
    )
    
    dt.fit(X_train, y_train)
    
    # Предсказания
    y_pred_train = dt.predict(X_train)
    y_pred_proba_train = dt.predict_proba(X_train)[:, 1]
    
    y_pred_test = dt.predict(X_test)
    y_pred_proba_test = dt.predict_proba(X_test)[:, 1]
    
    # Оценка
    print(f"\n📈 КАЧЕСТВО МОДЕЛИ")
    print(f"\nНа обучающей выборке (train):")
    print(f"   Accuracy: {(y_pred_train == y_train).mean():.3f}")
    print(f"   AUC-ROC: {roc_auc_score(y_train, y_pred_proba_train):.3f}")
    print(f"   F1-score: {f1_score(y_train, y_pred_train):.3f}")
    
    print(f"\nНа тестовой выборке (test):")
    print(f"   Accuracy: {(y_pred_test == y_test).mean():.3f}")
    print(f"   AUC-ROC: {roc_auc_score(y_test, y_pred_proba_test):.3f}")
    print(f"   F1-score: {f1_score(y_test, y_pred_test):.3f}")
    
    print(f"\n🎯 Матрица ошибок (test):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"   True Negatives (правильно определены остались): {cm[0, 0]}")
    print(f"   False Positives (ложно определены как уходящие): {cm[0, 1]}")
    print(f"   False Negatives (пропущены уходящие): {cm[1, 0]}")
    print(f"   True Positives (правильно определены уходящие): {cm[1, 1]}")
    
    print(f"\nПточность (Precision): {cm[1, 1] / (cm[1, 1] + cm[0, 1]):.3f}")
    print(f"Полнота (Recall): {cm[1, 1] / (cm[1, 1] + cm[1, 0]):.3f}")
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n⭐ Важность признаков:")
    for idx, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    return dt, X, y, X_test, y_test, y_pred_proba_test, feature_importance


def extract_rules(dt, feature_names, threshold=0.5):
    """
    Извлекает интерпретируемые правила из Decision Tree.
    """
    print(f"\n" + "="*70)
    print(f"📋 ПРАВИЛА МОДЕЛИ (порог: {threshold})")
    print("="*70)
    
    # Пример простых правил на основе важности
    rules = {
        'high_risk': {
            'description': 'Вероятность оттока > 50%',
            'conditions': [
                'days_to_activation > 3 дней',
                'txn_count_week1 < 2',
                'category_count_week1 < 2'
            ]
        },
        'medium_risk': {
            'description': 'Вероятность оттока 30-50%',
            'conditions': [
                'days_to_activation <= 3 дней',
                'txn_count_week1 >= 2',
                'category_count_week1 < 3'
            ]
        },
        'low_risk': {
            'description': 'Вероятность оттока < 30%',
            'conditions': [
                'days_to_activation <= 3 дней',
                'txn_count_week1 >= 3',
                'category_count_week1 >= 2'
            ]
        }
    }
    
    for risk_level, rule in rules.items():
        print(f"\n🔴 {rule['description']}:")
        for condition in rule['conditions']:
            print(f"   • {condition}")
    
    return rules


def save_results(dt, features_df, X_test, y_test, y_pred_proba_test, rules, feature_importance):
    """
    Сохраняет результаты модели и правила.
    """
    Path('results').mkdir(exist_ok=True)
    
    # Сохраняем правила
    rules_json = {
        'description': 'Правила для предсказания оттока клиентов',
        'rules': rules,
        'feature_importance': dict(zip(
            feature_importance['feature'],
            feature_importance['importance'].astype(float)
        ))
    }
    
    with open('results/model_rules.json', 'w') as f:
        json.dump(rules_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Правила сохранены в results/model_rules.json")
    
    # Сохраняем метрики
    metrics = {
        'auc_roc': float(roc_auc_score(y_test, y_pred_proba_test)),
        'accuracy': float((y_test == (y_pred_proba_test > 0.5)).mean()),
        'total_cards': len(features_df),
        'churn_rate': float(features_df['churn_day30'].mean())
    }
    
    with open('results/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Метрики сохранены в results/model_metrics.json")


def main():
    parser = argparse.ArgumentParser(description='Обучение модели предсказания оттока')
    parser.add_argument('--data-path', type=str, default=None, help='Путь к CSV файлу')
    args = parser.parse_args()
    
    print("\n🤖 УЗУМ БАНК: Модель предсказания оттока клиентов")
    print("="*70)
    
    # Загружаем данные
    df = load_data(args.data_path)
    if df is None:
        return
    
    # Извлекаем признаки
    print("\n🔧 Извлечение признаков...")
    features_df = extract_features(df)
    print(f"✅ Извлечено {len(features_df)} карт с признаками")
    
    # Обучаем модель
    dt, X, y, X_test, y_test, y_pred_proba_test, feature_importance = train_model(features_df)
    
    # Извлекаем правила
    rules = extract_rules(dt, X.columns)
    
    # Сохраняем результаты
    save_results(dt, features_df, X_test, y_test, y_pred_proba_test, rules, feature_importance)
    
    print(f"\n✅ Модель обучена и сохранена!")


if __name__ == '__main__':
    main()
