#!/usr/bin/env python3
"""
Единая функция загрузки и парсинга данных для всего проекта.
"""

import pandas as pd
from pathlib import Path

# Колонки, которые автоматически конвертируются в datetime
_KNOWN_DATE_COLS = {
    'month',
    'card_creation_date',
    'issue_date',
    'first_transaction_date',
    'transaction_date',
}


def load_data(data_path=None, fallback_path='data/uzum_hackathon_dataset.csv'):
    """
    Динамическая загрузка CSV-данных.

    Параметры
    ---------
    data_path : str | None
        Явный путь к файлу. Если None или файл не существует — используется fallback_path.
    fallback_path : str
        Путь по умолчанию, если data_path не задан или не найден.

    Возвращает
    ----------
    pd.DataFrame | None
        DataFrame с распарсенными датами и колонкой month_of_life (если применимо).
        None — если файл не найден ни по одному из путей.

    Автоматически:
    - Выбирает существующий файл (data_path → fallback_path)
    - Парсит все известные колонки дат (_KNOWN_DATE_COLS), присутствующие в файле
    - Вычисляет month_of_life, если есть колонки month и card_creation_date
    """
    path = _resolve_path(data_path, fallback_path)
    if path is None:
        print(f"Data file not found: {data_path or fallback_path}")
        return None

    print(f"Loading data from {path}...")
    df = pd.read_csv(path)

    # Парсинг дат — только колонки, которые реально есть в файле
    date_cols_present = [c for c in df.columns if c in _KNOWN_DATE_COLS]
    for col in date_cols_present:
        df[col] = pd.to_datetime(df[col])

    # Вычисление month_of_life — только если есть оба поля
    if 'month' in df.columns and 'card_creation_date' in df.columns:
        creation_period = df['card_creation_date'].dt.to_period('M')
        month_period = df['month'].dt.to_period('M')
        df['month_of_life'] = (month_period - creation_period).apply(lambda x: x.n)

    return df


def _resolve_path(data_path, fallback_path):
    """Возвращает первый существующий путь из data_path и fallback_path, или None."""
    if data_path and Path(data_path).exists():
        return Path(data_path)
    if Path(fallback_path).exists():
        return Path(fallback_path)
    return None
