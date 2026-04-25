# Uzum Bank: Dormancy Prediction and Trigger Engine

Проект для раннего выявления карт с риском "засыпания" и запуска персональных бонусных триггеров для роста транзакционной активности.

## 1. Business Problem

Часть новых карт быстро теряет активность. Задача проекта:

1. Предсказать вероятность засыпания на раннем этапе жизни карты.
2. Назначить подходящий триггер (тип, канал, день отправки).
3. Измерить фактический uplift через контролируемый эксперимент.

## 2. What Is Implemented

1. Единая загрузка и подготовка данных: `data_loader.py`.
2. EDA и поведенческая сегментация: `analysis.py`.
3. ML-модель риска засыпания: `model.py`.
4. Персональные бонусные рекомендации: `bonus_logic.py`.
5. Интерактивный демо-дашборд: `app.py`.

## 3. Data and Target

Источник: `data/uzum_hackathon_dataset.csv` (помесячные агрегаты по карте и категории).

Ключевые поля:

- `card_id`, `kiosk_name`
- `card_creation_date`, `month`
- `category`, `cnt`, `amt`
- `is_active`

Целевая постановка (4.3):

- Observation window: дни 0-14 от выпуска карты.
- Prediction target: заснет ли карта в следующие 30 дней.
- Label: `dormant_30d = 1`, если в днях 15-44 нет транзакций.

## 4. Features (aligned with spec 4.3)

1. Дата/время активации:
- `creation_dow`, `creation_is_weekend`, `days_to_first_txn`, `activated_week1`

2. Активность первой недели:
- `cnt_week1`, `cnt_early`

3. Разнообразие MCC:
- `n_cats_week1`, `n_cats_early`

4. Канал первой транзакции (one-hot):
- `first_txn_online`, `first_txn_offline`, `first_txn_transfer`, `first_txn_cash`, `first_txn_other`

## 5. Model and Metrics

Алгоритм: `DecisionTreeClassifier(max_depth=6, class_weight='balanced')`.

Метрики на test:

- AUC-ROC: 0.7359
- Precision: 0.6935
- Recall: 0.9281
- F1: 0.7938
- Threshold: 0.2625

Порог выбирается по Precision/Recall trade-off с приоритетом Recall (FN дороже FP в задаче удержания).

Артефакты:

- `results/model_metrics.json`
- `results/scored_cards.csv`

## 6. Trigger Logic

Для карт с высоким риском формируется рекомендация:

1. Тип триггера (push/sms/in-app/email).
2. Сообщение (персонализировано по топ-категории).
3. День отправки (в зависимости от риска).

Выгрузки:

- `results/bonus_candidates.csv`
- `results/bonus_candidates.json`

## 7. Measurability and Experiment Design

A/B-подход реализуем в текущей архитектуре:

1. Control: без бонусного триггера.
2. Treatment: триггер для CRITICAL/HIGH риска.

Primary KPI:

- uplift среднего числа транзакций на карту через 30 дней.

Secondary KPI:

- activation rate 7d,
- 30d retention,
- cost per reactivated card,
- incremental interchange.

## 8. Unit Economics (base scenario)

- Bonus cost: 200 UZS
- Avg transaction amount: 50,000 UZS
- Uplift: +3 transactions/card/month
- Interchange (1.5%): 750 UZS/month

Estimated monthly ROI:

`ROI = 750 / 200 = 3.75x`

## 9. Repository Structure

```text
uzum-bank/
├── README.md
├── requirements.txt
├── data_loader.py
├── analysis.py
├── model.py
├── bonus_logic.py
├── app.py
├── data/
│   └── uzum_hackathon_dataset.csv
└── results/
	├── diagnostics.json
	├── diagnostics.png
	├── model_metrics.json
	├── scored_cards.csv
	├── bonus_candidates.csv
	└── bonus_candidates.json
```

## 10. Run Locally

```bash
pip install -r requirements.txt

python3 analysis.py
python3 model.py
python3 bonus_logic.py
streamlit run app.py
```

## 11. Dashboard Pages

1. Overview
2. Model quality
3. Card scoring
4. Bonus recommendations
5. A/B and economics

## 12. Current Data-Backed Findings

- Total cards: 17,467
- Ever transacted: 10,764 (61.6%)
- Never transacted segment: 6,703 cards
- Dormancy rate in labeled set: 61.7%

Эти выводы получены из датасета и артефактов в `results/`, а не из интуитивных предположений.
