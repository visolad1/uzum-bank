# Uzum Bank: Churn Prediction & Targeted Bonuses

## Overview

A system for **early detection of customers likely to stop using their debit card**, and for sending them **personalized bonuses** to restore activity.

## Goals

- Increase **average transactions per card per month** from 4–5 to 9–10
- Improve **Activation Rate** (first transaction within 7 days) from 50% to 75%+
- Grow **share of cards transacting in 3+ MCC categories** from 20% to 45%+

## What We Analyze

- **First purchase date** — speed of card activation
- **Activity in the first month** — transaction count
- **Category diversity** — how many MCC categories the card was used in
- **Cash behavior** — ATM withdrawals
- **Transfers** — transfers only vs. actual purchases

## Solution

### 1. Decision Tree Model

```
IF first_txn_month > 0 AND
   n_categories < 2 AND
   txn_rate < 0.4
THEN risk = HIGH -> send bonus
```

**Why Decision Tree?**
- Transparent — all rules are visible
- Interpretable — easy to explain to business stakeholders
- Fast — scales to any volume
- Simple integration — rules can be hard-coded in the backend

### 2. Personalized Bonuses

```
Risk level?
├── HIGH   -> Send bonus targeting top category
├── MEDIUM -> In-app recommendation
└── LOW    -> Normal mode
```

### 3. A/B Test

- **Control** — no bonus sent
- **Treatment** — personalized bonus sent to HIGH RISK cards
- **Metric** — avg transactions per card per month after 30 days
- **Duration** — 1 month (2 weeks collection + 2 weeks measurement)

## Unit Economics

```
Bonus cost:                200 UZS
Average transaction:    50,000 UZS
Transaction uplift:         +3 ops/month
Interchange revenue (1.5%): 750 UZS/month

ROI = 750 / 200 = 3.75x per month
```

## Project Structure

```
uzum-bank/
├── README.md          # This file
├── analysis.py        # EDA and diagnostics
├── model.py           # Decision Tree model
├── bonus_logic.py     # Bonus dispatch logic
├── data/              # Dataset (git-ignored)
└── results/           # Output files (git-ignored)
```

## Setup & Usage

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run diagnostics

```bash
python analysis.py --data-path data/uzum_hackathon_dataset.csv
```

### Expected dataset columns

```
card_id, kiosk_name, card_creation_date, month,
category, cnt, amt, is_active
```

## Output

### `analysis.py`
- Activation and churn by month of card life
- Category entry vs. maturity patterns
- Region-level activity correlation
- Behavioral segments with is_active=1 rates

### `model.py`
- Trained Decision Tree
- Metrics: Precision, Recall, AUC-ROC
- Rules exported to JSON

### `bonus_logic.py`
- List of cards to target
- Personalized bonus offers

## Key Findings (from EDA)

- **61.6%** of cards made at least one transaction
- **41%** of churned cards went dormant in month 0 (issuance month)
- Cards using **3+ categories** have **100% is_active=1** rate
- Entry categories: Online payments (60%), Domestic transfers (51%)
- Top regions by transactional activity: REGION-001, REGION-069, REGION-047
