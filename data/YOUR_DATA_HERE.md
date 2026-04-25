# 📊 Загрузка локального датасета

## Как добавить ваши данные

Этот репозиторий содержит **шаблон для вашего датасета**. Вы предоставляете данные локально, и код их обработает.

## 📁 Структура данных

Ваш CSV файл должен содержать следующие колонки:

| Колонка | Тип | Описание | Пример |
|---------|-----|---------|--------|
| `card_id` | string | Уникальный ID карты | `CARD_001` |
| `issue_date` | datetime | Дата выпуска карты | `2026-04-01` |
| `first_transaction_date` | datetime | Дата первой транзакции | `2026-04-02` |
| `transaction_date` | datetime | Дата каждой транзакции | `2026-04-05` |
| `channel` | string | Канал платежа | `POS`, `Online`, `In-app` |
| `mcc_category` | string | MCC категория | `Supermarket`, `Restaurant`, `Transport`, `Telecom`, `Online Shopping` |
| `is_cash_withdrawal` | int | Снятие наличи (0/1) | `0` или `1` |
| `is_transfer_only` | int | Только переводы (0/1) | `0` или `1` |
| `active_day30` | int | Активна ли на день 30 (0/1) | `1` или `0` |

## 📝 Пример датасета

```csv
card_id,issue_date,first_transaction_date,transaction_date,channel,mcc_category,is_cash_withdrawal,is_transfer_only,active_day30
CARD_001,2026-04-01,2026-04-02,2026-04-02,POS,Supermarket,0,0,1
CARD_001,2026-04-01,2026-04-02,2026-04-05,Online,Online Shopping,0,0,1
CARD_002,2026-04-01,2026-04-08,2026-04-08,POS,Restaurant,0,0,0
CARD_002,2026-04-01,2026-04-08,2026-04-09,In-app,Telecom,0,1,0
CARD_003,2026-04-01,2026-04-03,2026-04-03,Online,Online Shopping,0,0,1
```

## 🚀 Как использовать

### Шаг 1: Подготовьте файл
```bash
# Скопируйте ваш CSV в папку data/
cp /path/to/your_data.csv data/your_data.csv
```

### Шаг 2: Запустите анализ
```bash
# С параметром --data-path
python analysis.py --data-path data/your_data.csv

# Или обновите путь в коде
# data_path = 'data/your_data.csv'  # в analysis.py
python analysis.py
```

### Шаг 3: Обучите модель
```bash
python model.py --data-path data/your_data.csv
```

### Шаг 4: Посмотрите результаты
```bash
# Отчёты генерируются в results/
ls -la results/

# Откройте HTML отчёт
open results/diagnostics.html
```

## 🔍 Проверка данных

### Обязательные условия:
- ✅ **Нет NULL** в ключевых полях (card_id, issue_date, mcc_category)
- ✅ **Даты в формате** `YYYY-MM-DD` или `ISO 8601`
- ✅ **Каналы** только из списка: `POS`, `Online`, `In-app`
- ✅ **MCC категории** согласованы (одно название для одной категории)
- ✅ **Минимум 1000 карт** с данными об активности за 30+ дней

### Проверить качество:
```python
import pandas as pd

# Загрузите свой файл
df = pd.read_csv('data/your_data.csv')

# Проверки
print(f"Карт: {df['card_id'].nunique()}")
print(f"Транзакций: {len(df)}")
print(f"NULL в активности: {df['active_day30'].isna().sum()}")
print(f"Каналы: {df['channel'].unique()}")
print(f"Категории: {df['mcc_category'].unique()}")
```

## 💾 Конфиденциальность

✅ **Все данные остаются локально** на вашем компьютере  
✅ **Модель обучается локально**  
✅ **Ничего не загружается на сервер**  
✅ **Результаты хранятся только в `/results`**  

## 📞 Что дальше?

1. **Диагностика** (`analysis.py`) — найдёт проблемные сегменты
2. **Модель** (`model.py`) — обучит Decision Tree
3. **Логика** (`bonus_logic.py`) — определит, кому отправить бонус
4. **A/B-тест** (`ab_test_design.py`) — спланирует тестирование
5. **Экономика** (`economics.py`) — рассчитает ROI

Все файлы готовы к запуску! 🚀
