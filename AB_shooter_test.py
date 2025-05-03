import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, t
from scipy.stats import zscore, ttest_ind
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu

# Шаг 1: Загрузка данных
money = pd.read_csv('Money.csv')
cash = pd.read_csv('Cash.csv')
cheaters = pd.read_csv('Cheaters.csv')
platforms = pd.read_csv('Platforms.csv')
ab = pd.read_csv('ABgroup.csv')

print("Данные успешно загружены.")

# Шаг 2: Удаление известных читеров
cheater_ids = set(cheaters[cheaters['cheaters'] == 1]['user_id'])
print(f"Читеров найдено: {len(cheater_ids)}")

# Удаляем строки с читерами
money_cleaned = money[~money['user_id'].isin(cheater_ids)]
cash_cleaned = cash[~cash['user_id'].isin(cheater_ids)]
ab_cleaned = ab[~ab['user_id'].isin(cheater_ids)]
platforms_cleaned = platforms[~platforms['user_id'].isin(cheater_ids)]

# Выводим статистику удаления
for original, cleaned, name in zip([money, cash, ab, platforms],
                                   [money_cleaned, cash_cleaned, ab_cleaned, platforms_cleaned],
                                   ['Money', 'Cash', 'AB Group', 'Platforms']):
    removed = len(original) - len(cleaned)
    print(f"{name}: удалено {removed} строк из {len(original)} ({(removed / len(original)) * 100:.2f}%)")

# Шаг 3: Поиск потенциальных читеров
# Объединяем данные money и cash для пользователей

money_sum = money_cleaned.groupby(['user_id', 'date']).agg({'money': 'sum'}).reset_index()
cash_sum = cash_cleaned.groupby(['user_id', 'date']).agg({'cash': 'sum'}).reset_index()

df_cash_money = pd.merge(money_sum, cash_sum, how='inner', on='user_id')

# Вычисляем потенциальных читеров 

df_non_payers = df_cash_money[df_cash_money['money'] == 0]
df_hidden_cheaters = df_non_payers[df_non_payers['cash'] >= 50000].reset_index()

all_cheaters = set(cheater_ids).union(set(df_hidden_cheaters['user_id']))

# Количество уникальных читеров
print(f"Всего уникальных подозрительных пользователей: {len(all_cheaters)}")

# Удаляем подозрительных пользователей

money_upd = money_cleaned[~money_cleaned['user_id'].isin(all_cheaters)]
cash_upd = cash_cleaned[~cash_cleaned['user_id'].isin(all_cheaters)]
ab_upd = ab_cleaned[~ab_cleaned['user_id'].isin(all_cheaters)]
platforms_upd = platforms_cleaned[~platforms_cleaned['user_id'].isin(all_cheaters)]

# Собираем уникальные значения для групп и платформ, для построения графиков по дням

ab_unique = ab_upd.drop_duplicates(subset = 'user_id')
platforms_unique = platforms_upd.drop_duplicates(subset = 'user_id')

ab_platform_df = pd.merge(ab_unique, platforms_unique, on='user_id', how='left')

money_visual = pd.merge(money_sum, ab_platform_df, on='user_id', how='left')
cash_visual = pd.merge(cash_sum, ab_platform_df, on='user_id', how='left')

# Шаг 5: Расчет метрик

def calculate_metrics(money_visual, cash_visual):
    # Объединяем данные money_visual и cash_visual
    merged_df = pd.merge(money_visual, cash_visual, on=['user_id', 'date', 'group', 'platform'], how='outer')
    
    # Группировка данных по user_id для сохранения информации о пользователях
    metrics = merged_df.groupby(['user_id', 'date', 'group', 'platform']).agg({
        'money': 'sum',
        'cash': 'sum'
    }).reset_index()

    # Добавляем количество пользователей и платящих пользователей
    metrics['users_count'] = metrics.groupby(['date', 'group', 'platform'])['user_id'].transform('count')
    paying_users = metrics[metrics['money'] > 0]
    metrics['paying_users_count'] = paying_users.groupby(['date', 'group', 'platform'])['user_id'].transform('count')

    # Рассчитываем метрики
    metrics['ARPU'] = metrics['money'] / metrics['users_count']
    metrics['ARPPU'] = metrics['money'] / metrics['paying_users_count']
    metrics['avg_cash'] = metrics['cash'] / metrics['users_count']
    metrics['ARPPU'] = metrics['ARPPU'].fillna(0)

    return metrics

visual_metrics = calculate_metrics(money_visual, cash_visual)

# Группировка данных по user_id
visual_metrics_grouped = visual_metrics.groupby(['platform', 'group', 'user_id']).agg({
    'money': 'sum',
    'cash': 'sum'
}).reset_index()

# Шаг 6: Анализ статистической значимости

def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data.dropna())
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    mean = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean - h, mean + h, mean

def analyze_metric(df, metric, metric_name):
    results = []
    for platform in df['platform'].unique():
        # Получаем данные для групп test и control
        test_data = df[(df['platform'] == platform) & (df['group'] == 'test')][metric]
        control_data = df[(df['platform'] == platform) & (df['group'] == 'control')][metric]

        # Вычисляем сумму метрики
        test_sum = test_data.sum()
        control_sum = control_data.sum()

        # Вычисляем доверительные интервалы
        test_ci_low, test_ci_high, _ = mean_confidence_interval(test_data)
        control_ci_low, control_ci_high, _ = mean_confidence_interval(control_data)

        # Проверяем пересечение доверительных интервалов
        significant = not (test_ci_high >= control_ci_low and control_ci_high >= test_ci_low)
        significance_marker = '✔' if significant else '-'

        # Добавляем результаты в список
        results.append({
            'platform': platform,
            'test_sum': round(test_sum, 2),
            'control_sum': round(control_sum, 2),
            f'{metric_name}_sum_diff': round(test_sum - control_sum, 2),
            'test_CI_low': round(test_ci_low, 2),
            'test_CI_high': round(test_ci_high, 2),
            'control_CI_low': round(control_ci_low, 2),
            'control_CI_high': round(control_ci_high, 2),
            'significant': significance_marker  # Галочка или минус
        })
    return pd.DataFrame(results)

arpu_results = analyze_metric(visual_metrics, 'ARPU', 'ARPU')
arppu_results = analyze_metric(visual_metrics, 'ARPPU', 'ARPPU')
avg_cash_results = analyze_metric(visual_metrics, 'avg_cash', 'avg_cash')

print("Результаты анализа метрик:")
print(arpu_results)
print(arppu_results)
print(avg_cash_results)
