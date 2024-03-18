import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ігноруємо warnings
warnings.filterwarnings("ignore")

# Настройка зовнішнього вигляду графіків у seaborn
sns.set_context(
    "notebook",
    font_scale=1.5,
    rc={
        "figure.figsize": (12, 9),
        "axes.titlesize": 18
    }
)

# Зчитуємо дані з CSV-файлу в об'єкт pandas DataFrame
df = pd.read_csv('Data.csv')

# Подивимося на значення, які приймають ознаки
for c in df.columns:
    n = df[c].nunique()
    print(c)
    if n <= 3:
        print(n, sorted(df[c].value_counts().to_dict().items()))
    else:
        print(n)
    print(10 * '-')

# Статистика за унікальними значеннями ознак

# Скільки кількісних ознак (без id)?
numeric_features = df.select_dtypes(include=[np.number]).columns
print("\nКількісні ознаки (без id):", len(numeric_features) - 1)  # віднімаємо id

# Скільки категоріальних ознак?
categorical_features = df.select_dtypes(include=[np.object]).columns
print("Категоріальні ознаки:", len(categorical_features))

# Кореляційна матриця
correlation_matrix = df.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, cmap="crest", annot=True, fmt=".2f")
plt.title("Кореляційна матриця")
plt.show()

# Візуалізація категоріальних ознак
df_uniques = pd.melt(frame=df, value_vars=['speed', 'rr', 'lin', 'distance', 'timeper', 'cat', 'grpsize', 'calf', 'behav', 'count'])
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 'value'])['value'].count()).sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}).reset_index()
sns.catplot(x='variable', y='count', hue='value', data=df_uniques, kind='bar', aspect=3)
plt.title("Категоріальні ознаки")
plt.show()

# Розподіл росту за speed
longformat = pd.melt(frame=df, value_vars='speed', id_vars='id')
plt.figure(figsize=(10, 6))
sns.violinplot(data=longformat, x='variable', y='value')
plt.title("Розподіл росту")
plt.show()

# Рангова кореляція
spearman_corr = df.corr(method='spearman')
plt.figure(figsize=(12, 9))
sns.heatmap(spearman_corr, cmap="crest", annot=True, fmt=".2f")
plt.title("Рангова кореляція Спірмена")
plt.show()