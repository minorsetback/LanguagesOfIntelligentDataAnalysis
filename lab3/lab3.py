import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Налаштування вигляду графіків у seaborn
sns.set_context(
    "notebook",
    font_scale=1.5,
    rc={
        "figure.figsize": (12, 9),
        "axes.titlesize": 18
    }
)

# Зчитування даних з CSV-файлу в об'єкт pandas DataFrame
df = pd.read_csv('mlbootcamp_train_Soroka.csv')
for c in df.columns:
    n = df[c].nunique()
    print(c)
    if n <= 3:
        print(n, sorted(df[c].value_counts().to_dict().items()))
    else:
        print(n)
    print(10 * '-')
    
correlation_matrix = df.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="crest")
plt.title("Correlation Matrix (Pearson)")
plt.show()

# Знаходження двох ознак з найбільшою кореляцією (за Пірсоном)
max_corr = correlation_matrix.unstack().sort_values(ascending=False)
print("Найбільш корелюючі ознаки:")
print(max_corr[(max_corr < 1)][:2])
df_uniques = pd.melt(frame=df, value_vars=['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'])
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.catplot(x='variable', y='count', hue='value', data=df_uniques, kind='bar', aspect=3)
plt.title('Count of Each Categorical Variable')
plt.show()

# Розділення елементів вибірки за значеннями цільової змінної
df_uniques = pd.melt(frame=df, value_vars=['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'], id_vars=['cardio'])
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 'value', 'cardio'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.catplot(x='variable', y='count', hue='value', col='cardio', data=df_uniques, kind='bar', aspect=1.5)
plt.suptitle('Count of Each Categorical Variable by Cardio')
plt.show()
# Перетворення DataFrame в "Long Format"
longformat = pd.melt(frame=df, value_vars='height', id_vars='gender')

# Побудова violinplot
plt.figure(figsize=(12, 8))
sns.violinplot(data=longformat, x='variable', y='value', hue='gender', scale='count')
plt.title('Distribution of Height by Gender')
plt.xlabel('Height')
plt.ylabel('Value')
plt.show()

# Побудова kdeplot для розподілу росту для чоловіків та жінок
plt.figure(figsize=(12, 8))
sns.kdeplot(df[df['gender'] == 1]['height'], label='Female', shade=True)
sns.kdeplot(df[df['gender'] == 2]['height'], label='Male', shade=True)
plt.title('Distribution of Height by Gender (KDE Plot)')
plt.xlabel('Height')
plt.ylabel('Density')
plt.legend()
plt.show()
# Рангова кореляція Спірмена та побудова heatmap
spearman_corr = df.corr(method='spearman')
plt.figure(figsize=(12, 9))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="crest")
plt.title("Spearman Correlation Matrix")
plt.show()

# Знаходження ознак з найбільшою ранговою кореляцією (Спірмен)
max_spearman_corr = spearman_corr.unstack().sort_values(ascending=False)
print("Найбільш корелюючі ознаки за Спірменом:")
print(max_spearman_corr[(max_spearman_corr < 1)][:2])
df['age_years'] = (df['age'] // 365.25).astype(int)

plt.figure(figsize=(12, 8))
sns.countplot(x='age_years', hue='cardio', data=df)
plt.title('Count of People by Age (Years) and Cardio Status')
plt.xlabel('Age (Years)')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
