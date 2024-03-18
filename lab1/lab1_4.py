import pandas as pd
import numpy as np

df = pd.read_csv('test.csv')

print("Перші 5 записів:")
print(df.head())

print("\nОстанні 5 записів:")
print(df.tail())

def extract_average(row):
    values = eval(row) 
    averages = [np.mean(sublist) for sublist in values]  
    return np.mean(averages)  

last_column_name = df.columns[-1]
df[last_column_name] = df[last_column_name].apply(extract_average)

last_column_stats = df[last_column_name].describe()

mode = df[last_column_name].mode().iloc[0]

median = df[last_column_name].median()

max_value = df[last_column_name].max()
min_value = df[last_column_name].min()

print("\nСтатистичні характеристики для останнього стовпця:")
print(last_column_stats)

print("\nМода для останнього стовпця:")
print(mode)

print("\nМедіана для останнього стовпця:")
print(median)

print("\nМаксимальне значення для останнього стовпця:")
print(max_value)

print("\nМінімальне значення для останнього стовпця:")
print(min_value)
