import pandas as pd
from pandas_profiling import ProfileReport

# Зчитування даних з csv-файлу
df = pd.read_csv('mlbootcamp_train_Soroka.csv', sep=';', index_col='id')

# Виведення перших 10 записів
print(df.head(10))
print(df.describe())

# Виконання профілювання датасету
profile = ProfileReport(df)
profile.to_file(output_file="AAA_data_profiling.html")

#Питання 1: Скільки чоловіків і жінок представлено в наборі даних?
gender_counts = df['gender'].value_counts()
print(gender_counts)

#Питання 2: Хто в середньому рідше вказує, що вживає алкоголь - чоловіки чи жінки?
average_alcohol = df.groupby('gender')['alco'].mean()
print(average_alcohol)

#Питання 3: У скільки разів відсоток курців серед чоловіків більше, ніж відсоток курців серед жінок?
male_smokers = df[df['gender'] == 2]['smoke'].mean()
female_smokers = df[df['gender'] == 1]['smoke'].mean()

ratio = male_smokers / female_smokers
print(round(ratio, 2))

#Питання 4: У кого в середньому тиск вище, у жінок чи чоловіків?
average_pressure = df.groupby('gender')[['ap_hi', 'ap_lo']].mean()
print(average_pressure)

#Питання 5: На скільки місяців відрізняється медіанне значення віку курців і тих, хто не курить?
median_age_smokers = df[df['smoke'] == 1]['age'].median()
median_age_non_smokers = df[df['smoke'] == 0]['age'].median()

months_difference = (median_age_smokers - median_age_non_smokers) * 12
print(months_difference)

#Питання 6: У скільки разів відрізняються частки хворих людей в двох підгрупах чоловіків?
df['age_years'] = round(df['age'] / 365)

group1 = df[(df['gender'] == 2) & (df['age_years'] >= 60) & (df['age_years'] <= 64)
            & (df['ap_hi'] < 120) & (df['cholesterol'] == 1)]

group2 = df[(df['gender'] == 2) & (df['age_years'] >= 60) & (df['age_years'] <= 64)
            & (df['ap_hi'] >= 160) & (df['ap_hi'] < 180) & (df['cholesterol'] == 3)]

ratio_sick_group1 = group1['cardio'].mean()
ratio_sick_group2 = group2['cardio'].mean()

ratio_difference = ratio_sick_group2 / ratio_sick_group1
print(round(ratio_difference, 2))

#Питання 7: Побудова нової ознаки - BMI та вибір вірних тверджень
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

median_bmi = df['bmi'].median()
median_bmi_healthy = df[df['cardio'] == 0]['bmi'].median()
median_bmi_sick = df[df['cardio'] == 1]['bmi'].median()

print("Median BMI:", median_bmi)
print("Median BMI for Healthy individuals:", median_bmi_healthy)
print("Median BMI for Sick individuals:", median_bmi_sick)

#Питання 8: Відфільтрування неточних даних і підрахунок відсотків видалених даних
filtered_data = df[
    (df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))
]

percentage_removed = (1 - len(filtered_data) / len(df)) * 100
print(round(percentage_removed, 2), "% of data was removed.")

#Питання 9: Підрахунок чоловіків і жінок з зайвою вагою
# Розрахунок BMI та визначення категорій
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
overweight_men = df[(df['gender'] == 2) & (df['bmi'] >= 25) & (df['bmi'] < 30)].shape[0]
overweight_women = df[(df['gender'] == 1) & (df['bmi'] >= 25) & (df['bmi'] < 30)].shape[0]

print("Overweight Men:", overweight_men)
print("Overweight Women:", overweight_women)
