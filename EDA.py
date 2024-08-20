import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

#Daten einlesen
data=pd.read_csv("sickness_table.csv")

# Grundlegende Informationen und Statistiken anzeigen
print("Grundlegende Informationen:")
print(data.info())

print("\nFehlende Werte pro Spalte:")
print(data.isnull().sum())

#Anzahl der Zeilen und Spalten (Datenqualität)
print(f"Zeilen und Spalten: {data.shape}")

#Eindeutige Werte im Datensatz (Datenqualität)
print("Eindeutige Werte im Datensatz:")
print(data.nunique())

print("\nBeschreibende Statistiken:")
print(data.describe())

# Umwandeln der Datumsspalte in ein geeignetes Datumsformat
data['date'] = pd.to_datetime(data['date'])

#Indexspalte löschen
data = data.drop(columns=["Unnamed: 0"])
data = data.drop(columns=["n_sby"])

# zusätzliche Features
data['day_of_week'] = data['date'].dt.dayofweek
data['week'] = data['date'].dt.isocalendar().week
data['month'] = data['date'].dt.month
data['season'] = data['month'] % 12 // 3 + 1  # 1: Winter, 2: Frühling, 3: Sommer, 4: Herbst
data['year'] = data['date'].dt.year

#Feiertage
de_holidays = holidays.Germany()
#Binäres Feature, ob Feiertag oder nicht
data['holiday'] = data['date'].apply(lambda x: 1 if x in de_holidays else 0)

# Line Plot: Anzahl der kranken Einsatzfahrenden im Zeitverlauf
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['n_sick'], label='Kranke Einsatzfahrer')
plt.xlabel('Datum')
plt.ylabel('Anzahl der kranken Einsatzfahrenden')
plt.title('Anzahl der kranken Einsatzfahrenden im Zeitverlauf')
plt.legend()
plt.show()

# Line Plot: Anzahl der Notrufe im Zeitverlauf
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['calls'], label='Notrufe', color='orange')
plt.xlabel('Datum')
plt.ylabel('Anzahl der Notrufe')
plt.title('Anzahl der Notrufe im Zeitverlauf')
plt.legend()
plt.show()

# Line Plot: Anzahl der benötigten Ersatzfahrer im Zeitverlauf
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['sby_need'], label='benötigte Ersatzfahrer', color='orange')
plt.xlabel('Datum')
plt.ylabel('Anzahl der benötigten Ersatzfahrer')
plt.title('Anzahl der benötigten Ersatzfahrer im Zeitverlauf')
plt.legend()
plt.show()

# Line Plot: Anzahl der Einsatzfahrer im Dienst im Zeitverlauf
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['n_duty'], label='Einsatzfahrer im Dienst', color='orange')
plt.xlabel('Datum')
plt.ylabel('Anzahl der Einsatzfahrer im Dienst')
plt.title('Anzahl der Einsatzfahrer im Dienst im Zeitverlauf')
plt.legend()
plt.show()

# Korrelationen zwischen den numerischen Variablen
correlation_matrix = data.corr()

# Heatmap der Korrelationen
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap der Korrelationen')
plt.show()

# Histogramme der wichtigsten Variablen
variables_to_plot = ['n_sick', 'calls', 'n_duty', 'sby_need']

plt.figure(figsize=(15, 10))
for i, variable in enumerate(variables_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[variable], bins=30, kde=True)
    plt.title(f'Verteilung von {variable}')
plt.tight_layout()
plt.show()

# Boxplot: Anzahl der kranken Einsatzfahrenden pro Monat
plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='n_sick', data=data)
plt.xlabel('Monat')
plt.ylabel('Anzahl der kranken Einsatzfahrenden')
plt.title('Anzahl der kranken Einsatzfahrenden pro Monat')
plt.show()

# Boxplot: Anzahl der kranken Einsatzfahrenden pro Wochentag
plt.figure(figsize=(12, 6))
sns.boxplot(x='day_of_week', y='n_sick', data=data)
plt.xlabel('Wochentag')
plt.ylabel('Anzahl der kranken Einsatzfahrenden')
plt.title('Anzahl der kranken Einsatzfahrenden pro Wochentag')
plt.show()

# Boxplot: Anzahl der kranken Einsatzfahrenden pro Season
plt.figure(figsize=(12, 6))
sns.boxplot(x='season', y='n_sick', data=data)
plt.xlabel('Jahreszeit')
plt.ylabel('Anzahl der kranken Einsatzfahrenden')
plt.title('Anzahl der kranken Einsatzfahrenden pro Jahreszeit')
plt.show()

# Boxplot: Anzahl der Notrufe pro Wochentag
plt.figure(figsize=(12, 6))
sns.boxplot(x='day_of_week', y='calls', data=data)
plt.xlabel('Wochentag')
plt.ylabel('Anzahl der Notrufe')
plt.title('Anzahl der Notrufe pro Wochentag')
plt.show()

# Boxplot: Anzahl der Notrufe pro Monat
plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='calls', data=data)
plt.xlabel('Monat')
plt.ylabel('Anzahl der Notrufe')
plt.title('Anzahl der Notrufe pro Monat')
plt.show()

# Boxplot: Anzahl der Notrufe pro Season
plt.figure(figsize=(12, 6))
sns.boxplot(x='season', y='calls', data=data)
plt.xlabel('Jahreszeit')
plt.ylabel('Anzahl der Notrufe')
plt.title('Anzahl der Notrufe pro Jahreszeit')
plt.show()

# Boxplot: Anzahl der benötigten Einsatzfahrer pro Season
plt.figure(figsize=(12, 6))
sns.boxplot(x='season', y='sby_need', data=data)
plt.xlabel('Jahreszeit')
plt.ylabel('Anzahl der benötigten Einsatzfahrer')
plt.title('Anzahl der benötigten Einsatzfahrer pro Jahreszeit')
plt.show()

# Durchschnittliche Anzahl Krankenstände und Notrufe nach Wochentag
average_sick_by_weekday = data.groupby(data['date'].dt.dayofweek)['n_sick'].mean()
average_calls_by_weekday = data.groupby(data['date'].dt.dayofweek)['calls'].mean()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
average_sick_by_weekday.plot(kind='bar', color='orange')
plt.title('Durchschnittliche Anzahl Krankenstände nach Wochentag')
plt.xlabel('Wochentag')
plt.ylabel('Durchschnittliche Anzahl Krankenstände')

plt.subplot(1, 2, 2)
average_calls_by_weekday.plot(kind='bar', color='skyblue')
plt.title('Durchschnittliche Anzahl Notrufe nach Wochentag')
plt.xlabel('Wochentag')
plt.ylabel('Durchschnittliche Anzahl Notrufe')

plt.tight_layout()
plt.show()

# Durchschnittliche Anzahl Krankenstände und Notrufe nach Monat
average_sick_by_month = data.groupby(data['date'].dt.month)['n_sick'].mean()
average_calls_by_month = data.groupby(data['date'].dt.month)['calls'].mean()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
average_sick_by_month.plot(kind='bar', color='orange')
plt.title('Durchschnittliche Anzahl Krankenstände nach Monat')
plt.xlabel('Monat')
plt.ylabel('Durchschnittliche Anzahl Krankenstände')

plt.subplot(1, 2, 2)
average_calls_by_month.plot(kind='bar', color='skyblue')
plt.title('Durchschnittliche Anzahl Notrufe nach Monat')
plt.xlabel('Monat')
plt.ylabel('Durchschnittliche Anzahl Notrufe')

plt.tight_layout()
plt.show()

# Durchschnittliche Anzahl Krankenstände und Notrufe nach Season
average_sick_by_season = data.groupby(data['season'])['n_sick'].mean()
average_calls_by_season = data.groupby(data['season'])['calls'].mean()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
average_sick_by_season.plot(kind='bar', color='orange')
plt.title('Durchschnittliche Anzahl Krankenstände nach Jahreszeit')
plt.xlabel('Jahreszeit')
plt.ylabel('Durchschnittliche Anzahl Krankenstände')

plt.subplot(1, 2, 2)
average_calls_by_season.plot(kind='bar', color='skyblue')
plt.title('Durchschnittliche Anzahl Notrufe nach Jahreszeit')
plt.xlabel('Jahreszeit')
plt.ylabel('Durchschnittliche Anzahl Notrufe')

plt.tight_layout()
plt.show()

# Durchschnittliche Anzahl Krankenstände und Notrufe nach Jahr
average_sick_by_year = data.groupby(data['year'])['n_sick'].mean()
average_calls_by_year = data.groupby(data['year'])['calls'].mean()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
average_sick_by_year.plot(kind='bar', color='orange')
plt.title('Durchschnittliche Anzahl Krankenstände nach Jahr')
plt.xlabel('Jahr')
plt.ylabel('Durchschnittliche Anzahl Krankenstände')

plt.subplot(1, 2, 2)
average_calls_by_year.plot(kind='bar', color='skyblue')
plt.title('Durchschnittliche Anzahl Notrufe nach Jahr')
plt.xlabel('Jahr')
plt.ylabel('Durchschnittliche Anzahl Notrufe')

plt.tight_layout()
plt.show()

# Durchschnittliche Anzahl Krankenstände und Notrufe nach Feiertage
average_sick_by_holiday = data.groupby(data['holiday'])['n_sick'].mean()
average_calls_by_holiday = data.groupby(data['holiday'])['calls'].mean()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
average_sick_by_holiday.plot(kind='bar', color='orange')
plt.title('Durchschnittliche Anzahl Krankenstände nach Feiertag')
plt.xlabel('Feiertag')
plt.ylabel('Durchschnittliche Anzahl Krankenstände')

plt.subplot(1, 2, 2)
average_calls_by_holiday.plot(kind='bar', color='skyblue')
plt.title('Durchschnittliche Anzahl Notrufe nach Feiertag')
plt.xlabel('Feiertag')
plt.ylabel('Durchschnittliche Anzahl Notrufe')

plt.tight_layout()
plt.show()