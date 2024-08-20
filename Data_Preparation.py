"""Hier werden die verfügbaren Rohdaten für das Modelltraining vorbereitet und
in einer CSV-Datei gespeichert"""

import pandas as pd
import holidays

#Daten laden
sickness_data = pd.read_csv('sickness_table.csv')

#Datum konvertieren
sickness_data['date'] = pd.to_datetime(sickness_data['date'])

#Indexspalte löschen
sickness_data = sickness_data.drop(columns=["Unnamed: 0"])
sickness_data = sickness_data.drop(columns=["n_sby"])

# zusätzliche Features
sickness_data['day_of_week'] = sickness_data['date'].dt.dayofweek
sickness_data['week'] = sickness_data['date'].dt.isocalendar().week
sickness_data['month'] = sickness_data['date'].dt.month
sickness_data['season'] = sickness_data['month'] % 12 // 3 + 1  # 1: Winter, 2: Frühling, 3: Sommer, 4: Herbst
sickness_data['year'] = sickness_data['date'].dt.year
#Feiertage
de_holidays = holidays.Germany()
#Binäres Feature, ob Feiertag oder nicht
sickness_data['holiday'] = sickness_data['date'].apply(lambda x: 1 if x in de_holidays else 0)
print(sickness_data.head(5))

#Bereinigte und vorverarbeitete Daten für das Modell exportieren
sickness_data.to_csv('sickness_modeling.csv', index=False)