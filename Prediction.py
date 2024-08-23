"""Hier wird das Trainierte Modell aufgerufen und anhand den historisierten WErten der letzten
30 Tage eine Vorhersage für sby_need für die kommenden 30 Tage durchgeführt
Da immer genügend Ersatzfahrer verfügbar sein sollen, wird anschließend noch ein Default-Wert von 15
Fahrern gesetzt, falls die Vorhersage=0. Anderenfalls wird die Vorhersage um 10% erhöht"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Dateinamen für Modelle und Scaler
trained_model_filename = 'best_model.pkl'
scaler_filename = 'scaler.pkl'
model_filename = 'best_model_data.pkl'

# Das trainierte Modell, den Scaler und die Features laden
with open(model_filename, 'rb') as file:
    loaded_data = pickle.load(file)

loaded_model = loaded_data['model']
loaded_scaler = loaded_data['scaler']
loaded_features = loaded_data['features']

print("Loaded features for prediction:", loaded_features)
print(loaded_model)
print(loaded_scaler)

# Beispielhafte Daten laden und vorbereiten
data = pd.read_csv('sickness_modeling.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Die letzten 30 Tage der relevanten Daten verwenden
last_30_days = data.tail(30)
print(last_30_days)

# Sicherstellen, dass nur die Features verwendet werden, die sowohl in `loaded_features` als auch in den Daten vorhanden sind
features_to_use = [feature for feature in loaded_features if feature in data.columns]
print("Features used for prediction:", features_to_use)

# DataFrame mit nur den relevanten Features erstellen
last_30_days_features = last_30_days[features_to_use]
print("Features data for prediction:", last_30_days_features)

# Vorhersage für die nächsten 30 Tage basierend auf den letzten 30 Tagen
def predict_next_30_days(last_30_days_features, model, scaler, features):
    # Sicherstellen, dass die Features in der richtigen Reihenfolge sind
    X_last_30_days = last_30_days_features[features]
    print("X-last30days:" , X_last_30_days)

    # Daten skalieren
    X_last_30_days_scaled = scaler.transform(X_last_30_days)
    print("X-last30days_scaled:" ,X_last_30_days_scaled)

    # Vorhersage für die nächsten 30 Tage machen
    predictions = model.predict(X_last_30_days_scaled)
    print(predictions)

    # Zukünftige Daten erzeugen
    future_dates = pd.date_range(start=last_30_days.index.max() + pd.Timedelta(days=1), periods=30, freq='D')
    future_predictions = pd.DataFrame(data=predictions, index=future_dates, columns=['predicted_sby_need'])
    print(future_dates)
    print(future_predictions)
    return future_predictions

# Vorhersage für die nächsten 30 Tage
future_predictions = predict_next_30_days(last_30_days_features, loaded_model, loaded_scaler, features_to_use)

# Visualisierung der Vorhersagen
plt.figure(figsize=(15, 7))
plt.plot(data.index, data['sby_need'], label='Actual sby_need', color='blue')
plt.plot(future_predictions.index, future_predictions['predicted_sby_need'], label='Future Predicted sby_need', color='green')
plt.title('Actual and Future Predicted sby_need')
plt.xlabel('Date')
plt.ylabel('sby_need')
plt.legend()
plt.show()

# Vorhersagen ausgeben
print(future_predictions)

# Anpassungen der Vorhersagen nach dem Erstellen der Vorhersagen
future_predictions['predicted_sby_need'] = future_predictions['predicted_sby_need'].apply(lambda x: 15 if x < 15 else x * 1.10)
print(future_predictions)
# Runden der Vorhersagen
future_predictions['predicted_sby_need'] = future_predictions['predicted_sby_need'].round(0)

# Visualisierung der Vorhersagen
plt.figure(figsize=(15, 7))
plt.plot(data.index, data['sby_need'], label='Actual sby_need', color='blue')
plt.plot(future_predictions.index, future_predictions['predicted_sby_need'], label='Future Predicted sby_need', color='green')
plt.title('Actual and Future Predicted sby_need')
plt.xlabel('Date')
plt.ylabel('sby_need')
plt.legend()
plt.show()