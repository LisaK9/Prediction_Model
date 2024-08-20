"""Hier werden meherer Modelle trainiert, Feature-Selection durchgeführt, sowie
Hyperparameter-Tuning angewendet. Die Modelle werden evaluiert und das Modell ausgewählt,
welches am Besten performed. Dies wird dann als pkl gespeichert und kann für die anschlienende
Vorhersage verwendet werden.
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Lasso, Ridge, LinearRegression
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Vorverarbeitete Daten laden
data = pd.read_csv('sickness_modeling.csv')
print(data.head(5))

#Features und Zielvariable festlegen
features = ['n_sick','calls','n_duty','month','day_of_week','season', 'week', 'holiday', 'year']
target = 'sby_need'

#Daten in Train- und Testdaten aufteilen
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#copying data for later comparisons
result_df=y_test.copy()
#making a pandas dataframe out of a pandas series
result_df=pd.DataFrame(result_df)
result_df=result_df.reset_index()
print(result_df)

#Initialisierung der Modelle

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "XGBoost Regressor": xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    "Support Vektor Regressor": SVR()
}

# Feature-Selektionstechniken
feature_selectors = {
    'RFE': RFE(estimator=SVR(kernel='linear'), n_features_to_select=6),
    'Lasso': Lasso(),
    'Univariate': SelectKBest(score_func=f_regression, k=6),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Hyperparameter-Raster für Grid Search
param_grid = {
    "Random Forest Regressor": {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5]
    },
    "Gradient Boosting Regressor": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5]
    },
    "XGBoost Regressor": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5]
    },
    "Support Vektor Regressor": {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1]
    }
}

#Funktion, um Modelle zu evaluieren
def evaluate_model(model, X_train, X_test, y_train, y_test):
    prediction = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    residuals = y_test - y_pred

    return prediction, mae, mse, rmse, r2, residuals, y_pred

# Features skalieren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results_baseline = {}
predictions_baseline = {}
residuals_baseline = {}
#Baseline Modell: Lineare Regression

for model_name, model in models.items():
    if model_name == 'Linear Regression':
        prediction, mae, mse, rmse, r2, residuals, y_pred = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        results_baseline[model_name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        residuals_baseline[model_name] = residuals
        predictions_baseline[model_name] = y_pred
        # combined dataframe for all predictions for future visualisation
        result_df[model_name] = predictions_baseline[model_name]

    else:
        pass


print(result_df)

evaluation_df = pd.DataFrame(results_baseline)
print(evaluation_df.T)

#Visualisierung der Vorhersagen
plt.figure(figsize=(15, 10))
for model_name in predictions_baseline.keys():
    plt.plot(result_df.index, result_df[target], label='Actual', color='blue')
    plt.plot(result_df.index, result_df[model_name], label=f'{model_name} Predicted')
    plt.title(f'Actual vs Predicted for {model_name}')
    plt.xlabel('Test Data Index')
    plt.ylabel('Ersatzfahrer')
    plt.legend()
    plt.show()

# Modelltraining, Feature Selection, Hyperparameter-Tuning
results = {}
residuals= {}
predictions_dict={}
best_estimators = {}
best_scalers = {}
best_selectors = {}

best_scores = {}
best_models = {}
best_params = {}

for fs_name, selector in feature_selectors.items(): #Feature-Selection
    selector.fit(X_train_scaled, y_train)
    if fs_name == 'Lasso':
        # Bei Lasso wird die Feature-Auswahl direkt auf Basis der Koeffizienten gemacht
        selected_features = np.where(selector.coef_ != 0)[0]
    elif fs_name == 'Random Forest':
        # Bei Random Forest ist die Feature-Auswahl basierend auf den Feature-Importances
        importances = selector.feature_importances_
        indices = np.argsort(importances)[::-1]
        selected_features = indices[:6]  # Wähle die Top 6 Features
    else:
        # Bei anderen Selektoren wie RFE und SelectKBest
        selected_features = selector.get_support(indices=True)

    # Daten mit ausgewählten Features erstellen
    X_train_selected = X_train_scaled[:, selected_features]
    X_test_selected = X_test_scaled[:, selected_features]

    for model_name, model in models.items(): #Modelle trainieren und Hyperparameter-Tuning
        key = f"{model_name} with {fs_name}"
        if model_name in param_grid: #Hyperparameter-Tuning
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model_name],
                                       scoring='neg_mean_squared_error', cv=5, verbose=1)
            grid_search.fit(X_train_selected, y_train)
            best_model = grid_search.best_estimator_
            best_params[key] = grid_search.best_params_
            print(f"\nBeste Hyperparameter für {key}: {grid_search.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train_selected, y_train)
            best_params[key] = "Standard-Hyperparameter"

        prediction, mae, mse, rmse, r2, residuals, y_pred = evaluate_model(model, X_train_selected, X_test_selected,
                                                                           y_train, y_test)
        key = f"{model_name} with {fs_name}"
        results[key] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        best_estimators[key] = best_model
        residuals[key] = residuals
        predictions_dict[key] = y_pred
        best_scalers[key] = scaler
        best_selectors[key] = selector

# Ergebnisse anzeigen
results_df1 = pd.DataFrame(results).T
print("Ergebnisse der Modelle:\n", results_df1)


# Hinzufügen der Vorhersagen zu result_df
for model_name, predictions in predictions_dict.items():
    result_df[model_name + '_pred'] = predictions
print(result_df)

# Visualisierung der Vorhersagen im Vergleich zu den tatsächlichen Werten
plt.figure(figsize=(15, 10))
for model_name in predictions_dict.keys():
    plt.plot(result_df.index, result_df[target], label='Actual', color='blue')
    plt.plot(result_df.index, result_df[model_name + '_pred'], label=f'{model_name} Predicted')
    plt.title(f'Actual vs Predicted for {model_name}')
    plt.xlabel('Test Data Index')
    plt.ylabel('Ersatzfahrer')
    plt.legend()
    plt.show()

import pickle

# Auswahl des besten Modells basierend auf MAE
best_model_key = min(results, key=lambda k: results[k]['MAE'])
print(f"Best model based on MAE: {best_model_key} with MAE: {results[best_model_key]['MAE']:.4f}")

# Überprüfen, ob das Modell trainiert wurde
if best_model_key not in best_estimators:
    # Trainiere das Modell, falls es noch nicht trainiert wurde
    best_selector = best_selectors[best_model_key]
    selected_features = best_selector.transform(X_train_scaled)

    best_model = models[best_model_key.split(" with ")[0]]
    best_model.fit(selected_features, y_train)
else:
    best_model = best_estimators[best_model_key]

# Modell als .pkl Datei speichern
model_filename = f"{best_model_key.replace(' ', '_').replace(':', '')}_best_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)
    print(f"Best model saved as {model_filename}")