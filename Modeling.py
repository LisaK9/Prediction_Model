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
    'RFE': RFE(estimator=SVR(kernel='linear'), n_features_to_select=10),
    'Lasso': Lasso(),
    'Univariate': SelectKBest(score_func=f_regression, k=10),
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
