import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

#Daten einlesen
data=pd.read_csv("sickness_table.csv")
print(data.head(5))
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

# Konvertierung der 'date'-Spalte in das Datumsformat
data['date'] = pd.to_datetime(data['date'])

#Indexspalte löschen
data = data.drop(columns=["Unnamed: 0"])
data = data.drop(columns=["n_sby"])

print("\nBeschreibende Statistiken:")
print(data.describe())

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

data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
# Plotten der Zeitreihen für die relevanten Variablen
fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Plot für n_sick
axs[0].plot(data.index, data['n_sick'], color='blue')
axs[0].set_title('n_sick (Anzahl der Krankenmeldungen)')
axs[0].set_ylabel('n_sick')

# Plot für calls
axs[1].plot(data.index, data['calls'], color='orange')
axs[1].set_title('calls (Anzahl der Anrufe)')
axs[1].set_ylabel('calls')

# Plot für n_duty
axs[2].plot(data.index, data['n_duty'], color='green')
axs[2].set_title('n_duty (Anzahl der Mitarbeiter im Dienst)')
axs[2].set_ylabel('n_duty')

# Plot für sby_need
axs[3].plot(data.index, data['sby_need'], color='red')
axs[3].set_title('sby_need (Zielvariable)')
axs[3].set_ylabel('sby_need')
axs[3].set_xlabel('Datum')

plt.tight_layout()
plt.show()

# Korrelationen zwischen den numerischen Variablen
correlation_matrix = data.corr()

# Heatmap der Korrelationen
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korrelationsmatrix')
plt.show()
print(data.dtypes)

# Ändern des Datentyps von 'week' zu int64
data['week'] = data['week'].astype('int64')

# VIF berechnen
X = add_constant(data.drop('sby_need', axis=1))  # Die Zielvariable ausschließen
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# VIF-Daten anzeigen
print("VIF: ",vif_data)


# Histogramme der wichtigsten Variablen
variables_to_plot = ['n_sick', 'calls', 'n_duty', 'sby_need']

plt.figure(figsize=(15, 10))
for i, variable in enumerate(variables_to_plot, 1):
    plt.subplot(2, 4, i)
    sns.histplot(data[variable], bins=30, kde=True)
    plt.title(f'Verteilung von {variable}')
plt.tight_layout()
plt.show()

# Zielvariable und unabhängige Variablen definieren
target_variable = 'sby_need'
independent_variables = [col for col in data.columns if col != target_variable]

#Scatterplot
# Gesamtgröße der Figur festlegen
plt.figure(figsize=(15, 10))
plt.title(f'Scatterplot der unabhängigen Variablen zur Zielvariable')
# Schleife zum Erstellen der Subplots
for i, feature in enumerate(independent_variables, 1):
    plt.subplot(3, 4, i)
    sns.scatterplot(x=data[feature], y=data[target_variable])
    plt.xlabel(feature)
    plt.ylabel(target_variable)
    # Pearson-Korrelation berechnen und anzeigen
    #correlation, p_value = pearsonr(data[feature], data[target_variable])
    #plt.text(0.05, 0.9, f'Pearson-Korrelation: {correlation:.2f}', transform=plt.transAxes, fontsize=10,
    #             color='red')

# Layout anpassen und Plot anzeigen
plt.tight_layout()
plt.show()


# Funktion für saisonale Trends
def plot_seasonal_trends_with_year_and_month(variable):
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 1, 1)
    plt.title(f'Seasonal Trends for {variable} (by Month and Year)')

    # Boxplot nach monat
    #plt.subplot(2, 1, 1)
    data['month'] = data.index.month
    sns.boxplot(x='month', y=variable, data=data, palette="viridis")
    plt.ylabel(variable)
    plt.xlabel('Month')
    plt.xticks(ticks=range(0, 12),
               labels=['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'])

    # Boxplot nach Jahr
    plt.subplot(2, 1, 2)
    sns.boxplot(x='year', y=variable, data=data, palette="viridis")
    plt.ylabel(variable)
    plt.xlabel('Year')
    plt.tight_layout()
    plt.show()


# Variablen für Plots
variables = ['n_sick', 'calls', 'sby_need', 'n_duty']

# Plot boxplots
for var in variables:
    plot_seasonal_trends_with_year_and_month(var)

# Funktion für saisonale Trends
def plot_seasonal_trends_with_season_and_weekday(variable):
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 1, 1)
    plt.title(f'Seasonal Trends for {variable} (by Day of Week and Season)')

    # Boxplot je WOchentag
    plt.subplot(2, 1, 1)
    sns.boxplot(x='day_of_week', y=variable, data=data, palette="viridis")
    plt.xlabel('Day of Week')
    plt.ylabel(variable)
    plt.xticks(ticks=range(7), labels=['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So'])

    # Boxplot je Season
    plt.subplot(2, 1, 2)
    sns.boxplot(x='season', y=variable, data=data, palette="viridis")
    plt.xlabel('Season')
    plt.ylabel(variable)
    plt.xticks(ticks=range(4), labels=['WI', 'FJ', 'SO', 'HE'])

    plt.tight_layout()
    plt.show()


# Variablen für Plots
variables = ['n_sick', 'calls', 'sby_need', 'n_duty']

# Plot boxplots
for var in variables:
    plot_seasonal_trends_with_season_and_weekday(var)

def plot_acf_pacf_and_decomposition(variable, data, lags=50, period=7, model='additive'):
    plt.figure(figsize=(16, 10))
    #plt.subplot(2, 1, 1)
    # Autokorrelationsfunktion (ACF) plotten
    plot_acf(data[variable], lags=lags)
    plt.title(f'Autocorrelation Function (ACF) for {variable}')

    # Partielle Autokorrelationsfunktion (PACF) plotten
    #plt.figure(figsize=(14, 5))
    #plt.subplot(2, 1, 2)
    plot_pacf(data[variable], lags=lags)
    plt.title(f'Partial Autocorrelation Function (PACF) for {variable}')
    #plt.tight_layout()
    #plt.show()

    # Saisonalität und Trend analysieren
    decomposition = seasonal_decompose(data[variable], model=model, period=period)
    decomposition.plot()
    plt.suptitle(f'Seasonal Decomposition of {variable}', fontsize=16)

    plt.show()




# Die Funktion für jede Variable aufrufen
for var in variables:
    plot_acf_pacf_and_decomposition(var, data, lags=50, period=7)

