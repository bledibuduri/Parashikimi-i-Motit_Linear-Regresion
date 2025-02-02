import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Importet për Prophet dhe SARIMA
from prophet import Prophet
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ===========================
# 1. Leximi dhe përgatitja e të dhënave
# ===========================
# Leximi i të dhënave (file CSV me kolonat 'Timestamp' dhe 'Humidity')
data = pd.read_csv('Humidity/humidity_filled.csv')

# Shikojmë rreshtat e parë të datasetit
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())

# Kontrollojmë vlerat null në kolonën 'Humidity'
null_values = data['Humidity'].isnull().sum()
print(f"Number of null values in 'Humidity' column: {null_values}")

# Konvertimi i kolonës 'Timestamp' në datetime dhe shtimi i kolonave shtesë
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data['Day'] = data['Timestamp'].dt.day
data['Hour'] = data['Timestamp'].dt.hour

# ===========================
# 2. Modeli i Regresionit Linear (si në versionin origjinal)
# ===========================
# Përcaktimi i variablave pavarur dhe të varur
X = data[['Year', 'Month', 'Day', 'Hour']]
y = data['Humidity']

# Ndarja në të dhëna trajnimi dhe test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Krijimi dhe trajnimi i modelit të regresionit linear
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Parashikimi mbi të dhënat e testuara dhe vlerësimi
y_pred = lin_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Linear Regression): {mse}")
print('R² (Linear Regression):', r2_score(y_test, y_pred))

# Parashikimi për vitin 2024 me modelin Linear Regression
date_rng = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:00:00', freq='H')
X_2024 = pd.DataFrame({
    'Year': date_rng.year,
    'Month': date_rng.month,
    'Day': date_rng.day,
    'Hour': date_rng.hour
})
y_2024_lin_pred = lin_model.predict(X_2024)

predicted_2024_lin = pd.DataFrame({
    'Timestamp': date_rng,
    'Humidity': y_2024_lin_pred
})

# ===========================
# 3. Modeli Prophet
# ===========================
# Përgatitja e të dhënave për Prophet (duhet të kenë kolonat "ds" dhe "y")
prophet_data = data[['Timestamp', 'Humidity']].rename(columns={'Timestamp': 'ds', 'Humidity': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)

# Krijimi i dataframe të së ardhmes për vitin 2024 (horare)
future_prophet = prophet_model.make_future_dataframe(periods=len(date_rng), freq='H')
forecast_prophet = prophet_model.predict(future_prophet)

# Mblidhim vetëm parashikimet për vitin 2024
forecast_prophet_2024 = forecast_prophet[forecast_prophet['ds'].between('2024-01-01', '2024-12-31')]
# Ruajmë kolonën "yhat" (parashikimi)
predicted_2024_prophet = forecast_prophet_2024[['ds', 'yhat']].rename(columns={'ds': 'Timestamp', 'yhat': 'Humidity'})
print("Prophet forecast for 2024 ready.")

# ===========================
# 4. Modeli SARIMA
# ===========================
# Për SARIMA, bëjmë indeksin me bazë Timestamp dhe caktojmë frekuencën H (orare)
ts = data.set_index('Timestamp')['Humidity'].asfreq('H')

# Shtojmë një model SARIMA; këtu përdorim parametrat (1,1,1)(1,1,1,24) për sezonin orar (24 orë në ditë)
sarima_model = sm.tsa.statespace.SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,24))
sarima_result = sarima_model.fit(disp=False)

# Parashikimi për hapat e ardhshëm = numri i orëve në vitin 2024
steps = len(date_rng)
sarima_forecast = sarima_result.get_forecast(steps=steps)
sarima_forecast_mean = sarima_forecast.predicted_mean
sarima_forecast_mean.index = date_rng  # Vendosim indeksin si datat e vitit 2024

predicted_2024_sarima = sarima_forecast_mean.reset_index()
predicted_2024_sarima.columns = ['Timestamp', 'Humidity']
print("SARIMA forecast for 2024 ready.")

# ===========================
# 5. Ruajtja e të dhënave të bashkuara (historike + parashikimet Linear Regression)
# ===========================
all_data = pd.concat([data[['Timestamp', 'Humidity']], predicted_2024_lin], ignore_index=True)
all_data.to_csv('Humidity/humidity_data_2018_2024.csv', index=False)
print("Të dhënat historike dhe parashikimet me Linear Regression janë ruajtur në 'humidity_data_2018_2024.csv'.")

# ===========================
# 6. Grafiku i parashikimeve për vitin 2024
# ===========================
plt.figure(figsize=(16, 8))
plt.plot(date_rng, y_2024_lin_pred, label='Linear Regression', color='red')
plt.plot(predicted_2024_prophet['Timestamp'], predicted_2024_prophet['Humidity'], label='Prophet', color='blue')
plt.plot(predicted_2024_sarima['Timestamp'], predicted_2024_sarima['Humidity'], label='SARIMA', color='green')
plt.title('Parashikimi i Lagështirës për Vitin 2024', fontsize=16)
plt.xlabel('Data', fontsize=14)
plt.ylabel('Lagështira (%)', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Mund të përdorim edhe një grafik interaktiv me Plotly për një shikim më të detajuar:
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=date_rng, y=y_2024_lin_pred, mode='lines', name='Linear Regression', line=dict(color='red')))
fig.add_trace(go.Scatter(x=predicted_2024_prophet['Timestamp'], y=predicted_2024_prophet['Humidity'], mode='lines', name='Prophet', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predicted_2024_sarima['Timestamp'], y=predicted_2024_sarima['Humidity'], mode='lines', name='SARIMA', line=dict(color='green')))

fig.update_layout(
    title='Parashikimi i Lagështirës për Vitin 2024',
    xaxis_title='Data',
    yaxis_title='Lagështira (%)',
    legend_title='Modeli',
    template='plotly_white'
)
fig.show()
