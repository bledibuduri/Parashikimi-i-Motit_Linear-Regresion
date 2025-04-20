import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from prophet import Prophet
import plotly.graph_objects as go
import plotly.subplots as sp
import warnings
warnings.filterwarnings("ignore")

# ==========================
# 1. Leximi dhe Përgatitja e të Dhënave
# ==========================
data = pd.read_csv('Humidity/humidity_filled.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Year'] = data['Timestamp'].dt.year

# Lista e viteve të disponueshme
years = sorted(data['Year'].unique())

# ==========================
# 2. Modeli i Regresionit Linear
# ==========================
X = data[['Year']]
y = data['Humidity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Krijimi i të dhënave për parashikimin 2024
date_rng = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
X_2024 = pd.DataFrame({'Year': date_rng.year})
y_2024_lin_pred = lin_model.predict(X_2024)

predicted_2024_lin = pd.DataFrame({'Timestamp': date_rng, 'Humidity': y_2024_lin_pred})

# ==========================
# 3. Modeli Prophet
# ==========================
prophet_data = data[['Timestamp', 'Humidity']].rename(columns={'Timestamp': 'ds', 'Humidity': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)

future_prophet = prophet_model.make_future_dataframe(periods=len(date_rng), freq='H')
forecast_prophet = prophet_model.predict(future_prophet)
forecast_prophet_2024 = forecast_prophet[forecast_prophet['ds'].between('2024-01-01', '2024-12-31')]
predicted_2024_prophet = forecast_prophet_2024[['ds', 'yhat']].rename(columns={'ds': 'Timestamp', 'yhat': 'Humidity'})

# ==========================
# 4. Modeli SARIMA
# ==========================
ts = data.set_index('Timestamp')['Humidity'].asfreq('H')
sarima_model = sm.tsa.statespace.SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,24))
sarima_result = sarima_model.fit(disp=False)

sarima_forecast = sarima_result.get_forecast(steps=len(date_rng))
sarima_forecast_mean = sarima_forecast.predicted_mean
sarima_forecast_mean.index = date_rng

predicted_2024_sarima = sarima_forecast_mean.reset_index()
predicted_2024_sarima.columns = ['Timestamp', 'Humidity']

# ==========================
# 5. Krijimi i Dashboard-it me Plotly
# ==========================
fig = sp.make_subplots(
    rows=len(years) + 3, cols=1, subplot_titles=[f"Të dhënat e vitit {year}" for year in years] +
    ["Parashikimi 2024 (Linear Regression)", "Parashikimi 2024 (Prophet)", "Parashikimi 2024 (SARIMA)"],
    shared_xaxes=True  # Ky ndryshim lejon që zoom-in të aplikohet në të gjitha grafiket
)

# Shto të dhënat historike në grafikë
for i, year in enumerate(years):
    yearly_data = data[data['Year'] == year]
    fig.add_trace(
        go.Scatter(x=yearly_data['Timestamp'], y=yearly_data['Humidity'], mode='lines', name=f'Viti {year}'), row=i+1, col=1
    )

# Shto parashikimet e vitit 2024
fig.add_trace(
    go.Scatter(x=predicted_2024_lin['Timestamp'], y=predicted_2024_lin['Humidity'], mode='lines',
               name='Linear Regression', line=dict(color='red')), row=len(years) + 1, col=1
)
fig.add_trace(
    go.Scatter(x=predicted_2024_prophet['Timestamp'], y=predicted_2024_prophet['Humidity'], mode='lines',
               name='Prophet', line=dict(color='blue')), row=len(years) + 2, col=1
)
fig.add_trace(
    go.Scatter(x=predicted_2024_sarima['Timestamp'], y=predicted_2024_sarima['Humidity'], mode='lines',
               name='SARIMA', line=dict(color='green')), row=len(years) + 3, col=1
)

# ==========================
# 6. Personalizimi dhe Shfaqja e Dashboard-it
# ==========================
fig.update_layout(
    height=3000, width=1200, 
    title_text="Analiza dhe Parashikimi i Lagështirës (2018 - 2024)", 
    showlegend=True
)

fig.show()
