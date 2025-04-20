import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from prophet import Prophet
import statsmodels.api as sm

# ===========================
# 1. Leximi dhe përgatitja e të dhënave
# ===========================
data = pd.read_csv('Humidity/humidity_filled.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data['Day'] = data['Timestamp'].dt.day
data['Hour'] = data['Timestamp'].dt.hour
data['doy_float'] = data['Timestamp'].dt.dayofyear + data['Timestamp'].dt.hour / 24.0

# ===========================
# 2. Modeli i Regresionit Linear
# ===========================
X = data[['Year', 'Month', 'Day', 'Hour']]
y = data['Humidity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

date_rng = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:00:00', freq='H')
X_2024 = pd.DataFrame({'Year': date_rng.year, 'Month': date_rng.month, 'Day': date_rng.day, 'Hour': date_rng.hour})
y_2024_lin_pred = lin_model.predict(X_2024)

predicted_2024_lin = pd.DataFrame({'Timestamp': date_rng, 'Humidity': y_2024_lin_pred})
predicted_2024_lin['doy_float'] = predicted_2024_lin['Timestamp'].dt.dayofyear + predicted_2024_lin['Timestamp'].dt.hour / 24.0

# ===========================
# 3. Modeli Prophet
# ===========================
prophet_data = data[['Timestamp', 'Humidity']].rename(columns={'Timestamp': 'ds', 'Humidity': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)
future_prophet = prophet_model.make_future_dataframe(periods=len(date_rng), freq='H')
forecast_prophet = prophet_model.predict(future_prophet)
forecast_prophet_2024 = forecast_prophet[forecast_prophet['ds'].between('2024-01-01', '2024-12-31')]
predicted_2024_prophet = forecast_prophet_2024[['ds', 'yhat']].rename(columns={'ds': 'Timestamp', 'yhat': 'Humidity'})
predicted_2024_prophet['doy_float'] = predicted_2024_prophet['Timestamp'].dt.dayofyear + predicted_2024_prophet['Timestamp'].dt.hour / 24.0

# ===========================
# 4. Modeli SARIMA
# ===========================
ts = data.set_index('Timestamp')['Humidity'].asfreq('H')
sarima_model = sm.tsa.statespace.SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,24))
sarima_result = sarima_model.fit(disp=False)
steps = len(date_rng)
sarima_forecast = sarima_result.get_forecast(steps=steps)
sarima_forecast_mean = sarima_forecast.predicted_mean
sarima_forecast_mean.index = date_rng
predicted_2024_sarima = sarima_forecast_mean.reset_index()
predicted_2024_sarima.columns = ['Timestamp', 'Humidity']
predicted_2024_sarima['doy_float'] = predicted_2024_sarima['Timestamp'].dt.dayofyear + predicted_2024_sarima['Timestamp'].dt.hour / 24.0

# ===========================
# 5. Krijimi i Dashboard-it me Dash
# ===========================
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Paraqitje Dinamike e të Dhënave të Lagështirës dhe Parashikimeve për 2024"),
    dcc.Store(id='shared-xaxis-range'),
    
    html.Div([
        html.H2("Zgjidhni Intervalin e Ditës së Vitit"),
        dcc.RangeSlider(
            id='dayofyear-range',
            min=1, max=366, step=1, value=[1, 366],
            marks={i: str(i) for i in range(1, 367, 30)}
        )
    ], style={'width': '80%', 'margin': 'auto'}),
    
    html.Hr(),
    html.Div(id='historical-graphs', style={'display': 'grid', 'grid-template-columns': 'repeat(2, 1fr)', 'gap': '20px'}),
    
    html.Hr(),
    html.Div(id='forecast-graphs', style={'display': 'grid', 'grid-template-columns': 'repeat(3, 1fr)', 'gap': '20px'})
])

@app.callback(
    Output('historical-graphs', 'children'),
    Output('forecast-graphs', 'children'),
    Input('dayofyear-range', 'value'),
    Input('shared-xaxis-range', 'data')
)
def update_graphs(day_range, shared_range):
    start_day, end_day = day_range
    historical_graphs = []
    years = sorted(data['Year'].unique())
    
    for i in range(0, len(years), 2):
        if i+1 < len(years):
            df1 = data[data['Year'] == years[i]]
            df2 = data[data['Year'] == years[i+1]]
            fig1 = px.line(df1, x='doy_float', y='Humidity', title=f"Lagështira në {years[i]}")
            fig2 = px.line(df2, x='doy_float', y='Humidity', title=f"Lagështira në {years[i+1]}")
            historical_graphs.append(html.Div([
                dcc.Graph(figure=fig1),
                dcc.Graph(figure=fig2)
            ]))
    
    forecast_graphs = [dcc.Graph(figure=px.line(df, x='doy_float', y='Humidity', title=f"2024 - {model}")) for model, df in {"Linear Regression": predicted_2024_lin, "Prophet": predicted_2024_prophet, "SARIMA": predicted_2024_sarima}.items()]
    return historical_graphs, forecast_graphs

if __name__ == '__main__':
    app.run_server(debug=True)
