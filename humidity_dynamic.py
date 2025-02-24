import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Importet për Prophet dhe SARIMA
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
# Krijohet një kolonë që përfaqëson ditën e vitit me pjesë dhore
data['doy_float'] = data['Timestamp'].dt.dayofyear + data['Timestamp'].dt.hour / 24.0

# ===========================
# 2. Modeli i Regresionit Linear
# ===========================
X = data[['Year', 'Month', 'Day', 'Hour']]
y = data['Humidity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))
print("Linear Regression R²:", r2_score(y_test, y_pred))

# Parashikimi për vitin 2024 me Linear Regression
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

# (Opsionale) Ruajtja e të dhënave bashkuara
all_data = pd.concat([data[['Timestamp', 'Humidity']], predicted_2024_lin], ignore_index=True)
all_data.to_csv('Humidity/humidity_data_2018_2024.csv', index=False)
print("Të dhënat historike dhe parashikimet me Linear Regression janë ruajtur në 'humidity_data_2018_2024.csv'.")

# ===========================
# 5. Krijimi i Dashboard-it me Dash dhe Sinkronizimi i Zoom-it
# ===========================
import dash
from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Paraquitje Dinamike e të Dhënave të Lagështirës dhe Parashikimeve për 2024"),
    # Store për ruajtjen e intervalit të përbashkët të x-aksit
    dcc.Store(id='shared-xaxis-range'),
    
    html.Div([
        html.H2("Zgjidhni Intervalin e Ditës së Vitit"),
        dcc.RangeSlider(
            id='dayofyear-range',
            min=1,
            max=366,
            step=1,
            value=[1, 366],
            marks={i: str(i) for i in range(1, 367, 30)}
        )
    ], style={'width': '80%', 'margin': 'auto'}),
    
    html.Hr(),
    html.H2("Të Dhënat Historike sipas Vitit"),
    html.Div(id='historical-graphs'),
    
    html.Hr(),
    html.H2("Parashikimet për Vitin 2024"),
    html.Div(id='forecast-graphs')
])

# Callback për krijimin e grafikëve, duke përdorur edhe intervalin e ruajtur në 'shared-xaxis-range'
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
    for yr in years:
        yr_int = int(yr)  # Konvertohet në int për t'u përdorur në id
        df_year = data[data['Year'] == yr]
        df_year_filtered = df_year[(df_year['doy_float'] >= start_day) & (df_year['doy_float'] <= end_day)]
        if df_year_filtered.empty:
            continue
        fig = px.line(
            df_year_filtered, 
            x='doy_float', 
            y='Humidity', 
            title=f"Lagështira në Vitin {yr_int}",
            labels={'doy_float': 'Dita e Vitit (me pjesë dhore)', 'Humidity': 'Lagështira (%)'}
        )
        if shared_range is not None:
            fig.update_layout(xaxis_range=shared_range)
        historical_graphs.append(
            dcc.Graph(
                id={'type': 'sync-graph-hist', 'index': yr_int},
                figure=fig
            )
        )
    
    forecast_graphs = []
    # Parashikimi me Linear Regression
    df_lr = predicted_2024_lin[(predicted_2024_lin['doy_float'] >= start_day) & (predicted_2024_lin['doy_float'] <= end_day)]
    fig_lr = px.line(
        df_lr, 
        x='doy_float', 
        y='Humidity',
        title="2024 - Parashikimi: Linear Regression",
        labels={'doy_float': 'Dita e Vitit (me pjesë dhore)', 'Humidity': 'Lagështira (%)'}
    )
    if shared_range is not None:
        fig_lr.update_layout(xaxis_range=shared_range)
    forecast_graphs.append(
        dcc.Graph(
            id={'type': 'sync-graph-forecast', 'index': 'Linear'},
            figure=fig_lr
        )
    )
    
    # Parashikimi me Prophet
    df_prophet = predicted_2024_prophet[(predicted_2024_prophet['doy_float'] >= start_day) & (predicted_2024_prophet['doy_float'] <= end_day)]
    fig_prophet = px.line(
        df_prophet, 
        x='doy_float', 
        y='Humidity',
        title="2024 - Parashikimi: Prophet",
        labels={'doy_float': 'Dita e Vitit (me pjesë dhore)', 'Humidity': 'Lagështira (%)'}
    )
    if shared_range is not None:
        fig_prophet.update_layout(xaxis_range=shared_range)
    forecast_graphs.append(
        dcc.Graph(
            id={'type': 'sync-graph-forecast', 'index': 'Prophet'},
            figure=fig_prophet
        )
    )
    
    # Parashikimi me SARIMA
    df_sarima = predicted_2024_sarima[(predicted_2024_sarima['doy_float'] >= start_day) & (predicted_2024_sarima['doy_float'] <= end_day)]
    fig_sarima = px.line(
        df_sarima, 
        x='doy_float', 
        y='Humidity',
        title="2024 - Parashikimi: SARIMA",
        labels={'doy_float': 'Dita e Vitit (me pjesë dhore)', 'Humidity': 'Lagështira (%)'}
    )
    if shared_range is not None:
        fig_sarima.update_layout(xaxis_range=shared_range)
    forecast_graphs.append(
        dcc.Graph(
            id={'type': 'sync-graph-forecast', 'index': 'SARIMA'},
            figure=fig_sarima
        )
    )
    
    return historical_graphs, forecast_graphs

# Callback për sinkronizimin e zoom-it midis grafikëve
@app.callback(
    Output('shared-xaxis-range', 'data'),
    Input({'type': 'sync-graph-hist', 'index': ALL}, 'relayoutData'),
    Input({'type': 'sync-graph-forecast', 'index': ALL}, 'relayoutData'),
    State('shared-xaxis-range', 'data')
)
def update_shared_range(hist_relayout_list, forecast_relayout_list, current_range):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Kërkojmë ndryshime në 'xaxis.range[0]' dhe 'xaxis.range[1]' nga të gjitha grafiket
    for relayoutData in (hist_relayout_list + forecast_relayout_list):
        if relayoutData and ('xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData):
            new_range = [relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
            return new_range
    return current_range

if __name__ == '__main__':
    app.run_server(debug=True)