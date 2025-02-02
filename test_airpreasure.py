import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Importet për Prophet dhe SARIMA
from prophet import Prophet
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ============================================
# 1. Leximi dhe Përgatitja e të Dhënave
# ============================================
# Lexo skedarin CSV. Ky file duhet të ketë kolonat 'Timestamp' dhe 'Air_Pressure'
data = pd.read_csv('AirPreasure/airpreasure.csv')

# Shiko rreshtat e parë të datasetit
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())

# Kontrollo vlerat null në kolonën 'Air_Pressure'
null_values = data['Air_Pressure'].isnull().sum()
print(f"Number of null values in 'Air_Pressure' column: {null_values}")

# Konverto 'Timestamp' në datetime dhe shto kolona shtesë për vitin, muajin, ditën dhe orën
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d.%m.%Y %H:%M:%S')
data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data['Day'] = data['Timestamp'].dt.day
data['Hour'] = data['Timestamp'].dt.hour

# ============================================
# 2. Parashikimi me Linear Regression për Air Pressure
# ============================================
# Përgatitja e të dhënave për modelin Linear
X = data[['Year', 'Month', 'Day', 'Hour']]  # Karakteristika: Viti, Muaji, Dita, Ora
y = data['Air_Pressure']                    # Target: Air Pressure

# Ndara të dhënat në trajnime dhe test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Krijo dhe trajno modelin Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Parashiko për të dhënat e testuara dhe llogarit gabimet
y_pred = lin_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Linear Regression): {mse}")
print("R²:", r2_score(y_test, y_pred))

# Parashikimi për vitin 2024 me Linear Regression
date_rng = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:00:00', freq='H')
X_2024 = pd.DataFrame({
    'Year': date_rng.year,
    'Month': date_rng.month,
    'Day': date_rng.day,
    'Hour': date_rng.hour
})
y_2024_lin_pred = lin_model.predict(X_2024)

# Ruaje parashikimet për vitin 2024 nga Linear Regression
predicted_2024_lin = pd.DataFrame({
    'Timestamp': date_rng,
    'Air_Pressure': y_2024_lin_pred
})

# ============================================
# 3. Parashikimi me Prophet për Air Pressure
# ============================================
# Përgatitja e të dhënave për Prophet: kolonat duhet të jenë 'ds' dhe 'y'
prophet_data = data[['Timestamp', 'Air_Pressure']].rename(columns={'Timestamp': 'ds', 'Air_Pressure': 'y'})
prophet_model = Prophet()
prophet_model.fit(prophet_data)

# Krijo dataframe për të ardhmen për vitin 2024 (orare)
future_prophet = prophet_model.make_future_dataframe(periods=len(date_rng), freq='H')
forecast_prophet = prophet_model.predict(future_prophet)

# Filtrimi i parashikimeve vetëm për vitin 2024
forecast_prophet_2024 = forecast_prophet[forecast_prophet['ds'].between('2024-01-01', '2024-12-31')]
predicted_2024_prophet = forecast_prophet_2024[['ds', 'yhat']].rename(columns={'ds': 'Timestamp', 'yhat': 'Air_Pressure'})

# ============================================
# 4. Parashikimi me SARIMA për Air Pressure
# ============================================
# Për SARIMA, vendosim indeksin sipas Timestamp me frekuencë orare
ts = data.set_index('Timestamp')['Air_Pressure'].asfreq('H')

# Përdorim një model SARIMA me parametrat (1,1,1) për komponentin jo sezonier dhe (1,1,1,24) për sezonier (24 orë për ditë)
sarima_model = sm.tsa.statespace.SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,24))
sarima_result = sarima_model.fit(disp=False)

# Parashikimi për hapat e ardhshëm, ku numri i hapave është i barabartë me numrin e orëve në vitin 2024
steps = len(date_rng)
sarima_forecast = sarima_result.get_forecast(steps=steps)
sarima_forecast_mean = sarima_forecast.predicted_mean
sarima_forecast_mean.index = date_rng  # Përdorim date_rng për indeksin

predicted_2024_sarima = sarima_forecast_mean.reset_index()
predicted_2024_sarima.columns = ['Timestamp', 'Air_Pressure']

# ============================================
# 5. Ruajtja e të Dhënave dhe Grafiku i Rezultateve
# ============================================
# Bashkimi i të dhënave historike me parashikimet nga Linear Regression (në rast se dëshiron ta ruash)
all_data = pd.concat([data[['Timestamp', 'Air_Pressure']], predicted_2024_lin], ignore_index=True)
all_data.to_csv('AirPreasure/airpreasure_data_2018_2024.csv', index=False)
print("Të dhënat janë ruajtur në file-in airpreasure_data_2018_2024.csv")

# Shiko disa statistika të përmbledhura
print(all_data.head())
print(all_data.shape)
print(all_data.info())
print(all_data.describe())

# Grafik interaktiv me Plotly për të krahasuar parashikimet për vitin 2024
fig = go.Figure()
fig.add_trace(go.Scatter(x=date_rng, y=y_2024_lin_pred, mode='lines', name='Linear Regression', line=dict(color='red')))
fig.add_trace(go.Scatter(x=predicted_2024_prophet['Timestamp'], y=predicted_2024_prophet['Air_Pressure'],
                         mode='lines', name='Prophet', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predicted_2024_sarima['Timestamp'], y=predicted_2024_sarima['Air_Pressure'],
                         mode='lines', name='SARIMA', line=dict(color='green')))

fig.update_layout(
    title='Parashikimi i Presionit të Ajrit për Vitin 2024',
    xaxis_title='Data',
    yaxis_title='Air Pressure',
    legend_title='Modeli',
    template='plotly_white'
)
fig.show()

# Gjithashtu, mund të shfaqim disa grafika statikë me matplotlib dhe seaborn
plt.figure(figsize=(20, 10))
plt.title("Air Pressure From 2018-2024 (Historike + Parashikimet Linear Regression)")
sns.lineplot(x=all_data['Timestamp'], y=all_data['Air_Pressure'], marker='o', color='red')
plt.xlabel("Data")
plt.ylabel("Air Pressure")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Grafiku i mesatareve mujore
all_data['Timestamp'] = pd.to_datetime(all_data['Timestamp'], utc=True, errors='coerce')
monthly_data = all_data.set_index('Timestamp').resample('M').mean()[1:]
plt.figure(figsize=(20,10))
plt.title("Monthly Average Air Pressure From 2018-2024")
sns.lineplot(x=monthly_data.index, y=monthly_data['Air_Pressure'])
plt.xlabel("Month")
plt.show()

# Grafiku i mesatares mujore me barplot
plt.figure(figsize=(20,10))
plt.title("Presioni i Ajrit - Mesatarja Mujore Nga 2018-2024")
sns.barplot(x=monthly_data.index.month, y=monthly_data['Air_Pressure'], palette="mako")
plt.xlabel("Month")
plt.ylabel("Air Pressure")
plt.show()
