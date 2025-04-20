import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Read the dataset
data = pd.read_csv('Temperature/temperature.csv')

# Convert 'Timestamp' column to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

# Set 'Timestamp' as the index
data.set_index('Timestamp', inplace=True)

# Check for null values before filling
print(f"Number of null values in 'Temperature' column before filling: {data['Temperature'].isnull().sum()}")

# Fill missing values using time-based interpolation
data['Temperature'] = data['Temperature'].interpolate(method='time')

# Check for null values after filling
print(f"Number of null values in 'Temperature' column after filling: {data['Temperature'].isnull().sum()}")

# Fit the SARIMA model
sarima_model = SARIMAX(data['Temperature'], 
                       order=(1, 1, 1),  # (p, d, q) - tune these values as needed
                       seasonal_order=(1, 1, 1, 24),  # (P, D, Q, S) - seasonal components (24 for hourly data)
                       enforce_stationarity=False,
                       enforce_invertibility=False)

sarima_results = sarima_model.fit()

# Generate future timestamps (hourly for 2024)
future_dates = pd.date_range(start='2024-01-01', periods=365 * 24, freq='H')

# Forecast for 2024
forecast = sarima_results.get_forecast(steps=365 * 24)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Save predictions for 2024 in a DataFrame
predicted_2024 = pd.DataFrame({'Timestamp': future_dates, 'Temperature': forecast_mean})
predicted_2024['Temperature'] = predicted_2024['Temperature'].fillna(data['Temperature'].mean())  # Fill any NaN values with mean

# Combine existing and predicted data
all_data = pd.concat([data[['Temperature']], predicted_2024.set_index('Timestamp')], ignore_index=False)

# Save combined data to a CSV file
all_data.to_csv('Temperature/Temperature_data_2018_2024_sarima2.csv', index=True)
print("Data saved to 'Temperature_data_2018_2024_sarima.csv'")

# ---- VISUALIZATIONS ----

# 1. Plot entire dataset (2018–2024)
plt.figure(figsize=(20, 10))
plt.plot(all_data.index, all_data['Temperature'], label='Temperature', color='blue')
plt.title("Temperature From 2018-2024")
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Calculate and plot monthly averages
monthly_data = all_data.resample('M').mean()[1:]  # Resample to monthly averages
avg_monthly_Temperature = monthly_data.reset_index()

plt.figure(figsize=(20, 10))
plt.title("Monthly Average Temperature From 2018-2024")
sns.lineplot(x=avg_monthly_Temperature['Timestamp'], y=avg_monthly_Temperature['Temperature'], marker='o', color='green')
plt.xlabel("Month")
plt.ylabel("Temperature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Bar plot for monthly averages
plt.figure(figsize=(20, 10))
plt.title("Temperature - Monthly Averages (2018-2024)")
sns.barplot(x=avg_monthly_Temperature['Timestamp'].dt.month, y=avg_monthly_Temperature['Temperature'], palette="mako")
plt.xlabel("Month")
plt.ylabel("Temperature")
plt.show()

# 4. Yearly averages
yearly_data = all_data.resample('Y').mean()
yearly_data.reset_index(inplace=True)

plt.figure(figsize=(20, 10))
plt.title("Yearly Average Temperature From 2018-2024")
sns.lineplot(x=yearly_data['Timestamp'], y=yearly_data['Temperature'], marker='o', color='purple')
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Interactive Plotly graph
fig = px.line(all_data, x=all_data.index, y='Temperature', title="Temperature From 2018-2024")
fig.update_traces(mode='lines+markers', line=dict(color='red'))
fig.show()

# 6. SARIMA forecast visualization
plt.figure(figsize=(20, 10))
plt.plot(all_data.index, all_data['Temperature'], label='Observed', color='blue')
plt.plot(predicted_2024['Timestamp'], predicted_2024['Temperature'], label='Forecasted', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title("SARIMA Forecast for Temperature")
plt.legend()
plt.show()
