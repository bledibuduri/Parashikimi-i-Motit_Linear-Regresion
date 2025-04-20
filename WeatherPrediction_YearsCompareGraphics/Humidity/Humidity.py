import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Read the dataset
data = pd.read_csv('Humidity/humidity_filled.csv')

# Convert 'Timestamp' column to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

# Set 'Timestamp' as the index
data.set_index('Timestamp', inplace=True)

# Check for null values before filling
print(f"Number of null values in 'Humidity' column before filling: {data['Humidity'].isnull().sum()}")

# Fill missing values using time-based interpolation
data['Humidity'] = data['Humidity'].interpolate(method='time')

# Check for null values after filling
print(f"Number of null values in 'Humidity' column after filling: {data['Humidity'].isnull().sum()}")

# Fit the SARIMA model
sarima_model = SARIMAX(data['Humidity'], 
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
predicted_2024 = pd.DataFrame({'Timestamp': future_dates, 'Humidity': forecast_mean})
predicted_2024['Humidity'] = predicted_2024['Humidity'].fillna(data['Humidity'].mean())  # Fill any NaN values with mean

# Combine existing and predicted data
all_data = pd.concat([data[['Humidity']], predicted_2024.set_index('Timestamp')], ignore_index=False)

# Save combined data to a CSV file
all_data.to_csv('Humidity/humidity_data_2018_2024_sarima.csv', index=True)
print("Data saved to 'humidity_data_2018_2024_sarima.csv'")

# ---- VISUALIZATIONS ----

# 1. Plot entire dataset (2018–2024)
plt.figure(figsize=(20, 10))
plt.plot(all_data.index, all_data['Humidity'], label='Humidity', color='blue')
plt.title("Humidity From 2018-2024")
plt.xlabel("Year")
plt.ylabel("Humidity")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Calculate and plot monthly averages
monthly_data = all_data.resample('M').mean()[1:]  # Resample to monthly averages
avg_monthly_humidity = monthly_data.reset_index()

plt.figure(figsize=(20, 10))
plt.title("Monthly Average Humidity From 2018-2024")
sns.lineplot(x=avg_monthly_humidity['Timestamp'], y=avg_monthly_humidity['Humidity'], marker='o', color='green')
plt.xlabel("Month")
plt.ylabel("Humidity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Bar plot for monthly averages
plt.figure(figsize=(20, 10))
plt.title("Humidity - Monthly Averages (2018-2024)")
sns.barplot(x=avg_monthly_humidity['Timestamp'].dt.month, y=avg_monthly_humidity['Humidity'], palette="mako")
plt.xlabel("Month")
plt.ylabel("Humidity")
plt.show()

# 4. Yearly averages
yearly_data = all_data.resample('Y').mean()
yearly_data.reset_index(inplace=True)

plt.figure(figsize=(20, 10))
plt.title("Yearly Average Humidity From 2018-2024")
sns.lineplot(x=yearly_data['Timestamp'], y=yearly_data['Humidity'], marker='o', color='purple')
plt.xlabel("Year")
plt.ylabel("Humidity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Interactive Plotly graph
fig = px.line(all_data, x=all_data.index, y='Humidity', title="Humidity From 2018-2024")
fig.update_traces(mode='lines+markers', line=dict(color='red'))
fig.show()

# 6. SARIMA forecast visualization
plt.figure(figsize=(20, 10))
plt.plot(all_data.index, all_data['Humidity'], label='Observed', color='blue')
plt.plot(predicted_2024['Timestamp'], predicted_2024['Humidity'], label='Forecasted', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title("SARIMA Forecast for Humidity")
plt.legend()
plt.show()
