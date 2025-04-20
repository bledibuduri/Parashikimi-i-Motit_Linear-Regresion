import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Read the dataset
data = pd.read_csv('C:/Users/Diellza/Downloads/WeatherPrediction_YearsCompareGraphics/WeatherPrediction_YearsCompareGraphics/WindSpeed/windspeed.csv')

# Convert 'Timestamp' column to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

# Set 'Timestamp' as the index
data.set_index('Timestamp', inplace=True)

# Check for null values before filling
print(f"Number of null values in 'WindSpeed' column before filling: {data['WindSpeed'].isnull().sum()}")

# Fill missing values using time-based interpolation
data['WindSpeed'] = data['WindSpeed'].interpolate(method='time')

# Check for null values after filling
print(f"Number of null values in 'WindSpeed' column after filling: {data['WindSpeed'].isnull().sum()}")

# Fit the SARIMA model
sarima_model = SARIMAX(data['WindSpeed'],
                       order=(1, 1, 1),
                       seasonal_order=(0, 1, 0, 24),  # Simplified seasonal component
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
predicted_2024 = pd.DataFrame({'Timestamp': future_dates, 'WindSpeed': forecast_mean})
predicted_2024['WindSpeed'] = predicted_2024['WindSpeed'].fillna(data['WindSpeed'].mean())  # Fill any NaN values with mean

# Combine existing and predicted data
all_data = pd.concat([data[['WindSpeed']], predicted_2024.set_index('Timestamp')], ignore_index=False)

# Save combined data to a CSV file
output_path = 'C:/Users/Diellza/Downloads/WeatherPrediction_YearsCompareGraphics/WeatherPrediction_YearsCompareGraphics/WindSpeed/windspeed_data_2018_2024_sarima.csv'
all_data.to_csv(output_path, index=True)
print(f"Data saved to '{output_path}'")

# ---- VISUALIZATIONS ----

# 1. Plot entire dataset (2018–2024)
plt.figure(figsize=(20, 10))
plt.plot(all_data.index, all_data['WindSpeed'], label='WindSpeed', color='blue')
plt.title("WindSpeed From 2018-2024")
plt.xlabel("Year")
plt.ylabel("WindSpeed")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Calculate and plot monthly averages
# Calculate monthly averages
monthly_data = all_data['WindSpeed'].resample('M').mean()
avg_monthly_windspeed = monthly_data.reset_index()

# Plot monthly averages
plt.figure(figsize=(15, 7))
plt.plot(monthly_data.index, monthly_data, marker='o', linestyle='-', color='orange')
plt.title("Monthly Average WindSpeed (2018-2024)")
plt.xlabel("Month")
plt.ylabel("WindSpeed (Average)")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# 3. Bar plot for monthly averages
plt.figure(figsize=(20, 10))
plt.title("WindSpeed - Monthly Averages (2018-2024)")
sns.barplot(x=avg_monthly_windspeed['Timestamp'].dt.month, y=avg_monthly_windspeed['WindSpeed'], palette="mako")
plt.xlabel("Month")
plt.ylabel("WindSpeed")
plt.show()

# 4. Yearly averages
yearly_data = all_data.resample('Y').mean()
yearly_data.reset_index(inplace=True)

plt.figure(figsize=(20, 10))
plt.title("Yearly Average WindSpeed From 2018-2024")
sns.lineplot(x=yearly_data['Timestamp'], y=yearly_data['WindSpeed'], marker='o', color='purple')
plt.xlabel("Year")
plt.ylabel("WindSpeed")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Interactive Plotly graph
import plotly.io as pio
pio.renderers.default = 'browser'  # Ensure the graph opens in the browser

fig = px.line(all_data.reset_index(), x='Timestamp', y='WindSpeed', title="WindSpeed From 2018-2024")
fig.update_traces(mode='lines+markers', line=dict(color='red'))
fig.show()

# 6. SARIMA forecast visualization
plt.figure(figsize=(20, 10))
plt.plot(all_data.index, all_data['WindSpeed'], label='Observed', color='blue')
plt.plot(predicted_2024['Timestamp'], predicted_2024['WindSpeed'], label='Forecasted', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title("SARIMA Forecast for WindSpeed")
plt.legend()
plt.show()
