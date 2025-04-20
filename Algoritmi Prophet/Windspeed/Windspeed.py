import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Read the dataset
data = pd.read_csv('Windspeed/windspeed.csv')

# Check for null values
null_values = data['WindSpeed'].isnull().sum()
print(f"Number of null values in 'WindSpeed' column: {null_values}")

# Convert 'Timestamp' column to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d.%m.%Y %H:%M:%S')

# Prepare the data for Prophet
prophet_data = data[['Timestamp', 'WindSpeed']].rename(columns={'Timestamp': 'ds', 'WindSpeed': 'y'})

# Create and fit the Prophet model
model = Prophet()
model.fit(prophet_data)

# Generate future dates (hourly for 2024)
future_dates = model.make_future_dataframe(periods=365 * 24, freq='H')
forecast = model.predict(future_dates)

# Save predictions for 2024 in a DataFrame
predicted_2024 = forecast[['ds', 'yhat']].rename(columns={'ds': 'Timestamp', 'yhat': 'WindSpeed'})

# Combine existing and predicted data
all_data = pd.concat([prophet_data.rename(columns={'ds': 'Timestamp', 'y': 'WindSpeed'}), predicted_2024], ignore_index=True)

# Save combined data to a CSV file
all_data.to_csv('Windspeed/windspeed_data_2018_2024.csv', index=False)
print("Data saved to 'windspeed_data_2018_2024.csv'")

# 1. Plot the entire dataset (2018–2024)
plt.figure(figsize=(20, 10))
plt.plot(all_data['Timestamp'], all_data['WindSpeed'], label='WindSpeed', color='blue')
plt.title("WindSpeed From 2018-2024")
plt.xlabel("Year")
plt.ylabel("WindSpeed")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Calculate and plot monthly averages
all_data['Timestamp'] = pd.to_datetime(all_data['Timestamp'], utc=True, errors='coerce')
monthly_data = all_data.set_index('Timestamp').resample('M').mean()[1:]
avg_monthly_windspeed = monthly_data.reset_index()

plt.figure(figsize=(20, 10))
plt.title("Monthly Average WindSpeed From 2018-2024")
sns.lineplot(x=avg_monthly_windspeed['Timestamp'], y=avg_monthly_windspeed['WindSpeed'], marker='o', color='green')
plt.xlabel("Month")
plt.ylabel("WindSpeed")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Bar plot for monthly averages
plt.figure(figsize=(20, 10))
plt.title("TWindSpeed - Monthly Averages (2018-2024)")
sns.barplot(x=avg_monthly_windspeed['Timestamp'].dt.month, y=avg_monthly_windspeed['WindSpeed'], palette="mako")
plt.xlabel("Month")
plt.ylabel("WindSpeed")
plt.show()

# 4. Yearly averages
yearly_data = all_data.set_index('Timestamp').resample('Y').mean()
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
fig = px.line(all_data, x='Timestamp', y='WindSpeed', title="WindSpeed From 2018-2024")
fig.update_traces(mode='lines+markers', line=dict(color='red'))
fig.show()

# 6. Correlation heatmap (optional, based on other columns in your data)
# Check if other columns exist for correlation
if 'Other_Column' in data.columns:
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

# Prophet forecast visualization
fig2 = model.plot(forecast)
plt.title("Prophet Forecast for WindSpeed")
plt.show()
