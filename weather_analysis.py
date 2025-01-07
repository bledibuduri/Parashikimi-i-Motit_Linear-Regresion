import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Leximi i të dhënave nga datasetet e ndryshme
air_pressure_data = pd.read_csv('AirPreasure/airpreasure_data_2018_2024.csv')
humidity_data = pd.read_csv('Humidity/humidity_data_2018_2024.csv')
temperature_data = pd.read_csv('Temperature/temperature_data_2018_2024.csv')
windspeed_data = pd.read_csv('Windspeed/windspeed_data_2018_2024.csv')

# Kontrolloni kolonat e datasetit për të siguruar që janë të njëjta
print(air_pressure_data.columns)
print(humidity_data.columns)
print(temperature_data.columns)
print(windspeed_data.columns)

# Bashkimi i të dhënave në një dataset të vetëm
# Ne supozojmë se këto datasetet kanë një kolonë të përbashkët 'Timestamp'
# Sigurohuni që 'Timestamp' të jetë formatuar si datetime
air_pressure_data['Timestamp'] = pd.to_datetime(air_pressure_data['Timestamp'])
humidity_data['Timestamp'] = pd.to_datetime(humidity_data['Timestamp'])
temperature_data['Timestamp'] = pd.to_datetime(temperature_data['Timestamp'])
windspeed_data['Timestamp'] = pd.to_datetime(windspeed_data['Timestamp'])


# Bashko të dhënat duke përdorur Timestamp si çelës
merged_data = pd.merge(air_pressure_data, humidity_data, on='Timestamp', how='inner')
merged_data = pd.merge(merged_data, temperature_data, on='Timestamp', how='inner')
merged_data = pd.merge(merged_data, windspeed_data, on='Timestamp', how='inner')

# Shikoni disa rreshta të datasetit të bashkuar
print(merged_data.head())

# Analiza e korelacionit për kolonat numerike
correlation_matrix = merged_data.corr()

# Vizualizimi i matrikës së korelacionit me heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix for Air Pressure, Humidity, Temperature, and Windspeed')
plt.show()
