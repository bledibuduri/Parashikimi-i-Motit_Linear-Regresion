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

# Shto kolona për vitin dhe muajin
merged_data['Year'] = merged_data['Timestamp'].dt.year
merged_data['Month'] = merged_data['Timestamp'].dt.month

# Grupimi sipas vitit dhe muajit dhe llogaritja e temperaturës dhe lagështisë mesatare për secilën kategori
monthly_avg = merged_data.groupby(['Year', 'Month'])[['Temperature', 'Humidity']].mean()

# Vizualizimi i temperaturës dhe lagështisë mesatare për secilën muaj brenda një grafiku
plt.figure(figsize=(12, 8))


handles, labels = [], []

# Itero për secilin vit dhe krijo një linjë për temperaturën dhe pastaj për lagështinë
for year in monthly_avg.index.get_level_values('Year').unique():
    year_data = monthly_avg.loc[year]
    
    # Temperatura
    temp_line, = plt.plot(year_data.index, year_data['Temperature'], marker='o', label=f'Temperature ({year} - °C)')
    handles.append(temp_line)
    labels.append(f'Temperature ({year} - °C)')

# Itero përsëri për të shtuar linjat për lagështinë
for year in monthly_avg.index.get_level_values('Year').unique():
    year_data = monthly_avg.loc[year]
    
    # Lagështia
    hum_line, = plt.plot(year_data.index, year_data['Humidity'], marker='o', label=f'Humidity ({year} - %)')
    handles.append(hum_line)
    labels.append(f'Humidity ({year} - %)')

# Titulli dhe etiketat
plt.title('Average Monthly Temperature and Humidity by Year', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Value', fontsize=12)

# Shtoni legjendën me renditje të personalizuar
plt.legend(handles, labels, loc='upper left')

# Shfaq grafikun
plt.tight_layout()
plt.show()

