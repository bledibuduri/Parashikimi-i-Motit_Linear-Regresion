import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Leximi i të dhënave nga datasetet e ndryshme
air_pressure_data = pd.read_csv('AirPreasure/airpreasure_data_2018_2024.csv')
humidity_data = pd.read_csv('Humidity/humidity_data_2018_2024.csv')
temperature_data = pd.read_csv('Temperature/temperature_data_2018_2024.csv')
windspeed_data = pd.read_csv('Windspeed/windspeed_data_2018_2024.csv')

# Sigurohuni që 'Timestamp' të jetë formatuar si datetime
air_pressure_data['Timestamp'] = pd.to_datetime(air_pressure_data['Timestamp'])
humidity_data['Timestamp'] = pd.to_datetime(humidity_data['Timestamp'])
temperature_data['Timestamp'] = pd.to_datetime(temperature_data['Timestamp'])
windspeed_data['Timestamp'] = pd.to_datetime(windspeed_data['Timestamp'])

# Bashkimi i të dhënave në një dataset të vetëm
merged_data = pd.merge(air_pressure_data, humidity_data, on='Timestamp', how='inner')
merged_data = pd.merge(merged_data, temperature_data, on='Timestamp', how='inner')
merged_data = pd.merge(merged_data, windspeed_data, on='Timestamp', how='inner')

# Krijimi i një periudhe kohore për secilin vit dhe muaj
merged_data['Year'] = merged_data['Timestamp'].dt.year
merged_data['Month'] = merged_data['Timestamp'].dt.month

# Grupimi sipas vitit dhe muajit dhe llogaritja e temperaturës dhe lagështisë mesatare
monthly_avg = merged_data.groupby(['Year', 'Month'])[['Temperature', 'Humidity']].mean().reset_index()

# Vizualizimi për çdo periudhë kohore (për çdo vit)
for year in range(2018, 2025):  # Për vitet nga 2018 deri në 2024
    plt.figure(figsize=(12, 6))
    plt.title(f"Average Temperature and Humidity in {year}")
    
    # Filtrimi i të dhënave për vitin aktual
    year_data = monthly_avg[monthly_avg['Year'] == year]

    # Vizualizimi i temperaturës dhe lagështisë për secilin muaj të vitit
    plt.plot(year_data['Month'], year_data['Temperature'], marker='o', color='tab:red', label='Temperature (°C)')
    plt.plot(year_data['Month'], year_data['Humidity'], marker='o', color='tab:blue', label='Humidity (%)')

    plt.xlabel('Month')
    plt.ylabel('Value')
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
