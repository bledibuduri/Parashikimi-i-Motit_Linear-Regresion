import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

#Leximi i të dhënave (duhet të ketë një file CSV me kolonat 'Timestamp' dhe 'Humidity')
data = pd.read_csv('Temperature/temperature.csv')

#Shikojmë rreshtat e parë të datasetit
print(data.head())

#Shikojmë se sa rreshta dhe kolona kemi brenda datasetit 
print(data.shape)

#Përdorim data.info() për të printuar informacione mbi DataFrame
print(data.info())

#Përdorim describre() për të parë disa statistika bazike 
print(data.describe())

# Kontrollon vlerat null ne kolonen 'Temperature'
null_values = data['Temperature'].isnull().sum()
print(f"Number of null values in 'Temperature' column: {null_values}")

#Konvertimi i kolonës 'Timestamp' në tipin datetime dhe ekstraktimi i vitit dhe ores
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data['Day'] = data['Timestamp'].dt.day
data['Hour'] = data['Timestamp'].dt.hour

#Përcaktimi i variablave të pavarur dhe të varur
X = data[['Year', 'Month', 'Day', 'Hour']]  # Feature columns (Viti, Muaji, Dita, Ora)
y = data['Temperature']  # Target column (Lagështira)

#Ndara të dhënat në trajnime dhe test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Krijimi dhe trajnimi i modelit të regresionit linear
model = LinearRegression()
model.fit(X_train, y_train)

#Parashikimi mbi të dhënat e testuara
y_pred = model.predict(X_test)

#Vlerësimi i modelit
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print('R²:', r2_score(y_test, y_pred))

#Parashikimi për të gjithë vitin 2024 (për çdo orë nga 1 janari deri më 31 dhjetor)
# Krijimi i të dhënave për secilën orë të vitit 2024
date_rng = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:00:00', freq='H')
X_2024 = pd.DataFrame({
    'Year': date_rng.year,
    'Month': date_rng.month,
    'Day': date_rng.day,
    'Hour': date_rng.hour
})

# Parashikimi i lagështirës për vitin 2024
y_2024_pred = model.predict(X_2024)

#Ruajtja e parashikimeve për vitin 2024 në një DataFrame
predicted_2024 = pd.DataFrame({
    'Timestamp': date_rng,
    'Temperature': y_2024_pred
})

# Bashkimi i të dhënave ekzistuese me ato të parashikuara për vitin 2024
# Përsëri ruajmë vetëm kolonat Timestamp dhe Temperature
all_data = pd.concat([data[['Timestamp', 'Temperature']], predicted_2024], ignore_index=True)

#Ruajtja e dataset-it të ri në një file CSV
all_data.to_csv('Temperature/temperature_data_2018_2024.csv', index=False)

print("Të dhënat janë ruajtur në file-in temperature_data_2018_2024.csv")

#Shikojmë rreshtat e parë të datasetit
print(all_data.head())

#Shikojmë se sa rreshta dhe kolona kemi brenda datasetit 
print(all_data.shape)

#Përdorim data.info() për të printuar informacione mbi DataFrame
print(all_data.info())

#Përdorim describre() për të parë disa statistika bazike 
print(all_data.describe())

# Krijoni grafik interaktiv me plotly
fig = px.line(all_data, x='Timestamp', y='Temperature', title="Temperature From 2018-2024")

# Personalizoni grafikun (opsional)
fig.update_traces(mode='lines+markers', line=dict(color='red'))

# Shfaq grafikun
fig.show()

all_data['Timestamp'] = pd.to_datetime(all_data['Timestamp'], utc=True, errors='coerce')
print(all_data['Timestamp'].dtype)
monthly_data = all_data.set_index('Timestamp').resample('M').mean()[1:]
yearly_data = all_data.set_index('Timestamp').resample('Y').mean()[1:]
print(monthly_data.head())
print(yearly_data)

plt.figure(figsize=(20, 10))
plt.title("Temperature From 2018-2024")
sns.lineplot(x=data['Timestamp'], y=data['Temperature'], marker='o', color='red')
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.xticks(rotation=45)  # Rrotullon etiketat e boshtit X për lehtësim leximi
plt.tight_layout()  # Siguron që grafiku të mos prerë ndonjë etiketë
#plt.show()  # Shfaq grafikun

avg_monthly_temperature = pd.DataFrame(all_data.groupby([all_data["Timestamp"].dt.month])["Temperature"].mean())
avg_monthly_temperature.reset_index(inplace=True)

plt.figure(figsize=(20,10))
plt.title("Monthly Average Temperature From 2018-2024")
sns.lineplot(x=avg_monthly_temperature['Timestamp'],y=avg_monthly_temperature['Temperature'])
plt.xlabel("Month")
#plt.show()

plt.figure(figsize=(20,10))
plt.title("Temperatura Mesatare Mujore Nga 2018-2024")
sns.barplot(x=avg_monthly_temperature['Timestamp'],y=avg_monthly_temperature['Temperature'], palette="mako")
plt.xlabel("Month")
plt.show()

#Analiza Korelacionit
#data.corr(method = 'pearson')

#plt.figure(figsize=(12,10))
#plt.title("Correlation between all columns")
#sns.heatmap(data= data.corr(), cmap="rocket", annot=True)
#plt.show()