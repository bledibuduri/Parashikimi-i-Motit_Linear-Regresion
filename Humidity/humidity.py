import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load existing data
df = pd.read_csv('Humidity/humidity_filled.csv')

# Check for null values in the 'humidity' column
null_values = df['Humidity'].isnull().sum()
#print(f"Number of null values in 'humidity' column: {null_values}")

# Konvertoni Timestamp në format datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M')

# Shndërrojeni Timestamp në një numër (në këtë rast ditë numerike)
df['Timestamp'] = df['Timestamp'].apply(lambda x: x.toordinal())

# Sigurohuni që Humidity është një numër dhe hiqni çdo karakter të panevojshëm
df['Humidity'] = df['Humidity'].astype(float)

# Pjesëtimi i të dhënave në variabla të pavarur dhe të varur
X = df[['Timestamp']]  # variablat e pavarur (Timestamp)
y = df['Humidity']  # variabli i varur (Humidity)

# Calculate the mean value of the 'humidity' column, excluding null values
mean_humidity = round(df['Humidity'].mean(),2)

# Fill null values with the calculated mean
df['Humidity'].fillna(mean_humidity, inplace=True)

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
print(df['Timestamp'].dtype)
monthly_data = df.set_index('Timestamp').resample('M').mean()[1:]
yearly_data = df.set_index('Timestamp').resample('Y').mean()[1:]
print(monthly_data.head())
print(yearly_data)

#plt.figure(figsize=(20, 10))
#plt.title("Humidity From 2018-2023")
#sns.lineplot(x=df['Timestamp'], y=df['Humidity'], marker='o', color='red')
#plt.xlabel("Year")
#plt.ylabel("Humidity")
#plt.xticks(rotation=45)  # Rrotullon etiketat e boshtit X për lehtësim leximi
#plt.tight_layout()  # Siguron që grafiku të mos prerë ndonjë etiketë
#plt.show()  # Shfaq grafikun

# Krijoni grafik interaktiv me plotly
fig = px.line(df, x='Timestamp', y='Humidity', title="Humidity From 2018-2023")

# Personalizoni grafikun (opsional)
fig.update_traces(mode='lines+markers', line=dict(color='red'))

# Shfaq grafikun
#fig.show()

avg_monthly_humidity = pd.DataFrame(df.groupby([df["Timestamp"].dt.month])["Humidity"].mean())
avg_monthly_humidity.reset_index(inplace=True)
#plt.figure(figsize=(20,10))
#plt.title("Monthly Average Humidity From 2018-2023")
#sns.barplot(x=avg_monthly_humidity['Timestamp'],y=avg_monthly_humidity['Humidity'], palette="mako")
#plt.xlabel("Month")
#plt.show()

#plt.figure(figsize=(20,10))
#plt.title("Monthly Average Humidity From 2018-2023")
#sns.lineplot(x=avg_monthly_humidity['Timestamp'],y=avg_monthly_humidity['Humidity'])
#plt.xlabel("Month")
#plt.show()

# Pjesëtimi i të dhënave në trajnimin dhe testimin
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Krijimi i modelit Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Parashikimi i të dhënave të testuara
y_pred = model.predict(X_test)

# Vlerësimi i modelit
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print('R²:', r2_score(y_test, y_pred))

# Parashikimi për vitin 2024 në çdo orë
# Krijoni një seri datash për vitin 2024 në intervale orësh
date_range = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:59:59', freq='H')

# Shndërrojeni datat në format ordinal
date_ordinals = date_range.to_series().apply(lambda x: x.toordinal()).values.reshape(-1, 1)

# Parashikimi për secilën datë të vitit 2024
predicted_humidity = model.predict(date_ordinals)

# Create a DataFrame for 2024 predictions
predicted_df = pd.DataFrame({
    'Timestamp': date_range,
    'Predicted_Humidity': predicted_humidity
})


# Ruani rezultatet në një skedar të ri CSV
predicted_df.to_csv('Humidity/predicted_humidity_2024_hourly.csv', index=False)

# Bashko të dhënat nëpërmjet rreshtave (row-wise concatenation)
df_combined = pd.concat([df, predicted_df], ignore_index=True)
# Shfaq rezultatet e bashkuara
print(df_combined.head())