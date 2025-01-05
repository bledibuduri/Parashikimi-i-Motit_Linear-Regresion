# Advanced Data Modeling - Weather Data Prediction

## Overview
This project involves the prediction of weather-related data, specifically air pressure, humidity, temperature, and wind speed, for the year 2024 using historical data from the Republic of Kosovo. The goal is to apply advanced data modeling techniques, focusing on Linear Regression, to forecast values for each of the weather parameters on an hourly basis.

The project is being conducted as part of the **Master's program** in **Advanced Data Modeling and Databases** at the **University of Prizren**, under the supervision of **Prof. Asoc. Dr. Zirije Hasani**. The data has been provided by the course instructor, Prof. Hasani, to facilitate the completion of this research project.

## Project Components

### 1. Data Collection
The dataset contains hourly weather data from 2017 to 2023 for the following parameters:
- **Air Pressure**
- **Humidity**
- **Temperature**
- **Wind Speed**

The data is provided in CSV files, with each file containing two columns:
- `Timestamp`: The date and time of the data entry.
- The respective weather parameter (`Air_Pressure`, `Humidity`, `Temperature`, `WindSpeed`).

### 2. Data Preprocessing
The data underwent preprocessing steps, including:
- **Timestamp Conversion**: The `Timestamp` column was converted into a numerical format (ordinal) to facilitate prediction using machine learning models.
- **Cleaning**: Weather parameters such as `WindSpeed` were cleaned to remove any unnecessary units (e.g., `m/s` for wind speed).

### 3. Model Development
For each weather parameter, the following steps were taken:
1. **Data Splitting**: The dataset was split into training and testing sets (80% training, 20% testing).
2. **Linear Regression**: A Linear Regression model was trained using the training data and evaluated on the testing set.
3. **Prediction for 2024**: The model was used to predict the weather data for each hour of 2024, resulting in a forecasted dataset for the entire year.

### 4. Data Visualization
The results were visualized using **Matplotlib** to compare the original data with the predicted values for the year 2024.

### 5. Output
The predictions for each parameter (Air Pressure, Humidity, Temperature, and Wind Speed) were saved in separate CSV files:
- `predicted_air_pressure_2024.csv`
- `predicted_humidity_2024.csv`
- `predicted_temperature_2024.csv`
- `predicted_windspeed_2024.csv`

Additionally, graphical representations of the original data versus predicted values for 2024 were created.

## Project Files

- `airpreasure.csv`: Original data for air pressure from 2017 to 2023.
- `humidity.csv`: Original data for humidity from 2017 to 2023.
- `temperature.csv`: Original data for temperature from 2017 to 2023.
- `windspeed.csv`: Original data for wind speed from 2017 to 2023.
- `predicted_air_pressure_2024.csv`: Predicted air pressure values for 2024.
- `predicted_humidity_2024.csv`: Predicted humidity values for 2024.
- `predicted_temperature_2024.csv`: Predicted temperature values for 2024.
- `predicted_windspeed_2024.csv`: Predicted wind speed values for 2024.
- 
### Scripts Overview:

- `temperature_prediction.py`: For predicting temperature in 2024.
- `humidity_prediction.py`: For predicting humidity in 2024.
- `windspeed_prediction.py`: For predicting wind speed in 2024.
- `airpressure_prediction.py`: For predicting air pressure in 2024.

## Requirements
- Python 3.11+
- pandas
- scikit-learn
- matplotlib

To install the required packages, run:

```bash
pip install pandas scikit-learn matplotlib
