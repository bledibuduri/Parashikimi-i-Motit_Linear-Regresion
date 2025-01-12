# Weather Data Prediction

## Overview
This project focuses on predicting weather-related data, specifically air pressure, humidity, temperature, and wind speed, for the year 2024 using historical data from the Republic of Kosovo. The primary goal is to use advanced data modeling techniques, such as Linear Regression, to forecast weather parameters on an hourly basis for 2024.

## Project Components

### 1. Data Collection
The dataset contains hourly weather data from 2017 to 2023 for the following parameters:
- Air Pressure
- Humidity
- Temperature
- Wind Speed

The data is stored in CSV files, where each file contains two columns:
- **Timestamp**: The date and time of the data entry.
- **Weather Parameter**: For example, `Air_Pressure`, `Humidity`, `Temperature`, `WindSpeed`.

### 2. Data Preprocessing
Before applying the linear regression model, the data is preprocessed:
- The `Timestamp` column is converted into the `datetime` type to extract year, month, day, and hour.
- Missing values in the dataset are checked and addressed (if any).
- The dataset is split into training and test sets for model evaluation.

### 3. Linear Regression Model
A linear regression model is used to predict the weather parameter (in this case, temperature) based on the extracted features (year, month, day, hour). The model is trained using the historical data and evaluated using metrics such as:
- **Mean Squared Error (MSE)**
- **R² Score**

### 4. Forecasting for 2024
After training the model, it is used to predict the weather data for each hour of the year 2024. A new dataset is generated for each hour from January 1st to December 31st, 2024. The forecasted temperature data is saved in a CSV file.

### 5. Data Visualization
Various visualizations are created to present the data and predictions:
- Interactive plots of the temperature trend from 2018 to 2024.
- Monthly and yearly average temperature visualizations using Seaborn and Matplotlib.
- Correlation analysis between different weather parameters.

### 6. Correlation Analysis
The correlation analysis is conducted by merging the datasets for different weather parameters (Air Pressure, Humidity, Temperature, and Wind Speed) and visualizing the relationships between them.

### 7. Conclusion
This project demonstrates how Linear Regression can be applied to predict weather data based on historical trends. The predictions for 2024 provide insights into the expected temperature patterns throughout the year. The project also includes various visualizations that help in understanding the trends and patterns in the weather data.

### 8. Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- plotly
- matplotlib
- seaborn

### 9. How to Run
1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`.
3. Place the historical weather data CSV files in the appropriate directory.
4. Run the script to generate predictions for 2024 and visualizations.

### 10. License
This project is licensed under the MIT License.
