# Project: Predicting Heat Demand from Weather Data

## Overview
This project models heat demand based on weather data.  
The goal is to understand how temperature and wind affect heating needs and to build a simple prediction model.


## Data
The dataset contains weather observations with:

- DateTime  
- Temperature  
- Wind speed  
- Pressure  
- City  

# Weather-data for Denmark 1980-2018

[Original repository on GitHub](https://github.com/Hvass-Labs/weather-denmark)


## Data Overview

* This data-set is intended for use in research on Machine Learning and Time-Series Prediction.
* The weather-data covers the period between 1980-2018 for five Danish cities: Aalborg, Aarhus, Esbjerg, Odense and Roskilde.
* Each data-point measures the temperature (Celcius), the barometric pressure (hecto-pascal or milli-bar), the wind-speed (meters per second), and the wind-direction (angular degrees).
* Some data is partially missing, e.g. periods of the barometric pressure in some cities.
* Some errors have been discovered in the data, e.g. a few temperature-measurements are clearly wrong.


## Source

The raw weather-data was originally obtained from the [National Climatic Data Center (NCDC) in USA](https://www7.ncdc.noaa.gov/CDO/cdoselect.cmd).

The raw data-file had some formatting problems that had to be corrected manually. This archive contains a cleaned-up version of the data-file, that is easier to load either as a CSV-file or as a so-called pickle-file for use in Pandas with Python.

## Method

### Preprocessing
- Converted DateTime to datetime format  
- Removed missing values  

### Feature Engineering
Heat demand is simulated as:

heat_demand = 25 - Temp + 0.5 * WindSpeed + noise

Where:
- Lower temperature → higher demand  
- Higher wind speed → more heat loss  
- Noise makes the data more realistic  

Additional feature:
- Hour of day (from DateTime)


## Model
A linear regression model is used.

Features:
- Temp  
- WindSpeed  
- Hour  

Target:
- heat_demand  

The data is split into training and test sets.


## Results
The model is evaluated using:

- Mean Squared Error (MSE)  
- R² score  

The predictions follow the overall trend of heat demand but are not perfect due to noise.


## Project Structure

weather-denmark-project/
- data/
  - weather-denmark.csv  
- src/
  - pipeline.py  
- README.md  



## Author
Camilla Trang