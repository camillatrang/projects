# Heat Demand Predictor

This project demonstrates a simple end-to-end machine learning workflow for predicting district heating demand based on weather conditions.

The model uses temperature, wind speed, and time of day to estimate heat demand and is deployed as an interactive web application using Streamlit.

## 🚀 Live App

https://projects-3jhx2exv9hpwxlgk5qc6ff.streamlit.app/


## Project Overview

The goal of this project is to:

- Load and process real-world weather data  
- Perform feature engineering  
- Train a machine learning model  
- Deploy an interactive prediction app  

The application allows users to adjust weather inputs and instantly see how heat demand changes.

## Data

The dataset contains historical weather observations from Denmark, including:

- DateTime  
- Temperature (°C)  
- Wind speed (m/s)  
- Pressure (hPa)  
- City  

The data covers the period 1980–2018 across multiple Danish cities (Aalborg, Aarhus, Esbjerg, Odense, and Roskilde).

**Note:**
- Some values are missing  
- Minor measurement errors exist in the dataset  

Source:  
https://github.com/Hvass-Labs/weather-denmark


## Technologies Used

- Python  
- Pandas & NumPy  
- scikit-learn  
- Streamlit  
- Matplotlib  


## Model

A **Linear Regression** model is used to predict heat demand based on:

- Temperature (°C)  
- Wind Speed  
- Hour of the day  

### Key insights:

- Lower temperature → higher heat demand  
- Higher wind speed → increased demand  
- Time of day had limited impact in this model  


## Data Processing

- Data is loaded from a compressed `.tar.gz` file hosted on GitHub  
- Timestamp is converted to datetime  
- Hour feature is extracted  
- Missing values are removed  


## How to Run Locally

```bash
git clone https://github.com/camillatrang/projects.git
cd projects/weather-denmark-project
pip install -r requirements.txt
streamlit run streamlit_app.py