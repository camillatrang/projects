import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import tarfile
import urllib.request
import io

# =========================
# LOAD DATA
# =========================
url = "https://github.com/Hvass-Labs/weather-denmark/raw/master/weather-denmark.tar.gz"

response = urllib.request.urlopen(url)
file_like_object = io.BytesIO(response.read())

with tarfile.open(fileobj=file_like_object, mode="r:gz") as tar:
    csv_file = tar.extractfile("weather-denmark.csv")
    df = pd.read_csv(csv_file)

# =========================
# PREPARE DATA
# =========================
df["DateTime"] = pd.to_datetime(df["DateTime"])
df["hour"] = df["DateTime"].dt.hour

# Simulation of heat demand based on temperature, wind speed, and hour of day
df["heat_demand"] = (
    25
    - df["Temp"]
    + 0.5 * df["WindSpeed"]
)

df = df.dropna()

X = df[["Temp", "WindSpeed", "hour"]]
y = df["heat_demand"]

model = LinearRegression()
model.fit(X, y)

# =========================
# STREAMLIT UI
# =========================
st.title("Heat Demand Predictor")

st.write(
    "Simple ML model predicting district heating demand based on temperature, wind, and time of day"
)

# Inputs
temp = st.slider("Temperature (°C)", -10, 30, 5)
wind = st.slider("Wind Speed", 0, 20, 5)
hour = st.slider("Hour of day", 0, 23, 12)

# Input display
st.write("Inputs:")
st.write(f"Temp: {temp}, Wind: {wind}, Hour: {hour}")

# Prediction
prediction = model.predict([[temp, wind, hour]])

# Output
st.subheader("Prediction")
st.metric("Heat Demand", f"{prediction[0]:.2f}")

# Model insight
st.subheader("Model insight")
st.write("Coefficients:")
st.json(dict(zip(["Temp", "Wind", "Hour"], model.coef_)))