import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# =========================
# LOAD & PREPARE DATA
# =========================
df = pd.read_csv("https://raw.githubusercontent.com/Hvass-Labs/weather-denmark/master/weather-denmark.csv")

df["DateTime"] = pd.to_datetime(df["DateTime"])
df["hour"] = df["DateTime"].dt.hour

df["heat_demand"] = (
    25
    - df["Temp"]
    + 0.5 * df["WindSpeed"]
    + np.random.normal(0, 2, len(df))
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

st.write("Predict heat demand based on weather conditions")


# Inputs
temp = st.slider("Temperature (°C)", -10, 30, 5)
wind = st.slider("Wind Speed", 0, 20, 5)
hour = st.slider("Hour of day", 0, 23, 12)


# Prediction
prediction = model.predict([[temp, wind, hour]])

st.subheader("Prediction")
st.write(f"Estimated heat demand: {prediction[0]:.2f}")