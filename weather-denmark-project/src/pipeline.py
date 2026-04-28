import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("https://raw.githubusercontent.com/Hvass-Labs/weather-denmark/master/weather-denmark.csv")

print(df.head())
print(df.columns)


# =========================
# 2. UNDERSTAND DATA
# =========================
print(df.info())
print(df.describe())


# =========================
# 3. PREPROCESSING
# =========================

# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Extract hour from DateTime
df["hour"] = df["DateTime"].dt.hour

# Create heat demand with noise (feature engineering)
df["heat_demand"] = (
    25 
    - df["Temp"] 
    + 0.5 * df["WindSpeed"] 
    + np.random.normal(0, 2, len(df))
)

# Remove missing values
df = df.dropna()

print(df[["Temp", "WindSpeed", "heat_demand"]].head())


# =========================
# 4. VISUALIZATION
# =========================
df_sample = df.head(200)

plt.plot(df_sample["DateTime"], df_sample["Temp"], label="Temp")
plt.plot(df_sample["DateTime"], df_sample["heat_demand"], label="Heat Demand")

plt.legend()
plt.xticks(rotation=45)
plt.title("Temperature vs Heat Demand")
plt.show()


# =========================
# 5. MACHINE LEARNING
# =========================

# Features and target
X = df[["Temp", "WindSpeed"]]
y = df["heat_demand"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on test data
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)

# Print metrics
print("MSE:", mse)
print("R2:", r2_score(y_test, y_pred))


# =========================
# 6. EVALUATION PLOT
# =========================
plt.plot(y_test.values[:100], label="Actual")
plt.plot(y_pred[:100], label="Prediction")

plt.legend()
plt.title("Actual vs Predicted Heat Demand")
plt.show()