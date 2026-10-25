import os
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from data_loader import load_data

os.makedirs("../plots", exist_ok=True)

CATEGORICAL = ["brand", "model", "fuelType", "transmission"]
NUMERIC = ["age", "mileage"]
TARGET = "price"

print ("Files loaded:")
df = load_data()
print("\nRows at start: ", len(df))

df = df.dropna()
df = df.drop_duplicates()

REFERENCE_YEAR = 2020
df = df[(df["year"] >= 1995) & (df["year"] <= REFERENCE_YEAR)]
df = df[df["price"] < df["price"].quantile(0.99)]

df["age"] = REFERENCE_YEAR - df["year"]
print("Rows after cleaning: ", len(df))

x = df[CATEGORICAL + NUMERIC]
y = df[TARGET]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Training: {len(x_train)} rows | Test: {len(x_test)} rows")

preprocessing = ColumnTransformer([
    ("text", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("numbers", "passthrough", NUMERIC)
])

pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("model", LinearRegression())
])

pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print(f"\nLinear Regression:  MAE = {mae:,.0f} £   RMSE = {rmse:,.0f} £   R2 = {r2:.3f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, s=4, alpha=0.3, color="#2a6f97")

limit = max(y_test.max(), predictions.max())

plt.plot([0, limit], [0, limit], color="red")
plt.title("Linear Regression: actual vs predicted price")
plt.xlabel("Actual price (£)")
plt.ylabel("Predicted price (£)")
plt.tight_layout()
plt.savefig("../plots/5_actual_vs_predicted_linear.png")
plt.close()

print("\nPlot saved: plots/5_actual_vs_predicted_linear.png")
