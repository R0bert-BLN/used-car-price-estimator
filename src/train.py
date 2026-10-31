import matplotlib

matplotlib.use("Agg")

import os
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from data_loader import load_data

MODEL_PATH = "../models/model.joblib"
INFO_PATH = "../models/info.json"
PLOTS_DIR = "../plots"
MODELS_DIR = "../models"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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
print(f"Training: {len(x_train)} rows")
print(f"Test: {len(x_test)} rows\n")

preprocessing = ColumnTransformer([
    ("text", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("numbers", "passthrough", NUMERIC)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=1,
    ),
}

results = {}
pipelines = {}
all_predictions = {}

for name, model in models.items():
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", model)])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results[name] = {
        "MAE": round(mae),
        "RMSE": round(rmse),
        "R2": round(r2, 3),
    }

    pipelines[name] = pipeline
    all_predictions[name] = predictions

    print(f"{name}:  MAE = {mae:,.0f} £   RMSE = {rmse:,.0f} £   R2 = {r2:.3f}")

best_model = min(results, key=lambda name: results[name]["MAE"])
print(f"Best model: {best_model}")
predictions = all_predictions[best_model]

plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, s=4, alpha=0.3, color="#2a6f97")
limit = max(y_test.max(), predictions.max())
plt.plot([0, limit], [0, limit], color="red")
plt.title(f"actual vs predicted price")
plt.xlabel("Actual price (£)")
plt.ylabel("Predicted price (£)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/5_actual_vs_predicted.png")
plt.close()

joblib.dump(pipelines[best_model], MODEL_PATH, compress=3)

info = {
    "reference_year": REFERENCE_YEAR,
    "best_model": best_model,
    "results": results,
    "brands": sorted(df["brand"].unique().tolist()),
    "fuel_types": sorted(df["fuelType"].unique().tolist()),
    "transmissions": sorted(df["transmission"].unique().tolist()),
    "models_by_brand": {
        brand: sorted(group["model"].unique().tolist())
        for brand, group in df.groupby("brand")
    },
}

with open(INFO_PATH, "w") as file:
    json.dump(info, file, indent=2, ensure_ascii=False)
