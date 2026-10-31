import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

MODEL_PATH = "../models/model.joblib"
INFO_PATH = "../models/info.json"
KM_TO_MILES = 1.609

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

with open(INFO_PATH) as file:
    info = json.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    price = None
    error = None
    values = {}

    if request.method == "POST":
        values = request.form.to_dict()

        try:
            brand = values["brand"]
            car_model = values["model"]
            year = int(values["year"])
            km = float(values["km"])

            if car_model not in info["models_by_brand"].get(brand, []):
                raise ValueError
            
            if not (1995 <= year <= info["reference_year"]) or km < 0:
                raise ValueError

            car = pd.DataFrame([{
                "brand": brand,
                "model": car_model,
                "fuelType": values["fuel"],
                "transmission": values["transmission"],
                "age": info["reference_year"] - year,
                "mileage": km / KM_TO_MILES,
            }])

            price = max(0, round(float(model.predict(car)[0])))
        except (KeyError, ValueError):
            error = (
                f"Invalid data. The year must be between 1995 and ",
                f"{info['reference_year']}, and the mileage a positive number.",
            )

    return render_template("index.html", info=info, price=price, error=error, values=values)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
