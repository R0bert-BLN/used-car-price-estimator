import glob
import os
import pandas as pd

DATA_DIR = "../data"
COLUMNS = ["model", "year", "price", "transmission", "mileage", "fuelType"]

def load_data():
    tables = []

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
       
        df = pd.read_csv(path)
        df = df[COLUMNS].copy()

        df["brand"] = name.capitalize()
        df["model"] = df["model"].str.strip()

        tables.append(df)
        
        print(f"{name}.csv: {len(df)} rows")

    if not tables:
        raise FileNotFoundError("No CSV files found in the data/ folder")

    df = pd.concat(tables, ignore_index=True)

    return df
